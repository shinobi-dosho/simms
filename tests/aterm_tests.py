"""Tests for the image-domain a-term correction (simms.skymodel.aterms).

The load-bearing assertion: the gridded a-term path must agree with the exact
per-component beam kernels (``predict_vis_beam``/``predict_vis_jones``) to
gridder accuracy, because its time blend is algebraically identical to the
kernels' per-antenna PA interpolation and its frequency knots are placed at
every channel when ``freq_tol == 0``. The reference predictions here therefore
share *no* image-domain code with the path under test.
"""

import numpy as np
import pytest
from astropy.io import fits

from simms.skymodel.aterms import _MapCache, aterm_cache_min_gib, attach_fits_aterm, select_freq_knots
from simms.skymodel.beams import BeamProvider, CosineTaperBeam, FitsBeamProvider, JimBeamProvider
from simms.skymodel.fits_skies import component_sky_from_fits_dft, predict_fits_block, prepare_fits_sky
from simms.skymodel.kernels import is_uniform_grid
from simms.skymodel.mstools import PreparedSky, attach_beam, predict_block, to_full_corr

from . import InitTest
from .predict_fits_tests import DEC0, RA0, make_header

# MeerKAT-ish site (radians); any site works, both paths share it.
LON, LAT = np.radians(21.443), np.radians(-30.712)
T_START = 60000.0 * 86400.0  # MS TIME seconds (MJD 60000)
CELL_DEG = 2.0 / 60.0  # 2 arcmin: a 64-pix image spans ~2 deg, well into the taper
NPIX = 64
PA_STEP = 1.0


class InitThisTest(InitTest):
    pass


@pytest.fixture
def params():
    return InitThisTest()


def observation(ntimes=5, nant=4, seed=7):
    """Times, baselines and uvw for a small heterogeneous observation."""
    rng = np.random.default_rng(seed)
    utimes = T_START + np.linspace(0.0, 2 * 3600.0, ntimes)
    ant1, ant2 = np.triu_indices(nant, k=1)
    nbl = ant1.size
    times = np.repeat(utimes, nbl)
    antenna1 = np.tile(ant1, ntimes)
    antenna2 = np.tile(ant2, ntimes)
    uvw = rng.normal(0.0, 400.0, (times.size, 3))
    duration = float(utimes[-1] - utimes[0]) + 1.0
    return times, antenna1, antenna2, uvw, duration


def hetero_beams():
    """Two distinct analytic beam types (MeerKAT + MeerKAT-Extension dishes)."""
    providers = [
        JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-MA-L-JIM-2026")),
        JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-EA-L-JIM-2026")),
    ]
    ant_type = np.array([0, 0, 1, 1], dtype=np.int64)
    type_is_altaz = np.array([True, True])
    return providers, ant_type, type_is_altaz


PIXELS = [(32, 32, 1.0), (20, 40, 2.0), (45, 25, 0.5)]  # (ra_pix, dec_pix, flux)


def write_flat_image(params, pixels=PIXELS, nstokes=1, stokes_values=None):
    """A FITS image with a few point pixels; returns its path."""
    if nstokes == 1:
        data = np.zeros((NPIX, NPIX), dtype=np.float64)
        for ira, idec, flux in pixels:
            data[idec, ira] = flux
    else:
        data = np.zeros((nstokes, NPIX, NPIX), dtype=np.float64)
        for (ira, idec, _), values in zip(pixels, stokes_values, strict=True):
            data[:, idec, ira] = values
    header = make_header(NPIX, nstokes=nstokes, nchan=1, cell=CELL_DEG)
    path = params.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data, header=header).writeto(path, overwrite=True)
    return path


def reference_component_sky(prepared_fits, bmat, ncorr):
    """A PreparedSky whose components sit exactly on the FITS model's pixels."""
    i_pix = np.array([p[0] for p in PIXELS], dtype=np.float64)
    j_pix = np.array([p[1] for p in PIXELS], dtype=np.float64)
    lmn = prepared_fits.grid.pixel_lmn(i_pix, j_pix)
    nsrc = lmn.shape[0]
    return PreparedSky(
        lmn=lmn,
        gauss_shape=np.zeros((nsrc, 3)),
        is_gauss=np.zeros(nsrc, dtype=np.bool_),
        bmat=bmat,
        lightcurve=np.ones((nsrc, 1)),
        unique_times=None,
        freqs=prepared_fits.chan_freqs,
        uniform_freqs=is_uniform_grid(prepared_fits.chan_freqs),
        ncorr=ncorr,
        polarisation=True,
    )


def attach_kwargs(duration, providers, ant_type, type_is_altaz, **overrides):
    kwargs = dict(
        ant_type=ant_type,
        providers=providers,
        type_is_altaz=type_is_altaz,
        ra0=RA0,
        dec0=DEC0,
        lon=LON,
        lat=LAT,
        t_start=T_START,
        duration=duration,
        pa_step=PA_STEP,
    )
    kwargs.update(overrides)
    return kwargs


def test_flat_sky_matches_exact_dft_kernel_heterogeneous(params):
    """Gridded a-terms == predict_vis_beam on a heterogeneous 2-corr observation."""
    times, antenna1, antenna2, uvw, duration = observation()
    providers, ant_type, type_is_altaz = hetero_beams()
    freqs = np.array([1.35e9, 1.40e9, 1.45e9])

    path = write_flat_image(params)
    prepared = prepare_fits_sky(path, RA0, DEC0, freqs, 5e7, ncorr=2, nrow=uvw.shape[0], backend="fft")
    assert prepared.backend == "fft"
    prepared = attach_fits_aterm(prepared, freq_tol=0.0, **attach_kwargs(duration, providers, ant_type, type_is_altaz))
    got = predict_fits_block(prepared, uvw, times=times, antenna1=antenna1, antenna2=antenna2, epsilon=1e-11)

    flux = np.array([p[2] for p in PIXELS])
    bmat = np.zeros((flux.size, 2, freqs.size), dtype=np.complex128)
    bmat[:, 0, :] = flux[:, None]
    bmat[:, 1, :] = flux[:, None]
    ref_sky = attach_beam(
        reference_component_sky(prepared, bmat, ncorr=2),
        ant_type,
        providers,
        type_is_altaz,
        RA0,
        DEC0,
        LON,
        LAT,
        T_START,
        duration,
        PA_STEP,
        ncorr=2,
    )
    ref = predict_block(ref_sky, uvw, times=times, antenna1=antenna1, antenna2=antenna2)

    scale = np.abs(ref).max()
    assert scale > 0.1  # the comparison is not vacuous
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-6 * scale)
    # And the beam genuinely does something: no-beam prediction differs.
    assert not np.allclose(predict_block(reference_component_sky(prepared, bmat, ncorr=2), uvw), got, atol=1e-3 * scale)


def test_full_jones_leakage_matches_exact_jones_kernel(params):
    """4-corr polarised sky + complex leakage beams == predict_vis_jones."""
    times, antenna1, antenna2, uvw, duration = observation(seed=3)
    freqs = np.array([1.39e9, 1.41e9])

    # Two smooth, complex, leakage-carrying beam cubes on an (l, m, freq) grid
    # comfortably covering the (rotated) image.
    grid = np.linspace(-0.06, 0.06, 41)
    bfreqs = np.array([1.35e9, 1.45e9])
    ll, mm = np.meshgrid(grid, grid, indexing="ij")

    def beam_cube(seed_phase):
        values = np.zeros((grid.size, grid.size, bfreqs.size, 4), dtype=np.complex128)
        for k, bf in enumerate(bfreqs):
            sigma = 0.03 * 1.4e9 / bf
            base = np.exp(-(ll**2 + mm**2) / (2 * sigma**2))
            values[:, :, k, 0] = base * np.exp(1j * seed_phase * ll * 30)
            values[:, :, k, 1] = 0.06 * base * (mm * 20 + 1j * ll * 10)
            values[:, :, k, 2] = 0.04 * base * (ll * 15 - 1j * mm * 25)
            values[:, :, k, 3] = 0.95 * base * np.exp(-1j * seed_phase * mm * 20)
        return values

    providers = [
        FitsBeamProvider.from_arrays(grid, grid, bfreqs, beam_cube(0.5), name="A"),
        FitsBeamProvider.from_arrays(grid, grid, bfreqs, beam_cube(-0.3), name="B"),
    ]
    ant_type = np.array([0, 1, 0, 1], dtype=np.int64)
    type_is_altaz = np.array([True, True])

    stokes_values = [(1.0, 0.2, -0.1, 0.05), (2.0, -0.3, 0.15, 0.0), (0.5, 0.1, 0.05, -0.02)]
    path = write_flat_image(params, nstokes=4, stokes_values=stokes_values)
    prepared = prepare_fits_sky(
        path, RA0, DEC0, freqs, 2e7, ncorr=4, nrow=uvw.shape[0], backend="fft", polarisation=True
    )
    eye = np.eye(2, dtype=np.complex128)
    prepared = attach_fits_aterm(
        prepared,
        freq_tol=0.0,
        full_jones=True,
        basis_transform=eye,
        **attach_kwargs(duration, providers, ant_type, type_is_altaz),
    )
    got = predict_fits_block(prepared, uvw, times=times, antenna1=antenna1, antenna2=antenna2, epsilon=1e-11)

    # Feed-basis coherency per component: [[I+Q, U+iV], [U-iV, I-Q]].
    bmat = np.zeros((len(PIXELS), 4, freqs.size), dtype=np.complex128)
    for s, (stokes_i, stokes_q, stokes_u, stokes_v) in enumerate(stokes_values):
        bmat[s, 0, :] = stokes_i + stokes_q
        bmat[s, 1, :] = stokes_u + 1j * stokes_v
        bmat[s, 2, :] = stokes_u - 1j * stokes_v
        bmat[s, 3, :] = stokes_i - stokes_q
    ref_sky = attach_beam(
        reference_component_sky(prepared, bmat, ncorr=4),
        ant_type,
        providers,
        type_is_altaz,
        RA0,
        DEC0,
        LON,
        LAT,
        T_START,
        duration,
        PA_STEP,
        ncorr=4,
        full_jones=True,
        basis_transform=eye,
    )
    ref = predict_block(ref_sky, uvw, times=times, antenna1=antenna1, antenna2=antenna2)

    scale = np.abs(ref).max()
    assert scale > 0.1
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-6 * scale)


def test_cube_sky_matches_exact_dft_kernel(params):
    """Per-channel (spectral cube) path: image and a-term both vary per channel."""
    times, antenna1, antenna2, uvw, duration = observation(seed=11)
    providers, ant_type, type_is_altaz = hetero_beams()
    freqs = np.array([1.40e9, 1.42e9])

    data = np.zeros((freqs.size, NPIX, NPIX), dtype=np.float64)
    per_chan = {}
    for ira, idec, flux in PIXELS:
        values = (flux, 0.6 * flux)
        data[:, idec, ira] = values
        per_chan[(ira, idec)] = values
    header = make_header(NPIX, nchan=freqs.size, cell=CELL_DEG, freqs=freqs)
    path = params.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data, header=header).writeto(path, overwrite=True)

    prepared = prepare_fits_sky(
        path, RA0, DEC0, freqs, 2e7, ncorr=2, nrow=uvw.shape[0], backend="perchan", spectrum="cube"
    )
    prepared = attach_fits_aterm(prepared, freq_tol=0.0, **attach_kwargs(duration, providers, ant_type, type_is_altaz))
    got = predict_fits_block(prepared, uvw, times=times, antenna1=antenna1, antenna2=antenna2, epsilon=1e-11)

    bmat = np.zeros((len(PIXELS), 2, freqs.size), dtype=np.complex128)
    for s, (ira, idec, _) in enumerate(PIXELS):
        bmat[s, 0, :] = per_chan[(ira, idec)]
        bmat[s, 1, :] = per_chan[(ira, idec)]
    ref_sky = attach_beam(
        reference_component_sky(prepared, bmat, ncorr=2),
        ant_type,
        providers,
        type_is_altaz,
        RA0,
        DEC0,
        LON,
        LAT,
        T_START,
        duration,
        PA_STEP,
        ncorr=2,
    )
    ref = predict_block(ref_sky, uvw, times=times, antenna1=antenna1, antenna2=antenna2)

    scale = np.abs(ref).max()
    assert scale > 0.1
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-6 * scale)


def test_poly_spectrum_matches_component_bridge(params):
    """POLY spectra: the a-term image path == the exact component-bridge path.

    Also exercises component_sky_from_fits_dft, which skysim uses whenever the
    DFT backend wins the cost model under a-term beams.
    """
    times, antenna1, antenna2, uvw, duration = observation(seed=13)
    providers, ant_type, type_is_altaz = hetero_beams()
    freqs = np.array([1.36e9, 1.40e9, 1.44e9])

    path = write_flat_image(params)
    spi = np.full((NPIX, NPIX), -0.7)
    spi_path = params.random_named_file(suffix=".fits")
    fits.PrimaryHDU(spi, header=make_header(NPIX, cell=CELL_DEG)).writeto(spi_path, overwrite=True)

    common = dict(ncorr=2, nrow=uvw.shape[0], spi_maps=[spi_path], ref_freq=1.40e9, linear_basis=True)
    prepared_fft = prepare_fits_sky(path, RA0, DEC0, freqs, 4e7, backend="fft", **common)
    prepared_fft = attach_fits_aterm(
        prepared_fft, freq_tol=0.0, **attach_kwargs(duration, providers, ant_type, type_is_altaz)
    )
    got = predict_fits_block(prepared_fft, uvw, times=times, antenna1=antenna1, antenna2=antenna2, epsilon=1e-11)

    prepared_dft = prepare_fits_sky(path, RA0, DEC0, freqs, 4e7, backend="dft", **common)
    bridged = attach_beam(
        # attach_beam needs the full correlation width, exactly as _BeamContext.attach does.
        to_full_corr(component_sky_from_fits_dft(prepared_dft)),
        ant_type,
        providers,
        type_is_altaz,
        RA0,
        DEC0,
        LON,
        LAT,
        T_START,
        duration,
        PA_STEP,
        ncorr=2,
    )
    ref = predict_block(bridged, uvw, times=times, antenna1=antenna1, antenna2=antenna2)

    scale = np.abs(ref).max()
    assert scale > 0.1
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-6 * scale)


def test_channel_chunking_is_invariant(params):
    """Chunked prediction (select_channels) == whole-band prediction."""
    times, antenna1, antenna2, uvw, duration = observation(seed=17)
    providers, ant_type, type_is_altaz = hetero_beams()
    freqs = np.linspace(1.35e9, 1.45e9, 4)

    path = write_flat_image(params)
    prepared = prepare_fits_sky(path, RA0, DEC0, freqs, 2.5e7, ncorr=2, nrow=uvw.shape[0], backend="fft")
    # A coarse freq_tol so knots straddle chunk boundaries (the interesting case).
    prepared = attach_fits_aterm(prepared, freq_tol=1e-2, **attach_kwargs(duration, providers, ant_type, type_is_altaz))

    kwargs = dict(times=times, antenna1=antenna1, antenna2=antenna2, epsilon=1e-11)
    whole = predict_fits_block(prepared, uvw, **kwargs)
    parts = np.concatenate(
        [predict_fits_block(prepared.select_channels(np.arange(a, a + 2)), uvw, **kwargs) for a in (0, 2)],
        axis=1,
    )
    np.testing.assert_allclose(parts, whole, rtol=0, atol=1e-10 * np.abs(whole).max())


def test_non_altaz_collapses_time_axis(params):
    """Equatorial-style mounts: a single time knot, still matching the kernel."""
    times, antenna1, antenna2, uvw, duration = observation(seed=19)
    providers, ant_type, _ = hetero_beams()
    type_is_altaz = np.array([False, False])
    freqs = np.array([1.40e9, 1.42e9])

    path = write_flat_image(params)
    prepared = prepare_fits_sky(path, RA0, DEC0, freqs, 2e7, ncorr=2, nrow=uvw.shape[0], backend="fft")
    prepared = attach_fits_aterm(prepared, freq_tol=0.0, **attach_kwargs(duration, providers, ant_type, type_is_altaz))
    assert prepared.aterm.tgrid.size == 1
    got = predict_fits_block(prepared, uvw, times=times, antenna1=antenna1, antenna2=antenna2, epsilon=1e-11)

    flux = np.array([p[2] for p in PIXELS])
    bmat = np.zeros((flux.size, 2, freqs.size), dtype=np.complex128)
    bmat[:, 0, :] = flux[:, None]
    bmat[:, 1, :] = flux[:, None]
    ref_sky = attach_beam(
        reference_component_sky(prepared, bmat, ncorr=2),
        ant_type,
        providers,
        type_is_altaz,
        RA0,
        DEC0,
        LON,
        LAT,
        T_START,
        duration,
        PA_STEP,
        ncorr=2,
    )
    ref = predict_block(ref_sky, uvw, times=times, antenna1=antenna1, antenna2=antenna2)
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-6 * np.abs(ref).max())


# --------------------------------------------------------------------------- frequency knots


class _LinearFreqBeam(BeamProvider):
    """A beam whose voltage is exactly linear in frequency (lerp is exact)."""

    def _eval(self, l_feed, m_feed, freqs):
        radius = (l_feed**2 + m_feed**2)[:, None]
        base = 1.0 - radius * (np.asarray(freqs)[None, :] / 1e9) * 20.0
        return np.stack([base, 0.9 * base], axis=-1)


class _QuadraticFreqBeam(BeamProvider):
    """A beam with genuine curvature in frequency, to force interior knots."""

    def _eval(self, l_feed, m_feed, freqs):
        radius = (l_feed**2 + m_feed**2)[:, None]
        x = np.asarray(freqs)[None, :] / 1e9
        base = 1.0 - 50.0 * radius * x**2
        return np.stack([base, base], axis=-1)


def _probe_lm():
    extent = np.radians(CELL_DEG) * NPIX / 2
    grid = np.linspace(-extent, extent, 9)
    ll, mm = np.meshgrid(grid, grid, indexing="ij")
    return ll.ravel(), mm.ravel()


def test_freq_knots_linear_beam_needs_only_endpoints():
    ell, emm = _probe_lm()
    freqs = np.linspace(1.0e9, 1.5e9, 16)
    knots = select_freq_knots([_LinearFreqBeam()], [False], np.zeros(1), ell, emm, freqs, tol=1e-6)
    np.testing.assert_array_equal(knots, [0, 15])


def test_freq_knots_tighten_with_tolerance_and_span_the_band():
    ell, emm = _probe_lm()
    freqs = np.linspace(1.0e9, 1.5e9, 32)
    loose = select_freq_knots([_QuadraticFreqBeam()], [False], np.zeros(1), ell, emm, freqs, tol=1e-2)
    tight = select_freq_knots([_QuadraticFreqBeam()], [False], np.zeros(1), ell, emm, freqs, tol=1e-4)
    for knots in (loose, tight):
        assert knots[0] == 0 and knots[-1] == 31
        assert np.all(np.diff(knots) > 0)
    assert tight.size > loose.size
    # tol <= 0 is the exact (every channel) setting.
    exact = select_freq_knots([_QuadraticFreqBeam()], [False], np.zeros(1), ell, emm, freqs, tol=0.0)
    np.testing.assert_array_equal(exact, np.arange(32))


def test_freq_lerp_is_exact_for_linear_beam(params):
    """Two knots reproduce the all-channel answer when the beam is linear in nu.

    Validates the visibility-domain frequency blend (weights, segment split)
    against the dense-knot limit, independently of any beam model error.
    """
    times, antenna1, antenna2, uvw, duration = observation(seed=23)
    providers = [_LinearFreqBeam()]
    ant_type = np.zeros(4, dtype=np.int64)
    type_is_altaz = np.array([False])
    freqs = np.linspace(1.35e9, 1.45e9, 6)

    path = write_flat_image(params)

    def run(tol):
        prepared = prepare_fits_sky(path, RA0, DEC0, freqs, 2e7, ncorr=2, nrow=uvw.shape[0], backend="fft")
        prepared = attach_fits_aterm(
            prepared, freq_tol=tol, **attach_kwargs(duration, providers, ant_type, type_is_altaz)
        )
        return prepared, predict_fits_block(
            prepared, uvw, times=times, antenna1=antenna1, antenna2=antenna2, epsilon=1e-11
        )

    prepared_sparse, sparse = run(1e-5)
    assert prepared_sparse.aterm.fknot_chan.size == 2
    prepared_exact, exact = run(0.0)
    assert prepared_exact.aterm.fknot_chan.size == freqs.size
    # The two agree to the complex64 storage precision of the voltage maps
    # (~1.2e-7 relative): lerping rounded knot maps vs rounding per-channel maps.
    np.testing.assert_allclose(sparse, exact, rtol=0, atol=5e-7 * np.abs(exact).max())


# --------------------------------------------------------------------------- end-to-end


@pytest.fixture(scope="module")
def e2e_ms():
    """A small heterogeneous MS whose phase centre matches make_header's (RA0, DEC0)."""
    from simms.telescope.generate_ms import create_ms

    holder = InitThisTest()
    ms = holder.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=["J2000", "1h0m0s", "-31deg"],
        dtime=600,
        ntimes=4,
        start_freq="1420MHz",
        dfreq="4MHz",
        nchan=2,
        correlations=["XX", "YY"],
        row_chunks=100000,
        sefd=None,
        column="DATA",
        start_time="2025-03-06T20:00:00",
        smooth=None,
        fit_order=None,
        subarray_range=[60, 68],
    )
    yield holder, ms


def _e2e_sky(holder):
    """A FITS image with one off-axis pixel, and the identical source as ASCII."""
    from astropy import units as u
    from astropy.coordinates import Angle
    from astropy.wcs import WCS

    npix, offset, flux = 256, 90, 4.0  # ~0.5 deg south: near the L-band half power
    header = make_header(npix)
    data = np.zeros((npix, npix), dtype=np.float64)
    data[npix // 2 - offset, npix // 2] = flux
    img = holder.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data, header=header).writeto(img, overwrite=True)

    ra_deg, dec_deg = WCS(header).celestial.wcs_pix2world([[npix // 2, npix // 2 - offset]], 0)[0]
    ra = Angle(ra_deg, unit=u.deg).to_string(unit=u.hourangle, sep="hms", precision=8)
    dec = Angle(dec_deg, unit=u.deg).to_string(unit=u.deg, sep="dms", precision=8)
    ascii_sky = holder.random_named_file(suffix=".txt")
    with open(ascii_sky, "w") as fh:
        fh.write("#format: name ra dec stokes_i\n")
        fh.write(f"S {ra} {dec} {flux}\n")

    beams = holder.random_named_file(suffix=".yaml")
    with open(beams, "w") as fh:
        fh.write("MKAT-MA:\n  jimbeam: MKAT-MA-L-JIM-2026\nMKAT-EA:\n  jimbeam: MKAT-EA-L-JIM-2026\n")
    return img, ascii_sky, beams


def test_e2e_fits_aterm_matches_ascii_dft_path(e2e_ms):
    """skysim end to end: gridded FITS a-terms == the exact ASCII beam path.

    Covers the full wiring -- pointing centre, blockwise TIME/ANTENNA args, the
    channel-chunk bookkeeping and dtype cast -- on a heterogeneous subarray.
    """
    from daskms import xds_from_ms

    from simms.apps import skysim

    from . import skysim_opts

    holder, ms = e2e_ms
    img, ascii_sky, beams = _e2e_sky(holder)

    # Smearing off throughout: the gridded a-term path cannot carry a per-visibility
    # decorrelation factor, so leaving the default on would compare a smeared
    # component prediction against an unsmeared gridded one. Smearing is covered on
    # its own in smearing_tests.py.
    opts = dict(primary_beam=beams, smearing="none")
    skysim.runit(skysim_opts(ms, ascii_sky=ascii_sky, column="ASCIIBEAM", **opts))
    skysim.runit(
        skysim_opts(
            ms,
            fits_sky=img,
            column="FITSATERM",
            predict_backend="fft",
            aterm_freq_tol=0.0,
            **opts,
        )
    )
    # Few bright pixels + a beam: the auto/dft backend takes the exact component bridge.
    skysim.runit(skysim_opts(ms, fits_sky=img, column="FITSBRIDGE", predict_backend="dft", **opts))

    ds = xds_from_ms(ms)[0]
    ref = ds.ASCIIBEAM.data.compute()
    aterm = ds.FITSATERM.data.compute()
    bridge = ds.FITSBRIDGE.data.compute()
    scale = np.abs(ref).max()
    assert scale > 0.1
    # complex64 DATA columns bound the achievable agreement (~1e-7 relative).
    np.testing.assert_allclose(aterm, ref, rtol=0, atol=1e-5 * scale)
    np.testing.assert_allclose(bridge, ref, rtol=0, atol=1e-5 * scale)


def test_e2e_average_mode_remains_available(e2e_ms):
    """--fits-beam-mode average keeps the legacy PA-averaged power beam."""
    from daskms import xds_from_ms

    from simms.apps import skysim

    from . import skysim_opts

    holder, ms = e2e_ms
    img, _, beams = _e2e_sky(holder)

    skysim.runit(skysim_opts(ms, fits_sky=img, column="NOBEAM"))
    skysim.runit(skysim_opts(ms, fits_sky=img, primary_beam=beams, column="AVGBEAM", fits_beam_mode="average"))
    ds = xds_from_ms(ms)[0]
    nobeam = ds.NOBEAM.data.compute()
    avg = ds.AVGBEAM.data.compute()
    assert np.all(np.isfinite(avg))
    assert not np.allclose(avg, nobeam)
    assert np.abs(avg).mean() < np.abs(nobeam).mean()


def test_product_cache_is_neutral(params, monkeypatch):
    """The per-segment apparent-beam-product cache must not change the answer.

    It memoises knot products across a segment's channels in diagonal mode;
    disabling it (budget 0) must give bit-comparable visibilities.
    """
    import simms.skymodel.aterms as aterms_mod

    times, antenna1, antenna2, uvw, duration = observation(seed=41)
    providers, ant_type, type_is_altaz = hetero_beams()
    freqs = np.linspace(1.38e9, 1.44e9, 4)

    data = np.zeros((freqs.size, NPIX, NPIX), dtype=np.float64)
    for ira, idec, flux in PIXELS:
        data[:, idec, ira] = [flux * (1.0 + 0.1 * k) for k in range(freqs.size)]
    header = make_header(NPIX, nchan=freqs.size, cell=CELL_DEG, freqs=freqs)
    path = params.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data, header=header).writeto(path, overwrite=True)

    def run():
        prepared = prepare_fits_sky(
            path, RA0, DEC0, freqs, 2e7, ncorr=2, nrow=uvw.shape[0], backend="perchan", spectrum="cube"
        )
        # A coarse tolerance so a segment spans several channels and the cache is reused.
        prepared = attach_fits_aterm(
            prepared, freq_tol=1e-2, **attach_kwargs(duration, providers, ant_type, type_is_altaz)
        )
        return predict_fits_block(prepared, uvw, times=times, antenna1=antenna1, antenna2=antenna2, epsilon=1e-11)

    with_cache = run()
    monkeypatch.setattr(aterms_mod, "PRODUCT_CACHE_FRACTION", 0.0)
    without_cache = run()
    assert np.abs(with_cache).max() > 0.1
    np.testing.assert_allclose(with_cache, without_cache, rtol=0, atol=1e-12 * np.abs(without_cache).max())


def test_memory_ceiling_raises_and_names_the_escape(params):
    """The voltage-map cache ceiling fails loudly, naming the mode that has none."""
    _times, _antenna1, _antenna2, uvw, duration = observation(seed=29)
    providers, ant_type, type_is_altaz = hetero_beams()
    freqs = np.array([1.40e9, 1.42e9])
    path = write_flat_image(params)
    prepared = prepare_fits_sky(path, RA0, DEC0, freqs, 2e7, ncorr=2, nrow=uvw.shape[0], backend="fft")

    with pytest.raises(MemoryError, match="fits-beam-mode average"):
        attach_fits_aterm(
            prepared, freq_tol=0.0, max_gib=1e-9, **attach_kwargs(duration, providers, ant_type, type_is_altaz)
        )

    # The sizing helper skysim pre-flights with must agree with the guard.
    needed = aterm_cache_min_gib(prepared.npix_l, prepared.npix_m, False)
    assert needed > 1e-9
    attach_fits_aterm(
        prepared, freq_tol=0.0, max_gib=needed, **attach_kwargs(duration, providers, ant_type, type_is_altaz)
    )


def test_aterm_rejects_circular_basis_model(params):
    """A circular-basis sky must be refused, not silently given wrong cross-hands."""
    _times, _antenna1, _antenna2, uvw, duration = observation(seed=31)
    providers, ant_type, type_is_altaz = hetero_beams()
    freqs = np.array([1.40e9, 1.42e9])
    path = write_flat_image(params)
    prepared = prepare_fits_sky(
        path, RA0, DEC0, freqs, 2e7, ncorr=2, nrow=uvw.shape[0], backend="fft", linear_basis=False
    )
    with pytest.raises(ValueError, match="linear feed basis"):
        attach_fits_aterm(prepared, freq_tol=0.0, **attach_kwargs(duration, providers, ant_type, type_is_altaz))


def test_component_bridge_rejects_circular_basis(params):
    """The DFT bridge exists to attach a beam, so it refuses a circular-basis model."""
    _, _, _, uvw, _ = observation(seed=33)
    freqs = np.array([1.40e9, 1.42e9])
    path = write_flat_image(params)
    prepared = prepare_fits_sky(
        path, RA0, DEC0, freqs, 2e7, ncorr=2, nrow=uvw.shape[0], backend="dft", linear_basis=False
    )
    with pytest.raises(ValueError, match="linear-feed-basis"):
        component_sky_from_fits_dft(prepared)


def test_e2e_memory_ceiling_falls_back_instead_of_crashing(e2e_ms):
    """An image too large for the cache degrades to the power beam, it does not crash.

    Guards the user-facing regression: `aterm` is the FITS-path default, so a run
    that produced output before must not start raising MemoryError.
    """
    from daskms import xds_from_ms

    from simms.apps import skysim

    from . import skysim_opts

    holder, ms = e2e_ms
    img, _, beams = _e2e_sky(holder)

    # Far below what any real image needs, so the ceiling is certain to bite.
    opts = skysim_opts(
        ms,
        fits_sky=img,
        primary_beam=beams,
        column="TOOBIG",
        predict_backend="fft",
        beam_grid_max_gib=1e-7,
    )
    skysim.runit(opts)  # must not raise
    got = xds_from_ms(ms)[0].TOOBIG.data.compute()
    assert np.all(np.isfinite(got))
    assert np.abs(got).max() > 0

    # It fell back to `average`, so it must match an explicit average run -- on the
    # same backend, since `average` applies a per-channel beam on the DFT path but a
    # single mid-band beam on the gridder path.
    skysim.runit(
        skysim_opts(
            ms,
            fits_sky=img,
            primary_beam=beams,
            column="AVGREF",
            fits_beam_mode="average",
            predict_backend="fft",
        )
    )
    ref = xds_from_ms(ms)[0].AVGREF.data.compute()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6 * np.abs(ref).max())


def test_e2e_circular_ms_falls_back_to_average(e2e_ms):
    """Diagonal a-terms on a circular MS degrade to the basis-independent power beam."""
    from daskms import xds_from_ms

    from simms.apps import skysim
    from simms.telescope.generate_ms import create_ms

    from . import skysim_opts

    holder, _ = e2e_ms
    img, _, beams = _e2e_sky(holder)
    ms = holder.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=["J2000", "1h0m0s", "-31deg"],
        dtime=600,
        ntimes=2,
        start_freq="1420MHz",
        dfreq="4MHz",
        nchan=2,
        correlations=["RR", "LL"],
        row_chunks=100000,
        sefd=None,
        column="DATA",
        start_time="2025-03-06T20:00:00",
        smooth=None,
        fit_order=None,
        subarray_range=[60, 68],
    )
    opts = skysim_opts(ms, fits_sky=img, primary_beam=beams, column="CIRC", pol_basis="circular")
    skysim.runit(opts)  # must not raise
    got = xds_from_ms(ms)[0].CIRC.data.compute()
    assert np.all(np.isfinite(got))
    assert np.abs(got).max() > 0


def test_e2e_full_jones_requires_four_correlations(e2e_ms):
    """--beam-jones full on the FITS path is honoured, so it must reject a 2-corr MS."""
    from simms.apps import skysim

    from . import skysim_opts

    holder, ms = e2e_ms  # the fixture MS is XX, YY
    img, _, beams = _e2e_sky(holder)
    opts = skysim_opts(ms, fits_sky=img, primary_beam=beams, column="FJ", beam_jones="full", predict_backend="fft")
    with pytest.raises(RuntimeError, match="4 correlations"):
        skysim.runit(opts)


def test_map_cache_survives_pickling():
    """The prepared model must remain picklable (process-based schedulers)."""
    import pickle

    original = _MapCache(1 << 20)
    original.put(("k",), np.ones(4))
    clone = pickle.loads(pickle.dumps(original))
    assert clone.max_bytes == original.max_bytes
    assert clone.get(("k",)) is None  # cache content deliberately dropped
