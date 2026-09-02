"""Time and bandwidth smearing of the predicted model.

An averaged MS holds visibilities the correlator integrated over a channel and
over a dump, so a monochromatic, instantaneous model over-predicts every source
away from the phase centre -- worse the longer the baseline. That is a modelling
choice for a simulation from scratch and a bug for ``skysim --mode subtract``,
which differences the model against real averaged data.

The reference here is a brute-force average of the phasor over a baseline that
actually rotates with the Earth and over a finite band, evaluated with explicit
loops; it shares no code with :mod:`simms.skymodel.smearing`, so it pins the
closed form *and* the signs of the fringe-rate terms.
"""

import numpy as np
import pytest
from daskms import xds_from_ms, xds_from_table

from simms import SCHEMADIR
from simms.apps import skysim
from simms.constants import OMEGA_EARTH, C
from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.skymodel.kernels import NO_SMEAR_UVW, predict_vis, predict_vis_beam, predict_vis_jones
from simms.skymodel.mstools import attach_smearing, predict_block, prepare_skymodel
from simms.skymodel.smearing import Smearing
from simms.telescope.generate_ms import create_ms

from . import InitTest, skysim_opts

SCHEMA = f"{SCHEMADIR}/source_schema.yaml"

NO_SMEAR = (False, 0.0, NO_SMEAR_UVW)


# --------------------------------------------------------------------------- reference


def rotating_uvw(hour_angle, dec, baseline):
    """``(u, v, w)`` of an equatorial baseline ``(Lx, Ly, Lz)`` tracking ``(H, dec)``."""
    lx, ly, lz = baseline
    sin_h, cos_h = np.sin(hour_angle), np.cos(hour_angle)
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    return np.array(
        [
            sin_h * lx + cos_h * ly,
            -sin_d * cos_h * lx + sin_d * sin_h * ly + cos_d * lz,
            cos_d * cos_h * lx - cos_d * sin_h * ly + sin_d * lz,
        ]
    )


def brute_force_smeared(baseline, dec0, hour_angle, freq, chan_width, exposure, lmn, nsample=2001):
    """Mean of ``exp(1j*phi)`` over the dump and the channel, by explicit sampling."""
    times = np.linspace(-0.5 * exposure, 0.5 * exposure, nsample)
    freqs = np.linspace(freq - 0.5 * chan_width, freq + 0.5 * chan_width, nsample)
    uvw = np.array([rotating_uvw(hour_angle + OMEGA_EARTH * t, dec0, baseline) for t in times])
    base = (uvw @ lmn) * 2.0 * np.pi / C
    return np.exp(1j * base[:, None] * freqs[None, :]).mean()


def one_source_lmn(theta, pa=0.7):
    """Direction cosines ``(l, m, n - 1)`` of a source ``theta`` radians off axis."""
    el, em = theta * np.cos(pa), theta * np.sin(pa)
    return np.array([el, em, np.sqrt(1.0 - el * el - em * em) - 1.0])


def kernel_vis(uvw, freqs, lmn, smearing=None, exposure=None, gauss_shape=None, uniform=True):
    """One unit-brightness source through :func:`predict_vis`, smeared or not."""
    uvw = np.ascontiguousarray(uvw, dtype=np.float64)
    lmn = np.ascontiguousarray(np.atleast_2d(lmn), dtype=np.float64)
    nrow, nchan, nsrc = uvw.shape[0], freqs.size, lmn.shape[0]
    smear_args = NO_SMEAR if smearing is None else (True, smearing.bw_half, smearing.row_uvw(uvw, exposure))
    shape = np.zeros((nsrc, 3)) if gauss_shape is None else np.ascontiguousarray(gauss_shape)
    vis = np.zeros((nrow, nchan, 1), dtype=np.complex128)
    predict_vis(
        uvw,
        freqs,
        uniform,
        lmn,
        shape,
        np.full(nsrc, gauss_shape is not None),
        np.ones((nsrc, 1, nchan), dtype=np.complex128),
        np.ones((nsrc, 1)),
        np.zeros(nrow, dtype=np.int64),
        vis,
        *smear_args,
    )
    return vis[:, :, 0]


# --------------------------------------------------------------------------- the maths


@pytest.mark.parametrize(
    "name, chan_width, exposure, offset_arcmin",
    [
        ("bandwidth only", 6.7e6, 1e-6, 50.0),
        ("time only", 1.0, 60.0, 50.0),
        ("both", 6.7e6, 8.0, 50.0),
        ("wide and slow", 1.6e7, 30.0, 25.0),
    ],
)
def test_analytic_factor_matches_a_brute_force_average(name, chan_width, exposure, offset_arcmin):
    dec0, hour_angle, freq = np.deg2rad(-30.7), np.deg2rad(20.0), 1.284e9
    baseline = np.array([3000.0, -5200.0, 1500.0])
    lmn = one_source_lmn(np.deg2rad(offset_arcmin / 60.0))
    uvw = rotating_uvw(hour_angle, dec0, baseline)[None, :]

    exact = brute_force_smeared(baseline, dec0, hour_angle, freq, chan_width, exposure, lmn)
    got = kernel_vis(uvw, np.array([freq]), lmn, Smearing.from_ms(chan_width, dec0), np.array([exposure]))[0, 0]

    # Decorrelation this deep is the regime the closed form is for; check the test
    # itself stays there rather than passing on a factor of ~1.
    assert abs(exact) < 0.95
    np.testing.assert_allclose(got, exact, rtol=5e-3, atol=1e-4)


def test_a_source_at_the_phase_centre_is_untouched():
    """``base`` and its time derivative both vanish there, so both sincs are exactly 1."""
    dec0, freq = np.deg2rad(-30.7), 1.284e9
    uvw = np.random.default_rng(5).normal(0, 4000, (16, 3))
    freqs = freq + np.arange(4) * 8e6
    lmn = np.array([0.0, 0.0, 0.0])
    smeared = kernel_vis(uvw, freqs, lmn, Smearing.from_ms(8e6, dec0), np.full(16, 8.0))
    np.testing.assert_allclose(smeared, kernel_vis(uvw, freqs, lmn), rtol=1e-12)


def test_decorrelation_deepens_with_offset_and_baseline_length():
    dec0, freqs = np.deg2rad(-30.0), np.array([1.4e9])
    smearing = Smearing.from_ms(8e6, dec0)
    exposure = np.array([8.0])
    lengths = np.array([200.0, 1000.0, 4000.0, 8000.0])
    offsets = np.deg2rad(np.array([1.0, 10.0, 30.0, 60.0]) / 60.0)

    amp = np.empty((offsets.size, lengths.size))
    for i, theta in enumerate(offsets):
        lmn = one_source_lmn(theta)
        for j, length in enumerate(lengths):
            uvw = rotating_uvw(np.deg2rad(30.0), dec0, np.array([0.6, -0.7, 0.4]) * length)[None, :]
            amp[i, j] = abs(kernel_vis(uvw, freqs, lmn, smearing, exposure)[0, 0])

    assert (amp <= 1.0 + 1e-12).all()
    assert (np.diff(amp, axis=0) < 0).all(), "amplitude must fall as the source moves off axis"
    assert (np.diff(amp, axis=1) < 0).all(), "amplitude must fall as the baseline lengthens"


def test_a_gaussian_component_is_smeared_like_a_point():
    """The envelope is the source's own shape; smearing multiplies it, not the reverse."""
    dec0, freqs = np.deg2rad(-30.0), 1.4e9 + np.arange(3) * 8e6
    uvw = np.random.default_rng(7).normal(0, 3000, (12, 3))
    lmn = one_source_lmn(np.deg2rad(40.0 / 60.0))
    shape = np.array([[3e-5, 1e-5, 0.6]])
    smearing, exposure = Smearing.from_ms(8e6, dec0), np.full(12, 8.0)

    plain = kernel_vis(uvw, freqs, lmn, gauss_shape=shape)
    smeared = kernel_vis(uvw, freqs, lmn, smearing, exposure, gauss_shape=shape)
    point_ratio = kernel_vis(uvw, freqs, lmn, smearing, exposure) / kernel_vis(uvw, freqs, lmn)
    np.testing.assert_allclose(smeared, plain * point_ratio, rtol=1e-10)


@pytest.mark.parametrize("nchan", [8, 3 * 256 + 7])
def test_the_uniform_recurrence_matches_the_general_path(nchan):
    """The smeared uniform kernel runs a *second* recurrence for sin(rate_half*nu).

    The general branch evaluates that ``sinc`` directly, so running one grid
    through both pins the recurrence; spanning several renormalisation intervals
    catches drift in it.
    """
    dec0 = np.deg2rad(-30.0)
    freqs = 1.4e9 + np.arange(nchan) * 1e6
    uvw = np.random.default_rng(17).normal(0, 3000, (4, 3))
    lmn = one_source_lmn(np.deg2rad(35.0 / 60.0))
    smearing, exposure = Smearing.from_ms(1e6, dec0), np.full(4, 8.0)

    by_recurrence = kernel_vis(uvw, freqs, lmn, smearing, exposure, uniform=True)
    directly = kernel_vis(uvw, freqs, lmn, smearing, exposure, uniform=False)
    np.testing.assert_allclose(by_recurrence, directly, rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------- the beam kernels


def _beam_inputs(nrow, nchan, nsrc, ncorr):
    """Unit beams, so the beam kernels must reproduce the plain one exactly."""
    return dict(
        antenna1=np.zeros(nrow, dtype=np.int64),
        antenna2=np.ones(nrow, dtype=np.int64),
        ant_type=np.zeros(2, dtype=np.int64),
        pa_lo=np.zeros(nrow, dtype=np.int64),
        pa_wt=np.zeros(nrow),
        nsrc=nsrc,
        nchan=nchan,
        ncorr=ncorr,
    )


@pytest.mark.parametrize("full_jones", [False, True])
def test_the_beam_kernels_smear_identically(full_jones):
    dec0 = np.deg2rad(-30.0)
    freqs = 1.4e9 + np.arange(4) * 8e6
    uvw = np.random.default_rng(13).normal(0, 3500, (9, 3))
    lmn = np.atleast_2d(one_source_lmn(np.deg2rad(45.0 / 60.0)))
    nrow, nchan, nsrc = uvw.shape[0], freqs.size, 1
    ncorr = 4 if full_jones else 2
    smearing, exposure = Smearing.from_ms(8e6, dec0), np.full(nrow, 8.0)
    smear_args = (True, smearing.bw_half, smearing.row_uvw(uvw, exposure))

    bmat = np.zeros((nsrc, ncorr, nchan), dtype=np.complex128)
    bmat[:, 0], bmat[:, -1] = 1.0, 1.0
    common = [
        uvw,
        freqs,
        True,
        lmn,
        np.zeros((nsrc, 3)),
        np.zeros(nsrc, dtype=np.bool_),
        bmat,
        np.ones((nsrc, 1)),
        np.zeros(nrow, dtype=np.int64),
    ]
    args = _beam_inputs(nrow, nchan, nsrc, ncorr)

    vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    if full_jones:
        grid = np.zeros((1, 2, nsrc, nchan, 2, 2), dtype=np.complex128)
        grid[..., 0, 0], grid[..., 1, 1] = 1.0, 1.0
        predict_vis_jones(
            *common,
            vis,
            args["antenna1"],
            args["antenna2"],
            args["ant_type"],
            grid,
            args["pa_lo"],
            args["pa_wt"],
            *smear_args,
        )
    else:
        grid = np.ones((1, 2, nsrc, nchan, 2), dtype=np.complex128)
        predict_vis_beam(
            *common,
            vis,
            args["antenna1"],
            args["antenna2"],
            args["ant_type"],
            grid,
            args["pa_lo"],
            args["pa_wt"],
            np.array([0, 1, 1, 0][:ncorr] if ncorr == 4 else [0, 1], dtype=np.int64),
            np.array([0, 1, 0, 1][:ncorr] if ncorr == 4 else [0, 1], dtype=np.int64),
            *smear_args,
        )

    expected = kernel_vis(uvw, freqs, lmn, smearing, exposure)
    np.testing.assert_allclose(vis[:, :, 0], expected, rtol=1e-10)
    np.testing.assert_allclose(vis[:, :, -1], expected, rtol=1e-10)


# --------------------------------------------------------------------------- construction


def test_from_ms_uses_the_median_of_mixed_channel_widths(caplog):
    widths = np.array([1.0e6, 1.0e6, 4.0e6])
    with caplog.at_level("WARNING", logger="simms.skymodel.smearing"):
        smearing = Smearing.from_ms(widths, np.deg2rad(-30.0))
    assert smearing.bw_half == pytest.approx(0.5e6)
    assert "CHAN_WIDTH is not uniform" in caplog.text


def test_a_descending_channel_grid_gives_a_positive_width():
    """CHAN_WIDTH is negative when the frequency axis descends; the factor is not."""
    assert Smearing.from_ms(np.full(4, -8e6), 0.0).bw_half == pytest.approx(4e6)


def test_predicting_a_smeared_model_without_the_exposure_is_an_error(tmp_path):
    sky = tmp_path / "sky.txt"
    sky.write_text("#format: name ra dec stokes_i\nS 0h0m20s -30d10m0s 3.0\n")
    prepared = prepare_skymodel(
        ASCIISkymodel(str(sky), source_schema_file=SCHEMA),
        np.array([1.4e9]),
        0.0,
        np.deg2rad(-30.0),
    )
    prepared = attach_smearing(prepared, Smearing.from_ms(8e6, np.deg2rad(-30.0)))
    with pytest.raises(ValueError, match="EXPOSURE"):
        predict_block(prepared, np.zeros((2, 3)))


# --------------------------------------------------------------------------- end to end


class _SmearTest(InitTest):
    """A MeerKAT track coarse enough in time and frequency to smear visibly."""

    def __init__(self, **ms_kwargs):
        self.test_files = []
        self.ms = self.random_named_directory(suffix=".ms")
        create_ms(
            self.ms,
            telescope_name="meerkat",
            pointing_direction=["J2000", "0deg", "-30deg"],
            dtime=30,
            ntimes=2,
            start_freq="1420MHz",
            dfreq="16MHz",
            nchan=2,
            correlations=["XX", "YY"],
            row_chunks=100000,
            sefd=None,
            column="DATA",
            start_time="2025-03-06T20:00:00",
            smooth=None,
            fit_order=None,
            subarray_range=[0, 60],
            **ms_kwargs,
        )
        self.sky = self.random_named_file(suffix=".txt")
        with open(self.sky, "w") as fh:
            fh.write("#format: name ra dec stokes_i\nS 0h02m00s -30d40m0s 3.0\n")


@pytest.fixture(scope="module")
def smear_ms():
    return _SmearTest()


def _column(ms, name):
    return getattr(xds_from_ms(ms)[0], name).data.compute()


def test_skysim_smears_an_off_axis_source(smear_ms):
    skysim.runit(skysim_opts(smear_ms.ms, ascii_sky=smear_ms.sky, column="PLAIN", smearing="none"))
    skysim.runit(skysim_opts(smear_ms.ms, ascii_sky=smear_ms.sky, column="DATA", smearing="analytic"))
    plain, smeared = _column(smear_ms.ms, "PLAIN"), _column(smear_ms.ms, "DATA")

    ratio = np.abs(smeared) / np.abs(plain)
    assert (ratio <= 1.0 + 1e-9).all(), "smearing can only take amplitude away"
    assert ratio.min() < 0.8, "the fixture is meant to smear appreciably"

    # Deeper on the long baselines: that gradient is exactly what over-subtraction
    # against real data leaves behind.
    uvw = _column(smear_ms.ms, "UVW")
    length = np.linalg.norm(uvw, axis=1)
    short, long_ = length < np.median(length), length > np.median(length)
    assert ratio[long_].mean() < ratio[short].mean()


def test_skysim_passes_the_ms_exposure_channel_width_and_phase_centre(smear_ms):
    """The written column must match a prediction built from the MS's own metadata."""
    skysim.runit(skysim_opts(smear_ms.ms, ascii_sky=smear_ms.sky, column="DATA", smearing="analytic"))

    ds = xds_from_ms(smear_ms.ms)[0]
    spw = xds_from_table(f"{smear_ms.ms}::SPECTRAL_WINDOW")[0]
    field = xds_from_table(f"{smear_ms.ms}::FIELD")[0]
    ra0, dec0 = field.PHASE_DIR.data[0][0].compute()
    freqs = spw.CHAN_FREQ.data[0].compute()
    chan_width = spw.CHAN_WIDTH.data[0].compute()

    prepared = prepare_skymodel(
        ASCIISkymodel(smear_ms.sky, source_schema_file=SCHEMA),
        freqs,
        float(ra0),
        float(dec0),
        ncorr=2,
        polarisation=True,
    )
    prepared = attach_smearing(prepared, Smearing.from_ms(chan_width, float(dec0)))
    expected = predict_block(
        prepared,
        ds.UVW.data.compute(),
        exposure=ds.EXPOSURE.data.compute(),
        out_dtype=ds.DATA.data.dtype,
    )
    np.testing.assert_allclose(_column(smear_ms.ms, "DATA"), expected, rtol=1e-6, atol=1e-8)


def test_smearing_none_reproduces_the_unsmeared_prediction(smear_ms):
    skysim.runit(skysim_opts(smear_ms.ms, ascii_sky=smear_ms.sky, column="DATA", smearing="none"))
    written = _column(smear_ms.ms, "DATA")

    ds = xds_from_ms(smear_ms.ms)[0]
    spw = xds_from_table(f"{smear_ms.ms}::SPECTRAL_WINDOW")[0]
    field = xds_from_table(f"{smear_ms.ms}::FIELD")[0]
    ra0, dec0 = field.PHASE_DIR.data[0][0].compute()
    prepared = prepare_skymodel(
        ASCIISkymodel(smear_ms.sky, source_schema_file=SCHEMA),
        spw.CHAN_FREQ.data[0].compute(),
        float(ra0),
        float(dec0),
        ncorr=2,
        polarisation=True,
    )
    expected = predict_block(prepared, ds.UVW.data.compute(), out_dtype=ds.DATA.data.dtype)
    np.testing.assert_allclose(written, expected, rtol=1e-6, atol=1e-8)


def _off_axis_image(holder, npix=128, cell=30.0 / 3600, offset=40):
    """A single off-axis pixel on the MS phase centre, for the gridder backends."""
    from astropy.io import fits

    header = fits.Header()
    header["CTYPE1"], header["CRVAL1"] = "RA---SIN", 0.0
    header["CDELT1"], header["CRPIX1"], header["CUNIT1"] = -cell, npix // 2 + 1, "deg"
    header["CTYPE2"], header["CRVAL2"] = "DEC--SIN", -30.0
    header["CDELT2"], header["CRPIX2"], header["CUNIT2"] = cell, npix // 2 + 1, "deg"
    header["BUNIT"] = "Jy/pixel"
    data = np.zeros((npix, npix))
    data[npix // 2 - offset, npix // 2] = 3.0
    path = holder.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data, header=header).writeto(path, overwrite=True)
    return path


def test_the_fits_gridder_backend_warns_and_predicts_unsmeared(smear_ms, caplog):
    """Only the per-visibility kernels can carry the factor; the gridder says so."""
    img = _off_axis_image(smear_ms)
    with caplog.at_level("WARNING", logger="skysim"):
        skysim.runit(skysim_opts(smear_ms.ms, fits_sky=img, column="GRID", predict_backend="fft", log_level="WARNING"))
    assert "not applied on the 'fft' FITS backend" in caplog.text

    skysim.runit(skysim_opts(smear_ms.ms, fits_sky=img, column="GRIDNONE", predict_backend="fft", smearing="none"))
    np.testing.assert_allclose(_column(smear_ms.ms, "GRID"), _column(smear_ms.ms, "GRIDNONE"), rtol=1e-6, atol=1e-8)


def test_the_fits_dft_backend_is_smeared(smear_ms):
    """Few bright pixels go through the component kernels, which do carry it."""
    img = _off_axis_image(smear_ms)
    skysim.runit(skysim_opts(smear_ms.ms, fits_sky=img, column="DFTSMEAR", predict_backend="dft"))
    skysim.runit(skysim_opts(smear_ms.ms, fits_sky=img, column="DFTPLAIN", predict_backend="dft", smearing="none"))
    ratio = np.abs(_column(smear_ms.ms, "DFTSMEAR")) / np.abs(_column(smear_ms.ms, "DFTPLAIN"))
    assert (ratio <= 1.0 + 1e-6).all()
    assert ratio.min() < 0.95
