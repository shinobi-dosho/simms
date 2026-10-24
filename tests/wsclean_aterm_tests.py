"""Independent WSClean diagonal screen fixtures and RIME regression tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy import units
from astropy.io import fits
from astropy.wcs import WCS

from simms.skymodel.fits_skies import component_sky_from_fits_dft, predict_fits_block, prepare_fits_sky
from simms.skymodel.mstools import PreparedSky, predict_block
from simms.skymodel.wsclean_aterms import (
    DiagonalScreens,
    attach_component_screens,
    attach_fits_screens,
    nearest_indices,
    read_parset,
)

from . import InitTest
from .aterm_tests import CELL_DEG, DEC0, NPIX, RA0, write_flat_image


@pytest.fixture
def params():
    return InitTest()


@pytest.fixture
def observation():
    from .primary_beam_tests import _Fixtures

    return _Fixtures()


def screen_file(params, times=(100.0, 120.0), freqs=(1e9, 1.2e9), nant=2, timeless=False):
    """An independently constructed complex, antenna-dependent linear spatial screen."""
    header = fits.Header()
    for i, (ctype, val, delta, unit) in enumerate(
        [
            ("RA---SIN", np.degrees(RA0), -CELL_DEG, "deg"),
            ("DEC--SIN", np.degrees(DEC0), CELL_DEG, "deg"),
            ("MATRIX", 0, 1, ""),
            ("ANTENNA", 0, 1, ""),
            ("FREQ", freqs[0], freqs[1] - freqs[0] if len(freqs) > 1 else 1e6, "Hz"),
            ("TIME", times[0], times[1] - times[0] if len(times) > 1 else 1, "s"),
        ][: 5 if timeless else 6],
        1,
    ):
        header[f"CTYPE{i}"] = ctype
        header[f"CRVAL{i}"] = val
        header[f"CDELT{i}"] = delta
        header[f"CRPIX{i}"] = NPIX // 2 + 1 if i < 3 else 1
        if unit:
            header[f"CUNIT{i}"] = unit
    data = np.empty((len(times), len(freqs), nant, 4, NPIX, NPIX), dtype=np.float32)
    yy, xx = np.indices((NPIX, NPIX))
    for ti in range(len(times)):
        for fi in range(len(freqs)):
            for ai in range(nant):
                # Every axis contributes, with distinct XX and YY complex phases.
                gx = 1 + 0.2 * ti + 0.1 * fi + 0.3 * ai + 0.001 * xx + 1j * (0.2 + 0.002 * yy)
                gy = 0.7 + 0.1 * ti + 0.2 * fi + 0.15 * ai + 0.002 * yy - 1j * (0.1 + 0.001 * xx)
                data[ti, fi, ai] = [gx.real, gx.imag, gy.real, gy.imag]
    path = params.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data[0] if timeless else data, header).writeto(path, overwrite=True)
    return path, data


def config(params, images, extra="", term="diagonal"):
    path = params.random_named_file(suffix=".parset")
    names = ", ".join(repr(str(Path(image).relative_to(Path(path).parent))) for image in images)
    Path(path).write_text(f"# a comment\naterms = [{term}]\n{term}.images = [{names}] # end\n" + extra)
    return path


def point_sky(freqs=(1e9, 1.2e9), ncorr=4):
    brightness = np.array([2.3, 0.4 + 0.2j, 0.4 - 0.2j, 1.7]) if ncorr == 4 else np.array([2.3, 1.7])
    return PreparedSky(
        lmn=np.zeros((1, 3)),
        gauss_shape=np.zeros((1, 3)),
        is_gauss=np.zeros(1, dtype=bool),
        bmat=np.broadcast_to(brightness[None, :, None], (1, ncorr, len(freqs))).copy(),
        lightcurve=np.ones((1, 1)),
        unique_times=None,
        freqs=np.array(freqs),
        uniform_freqs=True,
        ncorr=ncorr,
        polarisation=True,
    )


def test_complex_rime_nearest_and_reversed_baseline(params):
    path, data = screen_file(params)
    screens = DiagonalScreens.from_parset(config(params, [path]), 2)
    sky = attach_component_screens(point_sky(), screens, RA0, DEC0, 1)
    times = np.array([90.0, 110.0, 111.0, 130.0])
    a1, a2 = np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])
    actual = predict_block(sky, np.zeros((4, 3)), times=times, antenna1=a1, antenna2=a2)
    expected = np.empty_like(actual)
    for r, ti in enumerate([0, 0, 1, 1]):
        for fi in range(2):
            d = data[ti, fi, :, :, NPIX // 2, NPIX // 2]
            gains = np.stack((d[:, 0] + 1j * d[:, 1], d[:, 2] + 1j * d[:, 3]), -1)
            expected[r, fi] = (
                gains[a1[r]][:, None] * point_sky().bmat[0, :, fi].reshape(2, 2) * gains[a2[r]].conj()[None, :]
            ).ravel()
    np.testing.assert_allclose(actual, expected, rtol=2e-7)
    # Channel selection and arbitrary row partitioning preserve the same model.
    chunk = predict_block(
        sky.select_channels(np.array([1])), np.zeros((1, 3)), times=times[2:3], antenna1=a1[2:3], antenna2=a2[2:3]
    )
    np.testing.assert_allclose(chunk, actual[2:3, 1:2], rtol=1e-12)


def test_spatial_bilinear_absolute_wcs_and_zero_padding(params):
    path, _data = screen_file(params)
    screens = DiagonalScreens([path], 2)
    cel = WCS(fits.getheader(path)).celestial
    x, y = np.array([10.25, 32.0, -2.0]), np.array([20.75, 32.0, 10.0])
    ra, dec = np.radians(cel.all_pix2world(x, y, 0))
    sample = screens.sample(0, 0, 1, ra, dec)
    np.testing.assert_allclose(
        sample[0],
        [1.3 + 0.001 * x[0] + 1j * (0.2 + 0.002 * y[0]), 0.85 + 0.002 * y[0] - 1j * (0.1 + 0.001 * x[0])],
        rtol=1e-7,
    )
    assert np.all(sample[-1] == 0)
    # Rephasing the component model must not move the screen centre.
    from simms.skymodel.fits_skies import lm_to_radec
    from simms.utilities import radec2lm

    shifted_ra, shifted_dec = RA0 + 0.01, DEC0 - 0.01
    ell, emm = radec2lm(shifted_ra, shifted_dec, ra[:1], dec[:1])
    sky = replace(point_sky(), lmn=np.array([[ell[0], emm[0], np.sqrt(1 - ell[0] ** 2 - emm[0] ** 2) - 1]]))
    attached = attach_component_screens(sky, screens, shifted_ra, shifted_dec, 1)
    np.testing.assert_allclose(attached.beam_grid[1, 0, 0, 0], sample[0], rtol=2e-7)
    np.testing.assert_allclose(lm_to_radec(ell, emm, shifted_ra, shifted_dec), [ra[:1], dec[:1]], atol=1e-12)


def test_descending_axes_nondefault_crpix_and_timeless(params):
    path, data = screen_file(params, times=(120.0, 100.0), freqs=(1.2e9, 1e9))
    with fits.open(path, mode="update") as hdus:
        for axis in (5, 6):
            hdus[0].header[f"CRPIX{axis}"] = 3
            hdus[0].header[f"CRVAL{axis}"] += 2 * hdus[0].header[f"CDELT{axis}"]
    screens = DiagonalScreens([path], 2)
    np.testing.assert_equal(screens.times, [100.0, 120.0])
    np.testing.assert_equal(screens.freqs, [1e9, 1.2e9])
    sample = screens.sample(0, 0, 0, np.array([RA0]), np.array([DEC0]))
    d = data[1, 1, 0, :, 32, 32]
    np.testing.assert_allclose(sample, [[d[0] + 1j * d[1], d[2] + 1j * d[3]]])
    timeless, _ = screen_file(params, times=(0.0,), freqs=(1e9,), nant=1, timeless=True)
    only = DiagonalScreens([timeless], 17)
    assert only.nant_screen == 1
    with pytest.raises(ValueError, match="Timeless"):
        DiagonalScreens([timeless, timeless], 17)


def test_multiple_files_gaps_and_duplicates(params):
    one, _ = screen_file(params, times=(100.0, 120.0))
    two, _ = screen_file(params, times=(200.0, 220.0))
    screens = DiagonalScreens([two, one], 2)
    np.testing.assert_equal(screens.times, [100.0, 120.0, 200.0, 220.0])
    np.testing.assert_equal(nearest_indices(screens.times, [0.0, 110.0, 160.0, 161.0, 1000.0]), [0, 0, 1, 2, 3])
    with pytest.raises(ValueError, match="Duplicate"):
        DiagonalScreens([one, one], 2)


@pytest.mark.parametrize("option", ["aterms=[tec]\ntec.images=[]", "diagonal.unknown=3", "diagonal.window=bad"])
def test_reject_unsupported_config(params, option):
    path, _ = screen_file(params)
    parset = config(params, [path])
    if option.startswith("aterms"):
        Path(parset).write_text(option)
    else:
        with open(parset, "a") as handle:
            handle.write(option)
    with pytest.raises(ValueError):
        read_parset(parset)


@pytest.mark.parametrize(
    "key,value", [("CTYPE3", "STOKES"), ("CDELT5", 0), ("CRVAL5", -1e9), ("NAXIS3", 3), ("CUNIT6", "d")]
)
def test_reject_malformed_screens(params, key, value):
    path, _ = screen_file(params)
    if key == "CUNIT6":
        # Days are convertible; unlike time units such as metres they are valid.
        value = "m"
    with fits.open(path, mode="update") as hdus:
        if key == "NAXIS3":
            hdus[0].data = hdus[0].data[:, :, :, :3]
        else:
            hdus[0].header[key] = value
    with pytest.raises((ValueError, units.UnitConversionError)):
        DiagonalScreens([path], 2)


def test_screen_memory_caps(params):
    path, _ = screen_file(params)
    screens = DiagonalScreens([path], 2)
    with pytest.raises(MemoryError, match="beam-grid-max-gib"):
        attach_component_screens(point_sky(), screens, RA0, DEC0, 1e-20)


def test_gridder_matches_independent_component_prediction(params):
    path, _ = screen_file(params)
    screens = DiagonalScreens([path], 2)
    image = write_flat_image(params, nstokes=4, stokes_values=[(2.0, 0.3, 0.4, 0.2)] * 3)
    freqs = np.array([0.9e9, 1.1e9, 1.3e9])
    image_sky = prepare_fits_sky(image, RA0, DEC0, freqs, 1e6, 4, polarisation=True, nrow=4, backend="fft")
    dft = prepare_fits_sky(image, RA0, DEC0, freqs, 1e6, 4, polarisation=True, nrow=4, backend="dft")
    comp = attach_component_screens(component_sky_from_fits_dft(dft), screens, RA0, DEC0, 1)
    grid = attach_fits_screens(image_sky, screens, RA0, DEC0, 1)
    uvw = np.random.default_rng(43).normal(0, 100, (4, 3))
    times = np.array([99.0, 110.0, 111.0, 200.0])
    a1, a2 = np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])
    actual = predict_fits_block(grid, uvw, times=times, antenna1=a1, antenna2=a2)
    expected = predict_block(comp, uvw, times=times, antenna1=a1, antenna2=a2)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    chunk = predict_fits_block(
        grid.select_channels(np.array([1])), uvw[2:], times=times[2:], antenna1=a1[2:], antenna2=a2[2:]
    )
    np.testing.assert_allclose(chunk, actual[2:, 1:2], atol=1e-10)
    with pytest.raises(MemoryError):
        attach_fits_screens(image_sky, screens, RA0, DEC0, 1e-20)


def test_parset_named_term_quotes_and_supported_windows(params):
    path, _ = screen_file(params)
    for window in ("tukey", "hann", "raised_hann", "rectangular", "gaussian"):
        parset = config(params, [path], f"screen.type=diagonal\nscreen.window={window}", term="screen")
        assert read_parset(parset) == [str(Path(path).resolve())]


def test_skysim_ascii_wsclean_and_fits_bridge(params, monkeypatch, observation):
    from daskms import xds_from_ms

    from simms.apps import skysim
    from simms.apps.skysim import runit

    from . import skysim_opts
    from .predict_wsclean_tests import HEADER
    from .primary_beam_tests import _write_ascii_sky

    fx = observation
    # Imported sky-frame screens need neither antenna mount nor telescope labels.
    original_table_reader = skysim.xds_from_table

    def metadata_reader(name, *args, **kwargs):
        datasets = original_table_reader(name, *args, **kwargs)
        if name.endswith("::ANTENNA"):
            return [ds.drop_vars([c for c in ("MOUNT", "TELESCOPE_NAME") if c in ds]) for ds in datasets]
        return datasets

    monkeypatch.setattr(skysim, "xds_from_table", metadata_reader)
    path, _ = screen_file(params, times=(0.0,), freqs=(1.42e9, 1.424e9), nant=1, timeless=True)
    parset = config(params, [path])
    ascii_sky = _write_ascii_sky(fx)
    wsclean = fx.random_named_file(suffix=".txt")
    Path(wsclean).write_text(HEADER + "\np0,POINT,01:00:00.0,-31.00.00.0,5.0,[],true,1420000000,,,\n")
    data = np.zeros((NPIX, NPIX))
    data[32, 32] = 5.0
    from .predict_fits_tests import make_header

    image = fx.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data, make_header(NPIX, cell=CELL_DEG)).writeto(image, overwrite=True)
    outputs = []
    for inputs in (
        dict(ascii_sky=ascii_sky),
        dict(wsclean_sky=wsclean),
        dict(fits_sky=image, predict_backend="dft"),
        dict(fits_sky=image, predict_backend="fft"),
    ):
        runit(skysim_opts(fx.ms, aterm_config=parset, smearing="none", chan_chunks=1, row_chunks=7, **inputs))
        outputs.append(xds_from_ms(fx.ms, columns=["DATA"], group_cols=[])[0].DATA.data.compute())
    for output in outputs[1:]:
        np.testing.assert_allclose(output, outputs[0], atol=2e-5, rtol=2e-6)


def test_noise_only_validates_screen_options_before_write(params, observation):
    from simms.apps.skysim import runit

    from . import skysim_opts

    fx = observation
    invalid = config(params, [], "")
    with pytest.raises(ValueError, match="images"):
        runit(skysim_opts(fx.ms, aterm_config=invalid))
    with pytest.raises(RuntimeError, match="primary-beam"):
        runit(skysim_opts(fx.ms, aterm_config=invalid, primary_beam="unused"))
    for opts in (dict(beam_jones="full"), dict(fits_beam_mode="average")):
        with pytest.raises(RuntimeError, match="requires diagonal"):
            runit(skysim_opts(fx.ms, aterm_config=invalid, **opts))


def test_subsampling_reselects_screen_time_and_frequency(params):
    from simms.skymodel.smearing import SubsampleSmearing

    path, _ = screen_file(params)
    screens = DiagonalScreens([path], 2)
    image = write_flat_image(params)
    prepared = prepare_fits_sky(image, RA0, DEC0, np.array([1.1e9]), 1e8, 2, polarisation=False, nrow=1, backend="fft")
    prepared = attach_fits_screens(prepared, screens, RA0, DEC0, 1)
    sub = SubsampleSmearing.from_ms(np.array([2e8]), DEC0, 2, 2)
    smeared = replace(prepared, subsample=sub)
    args = dict(antenna1=np.array([0]), antenna2=np.array([1]))
    actual = predict_fits_block(smeared, np.zeros((1, 3)), times=np.array([110.0]), exposure=np.array([4.0]), **args)
    expected = []
    for time in (109.0, 111.0):
        for freq in (1.05e9, 1.15e9):
            expected.append(
                predict_fits_block(
                    replace(prepared, chan_freqs=np.array([freq])), np.zeros((1, 3)), times=np.array([time]), **args
                )
            )
    np.testing.assert_allclose(actual, np.mean(expected, axis=0), rtol=1e-12)
    assert (
        np.max(np.abs(actual - predict_fits_block(prepared, np.zeros((1, 3)), times=np.array([110.0]), **args))) > 0.1
    )


@pytest.mark.parametrize("mounts", ["rotating", "fixed", "mixed"])
def test_export_axes_artifacts_and_per_antenna_mounts(params, mounts, observation):
    from casacore.tables import table

    from simms.apps.primary_beam import runit
    from simms.skymodel.beams import JimBeamProvider, array_lonlat, parallactic_angle, resolve_beam
    from simms.utilities import radec2lm

    from .primary_beam_tests import _opts

    fx = observation
    with table(f"{fx.ms}::ANTENNA", readonly=False, ack=False) as ant:
        count = ant.nrows()
        mount_values = ["ALT-AZ"] * count
        if mounts == "fixed":
            mount_values[:] = ["EQUATORIAL"] * count
        elif mounts == "mixed":
            mount_values[1::2] = ["EQUATORIAL"] * len(mount_values[1::2])
        ant.putcol("MOUNT", mount_values)
        lon, lat = array_lonlat(ant.getcol("POSITION"))
    # Change the tracking centre: POINTING remains authoritative for export.
    with table(f"{fx.ms}::FIELD", readonly=False, ack=False) as field:
        phase = field.getcell("PHASE_DIR", 0)
        phase[0, 0] += 0.02
        field.putcell("PHASE_DIR", 0, phase)
    output = fx.random_named_file(suffix=".fits")
    parset = str(Path(output).with_suffix(".parset"))
    fx.register(parset)
    result = runit(
        _opts(
            "to-fits",
            ms=fx.ms,
            fits_format="wsclean",
            beam_pattern="L",
            output=output,
            npix=32,
            pixel_size="3arcmin",
            beam_pa_step=5,
        )
    )
    assert result.output == output
    assert result.files == [output, parset]
    assert result.ms == fx.ms
    assert read_parset(parset) == [str(Path(output).resolve())]
    assert '"' not in Path(parset).read_text() and "'" not in Path(parset).read_text()
    invalid_output = fx.random_named_file(suffix=" beam.fits")
    with pytest.raises(ValueError, match="paths"):
        runit(_opts("to-fits", ms=fx.ms, fits_format="wsclean", output=invalid_output))
    screen = DiagonalScreens.from_parset(parset, count)
    with fits.open(output) as hdus:
        h, data = hdus[0].header, hdus[0].data
        assert data.dtype.kind == "f" and data.dtype.itemsize == 4
        assert data.shape[2] == count
        assert data.shape[0] == 1 if mounts == "fixed" else data.shape[0] >= 2
        assert h["CDELT1"] < 0 < h["CDELT2"]
        np.testing.assert_allclose([h["CRVAL1"], h["CRVAL2"]], np.degrees([RA0, DEC0]), atol=1e-8)
    # WCS position must evaluate the analytic sky-frame beam with its actual mount.
    x, y = np.array([9.0]), np.array([20.0])
    ra, dec = np.radians(screen.celestial.all_pix2world(x, y, 0))
    ell, emm = radec2lm(RA0, DEC0, ra, dec)
    provider = resolve_beam("L", "L")
    assert isinstance(provider, JimBeamProvider)
    for antenna in (0, 1):
        ti = len(screen.times) - 1
        chi = parallactic_angle(screen.times[ti : ti + 1], RA0, DEC0, lon, lat)
        if mount_values[antenna] == "EQUATORIAL":
            chi[:] = 0
        expected = provider.voltage(ell, emm, screen.freqs[:1], float(chi[0]))[0, :, 0]
        np.testing.assert_allclose(screen.sample(ti, 0, antenna, ra, dec), expected, atol=2e-6)


@pytest.mark.parametrize("variable_shapes", [False, True])
def test_selected_data_description_preserves_seeded_noise_with_screens_and_gains(params, observation, variable_shapes):
    from casacore.tables import table
    from daskms import xds_from_ms

    from simms.apps.skysim import runit
    from simms.skymodel.mstools import vis_noise_from_sefd_and_ms

    from . import skysim_opts
    from .primary_beam_tests import _write_ascii_sky

    fx = observation
    # Only DDID=1 occurs in MAIN; it selects POL=1 and SPW=0 (or SPW=1 when
    # unused rows carry different shapes). Unused POL=0 is either a valid
    # four-correlation setup or deliberately unsuitable Stokes correlations.
    with table(f"{fx.ms}::POLARIZATION", readonly=False, ack=False) as pol:
        pol.addrows(1)
        for col in pol.colnames():
            pol.putcell(col, 1, pol.getcell(col, 0))
        pol.putcell("CORR_TYPE", 0, [9, 10, 11, 12] if variable_shapes else [1, 2])
        if variable_shapes:
            pol.putcell("NUM_CORR", 0, 4)
            pol.putcell("CORR_PRODUCT", 0, np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32))
    if variable_shapes:
        with table(f"{fx.ms}::SPECTRAL_WINDOW", readonly=False, ack=False) as spw:
            spw.addrows(1)
            for col in spw.colnames():
                spw.putcell(col, 1, spw.getcell(col, 0))
            for col in ("CHAN_FREQ", "CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION"):
                spw.putcell(col, 0, spw.getcell(col, 0)[:1])
            spw.putcell("NUM_CHAN", 0, 1)
    with table(f"{fx.ms}::DATA_DESCRIPTION", readonly=False, ack=False) as dd:
        dd.addrows(1)
        for col in dd.colnames():
            dd.putcell(col, 1, dd.getcell(col, 0))
        dd.putcell("POLARIZATION_ID", 1, 1)
        if variable_shapes:
            dd.putcell("SPECTRAL_WINDOW_ID", 1, 1)
    with table(fx.ms, readonly=False, ack=False) as main:
        main.putcol("DATA_DESC_ID", [1] * main.nrows())
        exposure = main.getcell("EXPOSURE", 0)
    np.testing.assert_allclose(vis_noise_from_sefd_and_ms(fx.ms, 100.0, spw_id=1), 100.0 / np.sqrt(2 * exposure * 4e6))
    path, _ = screen_file(params, times=(0.0,), freqs=(1.42e9, 1.424e9), nant=1, timeless=True)
    parset = config(params, [path])
    unity_path, _ = screen_file(params, times=(0.0,), freqs=(1.42e9, 1.424e9), nant=1, timeless=True)
    with fits.open(unity_path, mode="update") as hdus:
        hdus[0].data[:] = 0
        hdus[0].data[:, :, [0, 2]] = 1
    unity_config = config(params, [unity_path])
    gains = params.random_named_file(suffix=".yaml")
    Path(gains).write_text(
        "gains:\n  terms: [G]\n  spec:\n    - label: G\n      type: diagonal\n"
        "      axes: [time]\n      complex: true\n      period: {time: 120}\n      amplitude: 0.1\n"
    )
    sky = _write_ascii_sky(fx)
    residuals = []
    clean_models = []
    for screen, corruption in ((unity_config, None), (parset, None), (parset, gains)):
        output = []
        for sefd in (None, 100.0):
            runit(
                skysim_opts(
                    fx.ms,
                    ascii_sky=sky,
                    aterm_config=screen,
                    spw_id=1,
                    corruptions=corruption,
                    seed_gains=7,
                    seed_noise=13,
                    sefd=sefd,
                    smearing="none",
                    row_chunks=7,
                    chan_chunks=1,
                )
            )
            output.append(xds_from_ms(fx.ms, columns=["DATA"], group_cols=[])[0].DATA.data.compute())
        clean_models.append(output[0])
        residuals.append(output[1] - output[0])
    assert np.max(np.abs(clean_models[0] - clean_models[1])) > 0.1
    assert np.max(np.abs(clean_models[1] - clean_models[2])) > 0.1
    for residual in residuals[1:]:
        np.testing.assert_allclose(residual, residuals[0], atol=1e-6, rtol=0)


@pytest.mark.parametrize("backend", ["fft", "perchan"])
def test_imported_screen_poly_spectrum_matches_component_dft(params, backend):
    from .predict_fits_tests import make_header

    path, _ = screen_file(params)
    screens = DiagonalScreens([path], 2)
    image = write_flat_image(params, nstokes=4, stokes_values=[(2.0, 0.3, 0.4, 0.2)] * 3)
    spi = params.random_named_file(suffix=".fits")
    fits.PrimaryHDU(np.linspace(-1.5, 0.5, NPIX * NPIX).reshape(NPIX, NPIX), make_header(NPIX, cell=CELL_DEG)).writeto(
        spi, overwrite=True
    )
    freqs = np.array([0.9e9, 1.1e9, 1.3e9])
    kwargs = dict(ncorr=4, nrow=4, polarisation=True, linear_basis=True, spi_maps=[spi], ref_freq=1.1e9)
    dft = prepare_fits_sky(image, RA0, DEC0, freqs, 1e6, backend="dft", **kwargs)
    comp = attach_component_screens(component_sky_from_fits_dft(dft), screens, RA0, DEC0, 1)
    prepared = prepare_fits_sky(image, RA0, DEC0, freqs, 1e6, backend=backend, **kwargs)
    prepared = attach_fits_screens(prepared, screens, RA0, DEC0, 1)
    uvw = np.random.default_rng(48).normal(0, 100, (4, 3))
    rows = dict(
        times=np.array([99.0, 110.0, 111.0, 200.0]), antenna1=np.array([0, 1, 0, 1]), antenna2=np.array([1, 0, 1, 0])
    )
    expected = predict_block(comp, uvw, **rows)
    actual = predict_fits_block(prepared, uvw, **rows)
    np.testing.assert_allclose(actual, expected, atol=3e-6 * np.abs(expected).max(), rtol=0)
