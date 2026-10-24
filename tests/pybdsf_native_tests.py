"""Native PyBDSF output through the importer and the public prediction CLI."""

from pathlib import Path

import numpy as np
import pytest
from casacore.tables import makearrcoldesc, maketabdesc, table
from click.testing import CliRunner
from daskms import xds_from_ms

from simms import SCHEMADIR
from simms.apps.main import cli
from simms.exceptions import ASCIISkymodelError, ASCIISourceError
from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.telescope.generate_ms import create_ms

from . import InitTest

NATIVE = Path(__file__).parent / "fixtures" / "pybdsf-1.14.1-native.gaul"
MAPPER = str(Path(SCHEMADIR) / "bdsf_gaul_source_mapper.yaml")


@pytest.fixture
def params():
    return InitTest()


def native_rows():
    lines = NATIVE.read_text().splitlines()
    header = lines[5].removeprefix("#").split()
    return [dict(zip(header, line.split(), strict=True)) for line in lines[6:]]


def write(params, text):
    path = params.random_named_file(suffix=".gaul")
    Path(path).write_text(text)
    return path


def test_native_gaussians_preserve_physics():
    sources = ASCIISkymodel(str(NATIVE), source_schema_file=MAPPER).sources
    assert len(sources) == 43
    for src, row in zip(sources, native_rows(), strict=True):
        assert src.ra == pytest.approx(np.radians(float(row["RA"]) % 360))
        assert src.dec == pytest.approx(np.radians(float(row["DEC"])))
        assert src.stokes_i == float(row["Total_flux"])
        assert src.emaj == pytest.approx(np.radians(float(row["DC_Maj"])))
        assert src.emin == pytest.approx(np.radians(float(row["DC_Min"])))
        assert src.pa == pytest.approx(np.radians(float(row["DC_PA"])))
        assert src.cont_reffreq == 1.42001e9
        assert src.continuum_coefficients() == [0.0]
        assert not src.is_polarised
        assert src.get_brightness_matrix(np.array([1e9, 2e9]), 2)[0] == pytest.approx([src.stokes_i] * 2)
    assert sources[0].lineno == 6


def spectral_catalogue(reference="1.42001e+09 Hz"):
    return (
        "# Gaussian list for spectral.fits\n"
        + (f'# Reference frequency of the detection ("ch0") image: {reference}\n' if reference else "")
        + "# Gaus_id Source_id RA DEC Total_flux DC_Maj DC_Min DC_PA Spec_Indx Total_Q Total_U Total_V\n"
        + "0 0 -112 -14 1.5 0.01 0.005 45 -0.7 0.1 0.05 0.01\n"
    )


def test_native_spectral_index_and_polarisation(params):
    src = ASCIISkymodel(write(params, spectral_catalogue()), source_schema_file=MAPPER).sources[0]
    freqs = np.array([1.0e9, 1.42001e9, 2e9])
    brightness = src.get_brightness_matrix(freqs, 4)
    scaling = (freqs / 1.42001e9) ** -0.7
    np.testing.assert_allclose(brightness, np.array([1.6, 0.05 + 0.01j, 0.05 - 0.01j, 1.4])[:, None] * scaling)


@pytest.mark.parametrize("reference", [None, "0 Hz", "-2 Hz", "nan Hz", "wrong", "4 s"])
def test_nonflat_spectrum_requires_valid_frequency(params, reference):
    with pytest.raises((ASCIISourceError, ASCIISkymodelError), match="reference frequency"):
        ASCIISkymodel(write(params, spectral_catalogue(reference)), source_schema_file=MAPPER)


def test_native_ignored_columns_do_not_relax_simms_header(params):
    with pytest.raises(ASCIISkymodelError, match="unknown fields"):
        ASCIISkymodel(write(params, "#format: RA DEC Total_flux E_RA\n0 0 1 0\n"), source_schema_file=MAPPER)


@pytest.mark.parametrize("use_screen", [False, True])
def test_public_cli_native_gaul_model_column(params, use_screen):
    """Use all real Gaussians, and independently mapped Simms ASCII as the oracle."""
    ms = params.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="meerkat",
        pointing_direction=["J2000", "247.5deg", "-14deg"],
        dtime=600,
        ntimes=2,
        start_freq="1300MHz",
        dfreq="200MHz",
        nchan=2,
        correlations=["XX", "YY"],
        row_chunks=100000,
        sefd=None,
        column="MODEL_DATA",
        start_time="2025-03-06T20:00:00",
        smooth=None,
        fit_order=None,
        subarray_range=[0, 3],
    )
    with table(ms, readonly=False, ack=False) as tab:
        tab.addcols(maketabdesc(makearrcoldesc("MODEL_DATA", 0j, ndim=2, shape=[2, 2])))
        sentinel = np.full((tab.nrows(), 2, 2), 3 + 4j)
        tab.putcol("MODEL_DATA", sentinel)
    canonical = "#format: ra dec stokes_i emaj emin pa cont_reffreq cont_coeff_1\n"
    canonical += (
        "\n".join(
            " ".join(row[k] for k in ["RA", "DEC", "Total_flux", "DC_Maj", "DC_Min", "DC_PA"]) + " 1.42001e9 0"
            for row in native_rows()
        )
        + "\n"
    )
    canonical_path = write(params, canonical)
    extra = []
    if use_screen:
        from astropy.io import fits

        from .wsclean_aterm_tests import config, screen_file

        screen, _ = screen_file(params, times=(0.0,), freqs=(1.3e9, 1.5e9), nant=1, timeless=True)
        with fits.open(screen, mode="update") as hdus:
            hdus[0].header["CRVAL1"] = 247.5
            hdus[0].header["CRVAL2"] = -14.0
            hdus[0].header["CDELT1"] = -0.1
            hdus[0].header["CDELT2"] = 0.1
        extra = ["--aterm-config", config(params, [screen])]
    for sky, species, column in [
        (str(NATIVE), ["--ascii-species", "bdsf_gaul"], "MODEL_GAUL"),
        (canonical_path, [], "EXPECTED"),
    ]:
        result = CliRunner().invoke(
            cli,
            [
                "skysim",
                "--ascii-sky",
                sky,
                *species,
                "--column",
                column,
                "--smearing",
                "none",
                "--nworkers",
                "1",
                *extra,
                ms,
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
    if not use_screen:
        # Every advertised species must resolve its shipped mapper through the CLI.
        for species, header in [
            ("bdsf_srl", "Source_id RA DEC Total_flux"),
            ("aegean", "source ra dec int_flux"),
            ("wsclean", "Name RA DEC I"),
        ]:
            sky = write(params, f"#format: {header}\nS 247.5 -14 2.0\n")
            result = CliRunner().invoke(
                cli,
                [
                    "skysim",
                    "--ascii-sky",
                    sky,
                    "--ascii-species",
                    species,
                    "--column",
                    "SPECIES",
                    "--smearing",
                    "none",
                    "--nworkers",
                    "1",
                    ms,
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            prediction = xds_from_ms(ms, columns=["SPECIES"], group_cols=[])[0].SPECIES.data.compute()
            np.testing.assert_allclose(prediction, 2.0, atol=1e-10)
    ds = xds_from_ms(ms, columns=["MODEL_DATA", "MODEL_GAUL", "EXPECTED"], group_cols=[])[0]
    actual = ds.MODEL_GAUL.data.compute()
    assert actual.shape[-1] == 2
    assert np.isfinite(actual).all() and np.abs(actual).max() > 0
    np.testing.assert_allclose(actual, ds.EXPECTED.data.compute(), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(ds.MODEL_DATA.data.compute(), sentinel)


def test_native_gaul_matches_tigger_reader(params, monkeypatch):
    """Optional independent oracle; run with astro-tigger-lsm==1.8.0 installed."""
    monkeypatch.setenv("MPLCONFIGDIR", params.random_named_directory())
    gaul = pytest.importorskip("Tigger.Models.Formats.PyBDSMGaul")
    for path in [str(NATIVE), write(params, spectral_catalogue())]:
        actual = ASCIISkymodel(path, source_schema_file=MAPPER).sources
        expected = gaul.load(path).sources
        assert len(actual) == len(expected)
        for source, ref in zip(actual, expected, strict=True):
            assert source.ra == pytest.approx(ref.pos.ra % (2 * np.pi))
            assert source.dec == pytest.approx(ref.pos.dec)
            for stokes in "IQUV":
                assert getattr(source, f"stokes_{stokes.lower()}", 0) == pytest.approx(getattr(ref.flux, stokes))
            if ref.shape is not None:
                assert source.emaj == pytest.approx(ref.shape.ex)
                assert source.emin == pytest.approx(ref.shape.ey)
                assert source.pa == pytest.approx(ref.shape.pa)
            else:
                assert source.emaj == source.emin == 0
            freqs = np.array([1e9, 1.42001e9, 2e9])
            scale = ref.spectrum.normalized_intensity(freqs) if ref.spectrum else np.ones(3)
            np.testing.assert_allclose(source.get_brightness_matrix(freqs, 4)[0], (ref.flux.I + ref.flux.Q) * scale)
