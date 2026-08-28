import math
import os

import numpy as np
import pytest
from astropy.coordinates import Angle, Latitude, Longitude

from simms.exceptions import ASCIISkymodelError, ASCIISourceError
from simms.skymodel.ascii_skies import (
    ASCIISkymodel,
    ASCIISource,
    ASCIISourceSchema,
)
from simms.skymodel.source_factory import SourceType
from simms.utilities import load_yaml

from . import InitTest


class InitThisTest(InitTest):
    def __init__(self):
        # important to have this first, else files/dirs created before it will not be tracked
        self.test_files = []

    def write_temp_file(self, content: str, suffix: str = ".txt") -> str:
        path = self.random_named_file(suffix=suffix)
        with open(path, "w") as f:
            f.write(content)
        return path


@pytest.fixture
def params():
    return InitThisTest()


def load_default_schema() -> ASCIISourceSchema:
    """Load the default schema via ASCIISkymodel helper but without reading a file of sources."""
    # Create a tiny valid model file to force schema load, then ignore parsing
    from simms import SCHEMADIR

    schema_path = os.path.join(SCHEMADIR, "source_schema.yaml")
    schema = ASCIISourceSchema(**load_yaml(schema_path))
    return schema


def test_source_type_validation():
    src = SourceType("Test Source", required=["ra", "dec", "a|b|c", "e|f"])
    with pytest.raises(ASCIISourceError):
        src.is_valid(["ra", "dec", "a"], raise_exception=True)

    # 'a' satisfies 'a|b|c' but nothing satisfies 'e|f'
    assert not src.is_valid(["ra", "dec", "a"])
    assert src.is_valid(["ra", "dec", "a", "e"])
    assert not src.is_valid(["dec", "a", "e"])


def test_set_source_param_conversions():
    schema = load_default_schema()
    src = ASCIISource(schema)

    # RA/Dec should convert to radians using schema units (deg)
    src.set_source_param("ra", "180")
    src.set_source_param("dec", "-30")
    src.set_source_param("stokes_i", "1.5")
    src.set_source_param("name", "SrcA")

    assert pytest.approx(src.ra, rel=0, abs=1e-12) == math.pi
    assert pytest.approx(src.dec, rel=0, abs=1e-12) == Latitude("-30 deg").to("rad").value
    assert pytest.approx(src.stokes_i, rel=0, abs=1e-12) == 1.5
    assert src.name == "SrcA"


def test_alias_mapper():
    schema = load_default_schema()
    src = ASCIISource(schema)

    a2f = src.alias_to_field_mapper()
    f2a = src.field_to_alias_mapper()

    # Default schema has no aliases set, so mapping is identity both ways
    for key in ["ra", "dec", "stokes_i", "name"]:
        assert a2f[key] == key
        assert f2a[key] == key


def test_finalise_point_vs_extended_and_polarisation():
    schema = load_default_schema()

    # Point, unpolarised
    s1 = ASCIISource(schema)
    for k, v in {"ra": "0", "dec": "0", "stokes_i": 1.0}.items():
        s1.set_source_param(k, v)
    s1.finalise()
    assert s1.is_point is True
    assert s1.is_polarised is False

    # Extended, polarised (I and Q specified)
    s2 = ASCIISource(schema)
    for k, v in {"ra": 0, "dec": 0, "stokes_i": 2.0, "stokes_q": 0.5, "emaj": "0.1", "emin": "0.05"}.items():
        s2.set_source_param(k, v)
    s2.finalise()
    assert s2.is_point is False
    assert s2.is_polarised is True


def test_circular_polarisation_errors():
    schema = load_default_schema()
    # Circular polarisation polarised (I, Q, U specified)
    src = ASCIISource(schema)
    for k, v in {
        "ra": 0,
        "dec": 0,
        "stokes_i": 2.0,
        "stokes_q": 0.5,
        "stokes_u": 0.1,
        "stokes_v": 0.01,
        "emaj": "0.1",
        "emin": "0.05",
    }.items():
        src.set_source_param(k, v)
    src.finalise()

    freqs = [1e6, 1.2e6, 1.3e6]
    bmatrix = src.get_brightness_matrix(chan_freqs=freqs, ncorr=2, linear_basis=True)
    assert pytest.approx(src.stokes.I, abs=1e-6) == src.stokes_i

    # check as if basis was circular. This should fail
    with pytest.raises(AssertionError):
        assert pytest.approx(src.stokes.V, abs=1e-6) == src.stokes_q
        assert pytest.approx(src.stokes.Q, abs=1e-6) == src.stokes_u
        assert pytest.approx(src.stokes.U, abs=1e-6) == src.stokes_v
    assert bmatrix.shape == (2, 3)

    # set correct basis, it should work now
    bmatrix = src.get_brightness_matrix(chan_freqs=freqs, ncorr=4, linear_basis=False)
    assert pytest.approx(src.stokes.V, abs=1e-6) == src.stokes_q
    assert pytest.approx(src.stokes.Q, abs=1e-6) == src.stokes_u
    assert pytest.approx(src.stokes.U, abs=1e-6) == src.stokes_v
    assert bmatrix.shape == (4, 3)


def test_finalise_transient_field_validation():
    schema = load_default_schema()

    # No transient fields -> not transient
    s1 = ASCIISource(schema)
    for k, v in {"ra": 0, "dec": 0, "stokes_i": 1.0}.items():
        s1.set_source_param(k, v)
    s1.finalise()
    assert s1.is_transient is False

    # Partial transient fields -> error (ensure time values include units since schema has null units)
    s2 = ASCIISource(schema)
    for k, v in {"ra": 0, "dec": 0, "stokes_i": 1.0, "transient_start": "0 s", "transient_period": "10 s"}.items():
        s2.set_source_param(k, v)
    with pytest.raises(ASCIISourceError):
        s2.finalise()

    # All transient fields -> transient True
    s3 = ASCIISource(schema)
    for k, v in {
        "ra": 0,
        "dec": 0,
        "stokes_i": 1.0,
        "transient_start": "0 s",
        "transient_period": "10 s",
        "transient_ingress": "2 s",
        "transient_absorb": 0.1,
    }.items():
        s3.set_source_param(k, v)
    s3.finalise()
    assert s3.is_transient is True


def test_skymodel_header_required_and_format_errors(params):
    # Missing #format: line
    bad1 = params.write_temp_file("ra dec stokes_i\n0 0 1\n")
    with pytest.raises(ASCIISkymodelError):
        ASCIISkymodel(bad1)

    # Missing required headers
    bad2 = params.write_temp_file("#format: ra dec\n0 0\n")
    with pytest.raises(ASCIISkymodelError):
        ASCIISkymodel(bad2)

    # Row length mismatch
    bad3 = params.write_temp_file("#format: ra dec stokes_i\n0 0\n")
    with pytest.raises(ASCIISkymodelError):
        ASCIISkymodel(bad3)


def test_skymodel_parsing_space_delimited(params):
    content = """
#format: ra dec stokes_i name
0 0 1 Src1
# a comment that should be skipped
180 -30 2.5 Src2
""".strip()
    path = params.write_temp_file(content)

    model = ASCIISkymodel(path)
    assert len(model.sources) == 2

    s1, s2 = model.sources
    assert pytest.approx(s1.ra, abs=1e-12) == 0.0
    assert pytest.approx(s1.dec, abs=1e-12) == 0.0
    assert pytest.approx(s1.stokes_i, abs=1e-12) == 1.0
    assert s1.name == "Src1"

    assert pytest.approx(s2.ra, abs=1e-12) == math.pi
    assert pytest.approx(s2.dec, abs=1e-12) == Angle("-30 deg").to("rad").value
    assert pytest.approx(s2.stokes_i, abs=1e-12) == 2.5
    assert s2.name == "Src2"


def test_skymodel_parsing_csv_with_alias(params):
    # Custom schema with aliases (avoid tabs/spaces in YAML by building lines explicitly)
    schema_yaml = "\n".join(
        [
            "info: Test schema",
            "parameters:",
            "  ra:",
            "    info: Right ascension",
            "    alias: RA",
            "    units: deg",
            "    ptype: longitude",
            "    required: true",
            "  dec:",
            "    info: Declination",
            "    alias: DEC",
            "    units: deg",
            "    ptype: latitude",
            "    required: true",
            "  stokes_i:",
            "    info: Stokes I",
            "    alias: I",
            "    units: Jy",
            "    ptype: flux",
            "    required: true",
            "  name:",
            "    info: Name",
            "    alias: NAME",
            "    units: null",
            "    ptype: string",
        ]
    )
    schema_path = params.write_temp_file(schema_yaml, suffix=".yaml")

    csv_content = "\n".join(
        [
            "#format: RA,DEC,I,NAME",
            "0,0,1.0,S0",
            "45,-45,0.5,S1",
        ]
    )
    data_path = params.write_temp_file(csv_content, suffix=".csv")

    model = ASCIISkymodel(data_path, delimiter=",", source_schema_file=schema_path)
    assert len(model.sources) == 2

    s0, s1 = model.sources
    assert pytest.approx(s0.ra, abs=1e-12) == 0.0
    assert pytest.approx(s0.dec, abs=1e-12) == 0.0
    assert pytest.approx(s0.stokes_i, abs=1e-12) == 1.0
    assert s0.name == "S0"

    assert pytest.approx(s1.ra, abs=1e-12) == Longitude("45 deg").to("rad").value
    assert pytest.approx(s1.dec, abs=1e-12) == Latitude("-45 deg").to("rad").value
    assert pytest.approx(s1.stokes_i, abs=1e-12) == 0.5
    assert s1.name == "S1"


SEXAGESIMAL_HEADER = "#format: name ra_h ra_m ra_s dec_d dec_m dec_s stokes_i"


def parse_single_source(params, content: str, **kwargs):
    model = ASCIISkymodel(params.write_temp_file(content), **kwargs)
    assert len(model.sources) == 1
    return model.sources[0]


def test_sexagesimal_join_negative_dec(params):
    # Regression: the sign of dec_d must apply to the arcmin/arcsec components
    # too. The naive sum gave -65 + 45' + 9.08'' = -64.2475 deg for PKS0407-65.
    src = parse_single_source(params, f"{SEXAGESIMAL_HEADER}\nPKS0407-65 4 8 20.38 -65 45 9.08 15.0\n")
    assert pytest.approx(math.degrees(src.dec), abs=1e-9) == -(65 + 45 / 60 + 9.08 / 3600)
    # Regression: ra_m/ra_s are minutes/seconds of time (15 arcmin/arcsec each),
    # not arcmin/arcsec.
    assert pytest.approx(math.degrees(src.ra), abs=1e-9) == (4 + 8 / 60 + 20.38 / 3600) * 15


def test_sexagesimal_join_minus_zero_degrees(params):
    # Declinations between 0 and -1 deg have dec_d == "-00"; the sign must
    # survive even though the degrees component parses to 0.0.
    src = parse_single_source(params, f"{SEXAGESIMAL_HEADER}\nJ0000-0045 0 0 0 -00 45 9.08 1.0\n")
    assert pytest.approx(math.degrees(src.dec), abs=1e-9) == -(45 / 60 + 9.08 / 3600)


def test_sexagesimal_join_positive(params):
    src = parse_single_source(params, f"{SEXAGESIMAL_HEADER}\nSrc 12 30 0 30 30 30 1.0\n")
    assert pytest.approx(math.degrees(src.ra), abs=1e-9) == 187.5
    assert pytest.approx(math.degrees(src.dec), abs=1e-9) == 30 + 30 / 60 + 30 / 3600


def test_sexagesimal_join_partial_components(params):
    # Only some components in the header: the missing ones default to 0
    src = parse_single_source(params, "#format: ra_h dec_d stokes_i\n6 -30 1.0\n")
    assert pytest.approx(math.degrees(src.ra), abs=1e-9) == 90.0
    assert pytest.approx(math.degrees(src.dec), abs=1e-9) == -30.0


def test_direct_field_wins_over_join(params):
    # If ra/dec are given directly, the sexagesimal components are ignored
    src = parse_single_source(params, "#format: ra dec dec_d dec_m stokes_i\n10 -10 -65 45 1.0\n")
    assert pytest.approx(math.degrees(src.dec), abs=1e-9) == -10.0


def test_join_out_of_range_dec_rejected(params):
    with pytest.raises(ASCIISourceError, match="dec"):
        parse_single_source(params, f"{SEXAGESIMAL_HEADER}\nBad 0 0 0 -95 0 0 1.0\n")


def test_join_signed_minutes_rejected(params):
    # The sign belongs on the leading component only; signed arcminutes are
    # ambiguous and must be a clear error, not a silently wrong coordinate
    with pytest.raises(ASCIISourceError, match="dec"):
        parse_single_source(params, f"{SEXAGESIMAL_HEADER}\nBad 0 0 0 -65 -45 -9.08 1.0\n")


def test_join_backward_compat_old_component_schema(params):
    # Custom user schemas copied from the old default declare the components
    # with units hour/min/sec and ptype longitude; joins must still be correct
    schema_yaml = "\n".join(
        [
            "info: Old-style schema",
            "parameters:",
            "  name: {info: Name, units: null, ptype: string}",
            "  ra: {info: RA, units: deg, ptype: longitude, join: [ra_h, ra_m, ra_s]}",
            "  ra_h: {info: RA, units: hour, ptype: longitude}",
            "  ra_m: {info: RA, units: min, ptype: longitude}",
            "  ra_s: {info: RA, units: sec, ptype: longitude}",
            "  dec: {info: Dec, units: deg, ptype: latitude, join: [dec_d, dec_m, dec_s]}",
            "  dec_d: {info: Dec, units: deg, ptype: longitude}",
            "  dec_m: {info: Dec, units: arcmin, ptype: longitude}",
            "  dec_s: {info: Dec, units: arcsec, ptype: longitude}",
            "  stokes_i: {info: Stokes I, units: Jy, ptype: flux, required: true}",
        ]
    )
    schema_path = params.write_temp_file(schema_yaml, suffix=".yaml")
    src = parse_single_source(
        params,
        f"{SEXAGESIMAL_HEADER}\nPKS0407-65 4 8 20.38 -65 45 9.08 15.0\n",
        source_schema_file=schema_path,
    )
    assert pytest.approx(math.degrees(src.dec), abs=1e-9) == -(65 + 45 / 60 + 9.08 / 3600)
    assert pytest.approx(math.degrees(src.ra), abs=1e-9) == (4 + 8 / 60 + 20.38 / 3600) * 15


def test_blank_and_comment_lines_skipped_and_lineno_tracked(params):
    content = "\n".join(
        [
            "#format: ra,dec,stokes_i,name",
            "0,0,1.0,S0",
            "",
            "# a comment",
            "45,-45,0.5,S1",
            "",
        ]
    )
    model = ASCIISkymodel(params.write_temp_file(content, suffix=".csv"), delimiter=",")
    assert len(model.sources) == 2
    assert [src.name for src in model.sources] == ["S0", "S1"]
    # lineno maps a parsed source back to its line in the file (header is line 0)
    assert [src.lineno for src in model.sources] == [1, 4]


def test_value_or_default():
    schema = load_default_schema()
    src = ASCIISource(schema)
    src.set_source_param("stokes_i", "2.0")

    # a set field returns its value; unset fields fall back to the schema null value
    assert src.value_or_default("stokes_i") == pytest.approx(2.0)
    assert src.value_or_default("pa") == 0  # angle fields default to 0
    assert src.value_or_default("cont_reffreq") is None  # frequency fields have no default


def test_continuum_coefficients_truncation():
    schema = load_default_schema()

    src = ASCIISource(schema)
    src.set_source_param("cont_coeff_1", "-0.7")
    assert src.continuum_coefficients() == pytest.approx([-0.7])

    # unset coefficient between set ones is zero-filled, trailing ones dropped
    src2 = ASCIISource(schema)
    src2.set_source_param("cont_coeff_1", "-0.7")
    src2.set_source_param("cont_coeff_3", "0.1")
    assert src2.continuum_coefficients() == pytest.approx([-0.7, 0.0, 0.1])

    src3 = ASCIISource(schema)
    assert src3.continuum_coefficients() == []


def gauss_line_profile(freqs, flux, centre, width):
    from simms.constants import FWHM_scale_fact

    sigma = width / FWHM_scale_fact
    return flux * np.exp(-((freqs - centre) ** 2) / (2 * sigma**2))


def test_line_source_from_peak_frequency(params):
    # Regression for issue #152: is_line never activated, so every line source
    # silently fell through to the (flat) continuum branch.
    src = parse_single_source(params, "#format: ra dec stokes_i line_peak line_width\n0 -30 1.5 1.42e9 1e7\n")
    assert src.is_line is True

    freqs = np.linspace(1.40e9, 1.44e9, 81)
    bmatrix = src.get_brightness_matrix(freqs, ncorr=2)
    # ncorr=2, unpolarised: XX = I
    assert bmatrix[0] == pytest.approx(gauss_line_profile(freqs, 1.5, 1.42e9, 1e7))


def test_line_source_from_restfreq_and_redshift(params):
    # observed centre = line_restfreq / (1 + line_redshift)
    restfreq, redshift = 1.420405751e9, 0.1
    src = parse_single_source(
        params,
        f"#format: ra dec stokes_i line_restfreq line_redshift line_width\n0 -30 1.0 {restfreq} {redshift} 1e6\n",
    )
    assert src.is_line is True

    centre = restfreq / (1 + redshift)
    freqs = np.linspace(centre - 5e6, centre + 5e6, 101)
    bmatrix = src.get_brightness_matrix(freqs, ncorr=2)
    assert bmatrix[0] == pytest.approx(gauss_line_profile(freqs, 1.0, centre, 1e6))


def test_line_source_restfreq_without_redshift(params):
    # redshift defaults to 0: the observed centre is the rest frequency
    src = parse_single_source(params, "#format: ra dec stokes_i line_restfreq line_width\n0 -30 1.0 1.42e9 1e6\n")
    assert src.is_line is True

    freqs = np.linspace(1.415e9, 1.425e9, 101)
    bmatrix = src.get_brightness_matrix(freqs, ncorr=2)
    assert bmatrix[0] == pytest.approx(gauss_line_profile(freqs, 1.0, 1.42e9, 1e6))


def test_line_source_peak_wins_over_restfreq(params):
    # line_peak takes precedence when both parametrisations are given
    src = parse_single_source(
        params,
        "#format: ra dec stokes_i line_peak line_restfreq line_redshift line_width\n0 -30 1.0 1.30e9 1.42e9 0.1 1e6\n",
    )
    freqs = np.linspace(1.295e9, 1.305e9, 101)
    bmatrix = src.get_brightness_matrix(freqs, ncorr=2)
    assert bmatrix[0] == pytest.approx(gauss_line_profile(freqs, 1.0, 1.30e9, 1e6))


def test_unknown_header_field_error(params):
    # an unknown column must be a clear error naming the field, not a raw
    # KeyError from the alias lookup
    with pytest.raises(ASCIISkymodelError, match="line_ref_freq"):
        parse_single_source(params, "#format: ra dec stokes_i line_ref_freq\n0 -30 1.0 1.42e9\n")


def test_line_source_partial_fields_error(params):
    # some but not all line fields must be an error, not silently a continuum source
    with pytest.raises(ASCIISourceError, match="Spectral Line Source"):
        parse_single_source(params, "#format: ra dec stokes_i line_peak\n0 -30 1.0 1.42e9\n")
    with pytest.raises(ASCIISourceError, match="Spectral Line Source"):
        parse_single_source(params, "#format: ra dec stokes_i line_width\n0 -30 1.0 1e6\n")


def test_line_source_nonpositive_width_error(params):
    # gauss_1d divides by the width: zero or negative must fail at parse time,
    # not as NaNs in the predicted visibilities
    with pytest.raises(ASCIISourceError, match="line_width"):
        parse_single_source(params, "#format: ra dec stokes_i line_peak line_width\n0 -30 1.0 1.42e9 0\n")
    with pytest.raises(ASCIISourceError, match="line_width"):
        parse_single_source(params, "#format: ra dec stokes_i line_peak line_width\n0 -30 1.0 1.42e9 -1e6\n")


def test_line_source_unphysical_redshift_error(params):
    # 1 + redshift <= 0 gives a zero/negative observed line centre
    with pytest.raises(ASCIISourceError, match="line_redshift"):
        parse_single_source(
            params, "#format: ra dec stokes_i line_restfreq line_redshift line_width\n0 -30 1.0 1.42e9 -1 1e6\n"
        )
    # blueshifts (-1 < z < 0) remain valid
    src = parse_single_source(
        params, "#format: ra dec stokes_i line_restfreq line_redshift line_width\n0 -30 1.0 1.42e9 -0.5 1e6\n"
    )
    assert src.is_line is True


def test_transient_brightness_matrix_time_axis():
    schema = load_default_schema()
    src = ASCIISource(schema)
    for k, v in {
        "ra": 0,
        "dec": 0,
        "stokes_i": 1.0,
        "transient_start": "100 s",
        "transient_period": "500 s",
        "transient_ingress": "50 s",
        "transient_absorb": 0.5,
    }.items():
        src.set_source_param(k, v)
    src.finalise()
    assert src.is_transient is True

    freqs = [1.0e9, 1.1e9, 1.2e9]
    unique_times = 1e9 + 60.0 * np.arange(20)
    bmatrix = src.get_brightness_matrix(freqs, ncorr=4, unique_times=unique_times)
    # transient sources get a time axis: (ncorr, ntime, nchan)
    assert bmatrix.shape == (4, 20, 3)
    # flux dips by transient_absorb mid-transit relative to out-of-transit
    out_of_transit = bmatrix[0, 0, 0].real
    mid_transit = bmatrix[0, :, 0].real.min()
    assert pytest.approx(mid_transit / out_of_transit, abs=0.05) == 0.5

    # a mapper expands unique times onto arbitrary output rows
    mapper = [0, 0, 5, 19]
    bmatrix_mapped = src.get_brightness_matrix(freqs, ncorr=4, unique_times=unique_times, time_index_mapper=mapper)
    assert bmatrix_mapped.shape == (4, 4, 3)


MIXED_HEADER = (
    "#format: name ra dec stokes_i line_peak line_width "
    "transient_start transient_period transient_ingress transient_absorb"
)


def test_mixed_source_types_in_one_catalogue(params):
    # A catalogue is one table, so a file holding a line source and a transient source
    # must carry both sets of columns; the rows that are not of a type null theirs out.
    # Before this was handled, a null column still typed the source, and the line source
    # checks tripped over the resulting None line_width.
    model = ASCIISkymodel(
        params.write_temp_file(
            "\n".join(
                [
                    MIXED_HEADER,
                    "LINE 0.0 -30.0 1.0 1.42e9 1e7 null null null null",
                    "TRANS 0.1 -30.1 2.0 null null 100 500 50 0.5",
                    "BOTH 0.2 -30.2 3.0 1.3e9 5e6 200 400 40 0.2",
                    "PLAIN 0.3 -30.3 4.0 null null null null null null",
                    "",
                ]
            )
        )
    )
    types = {src.name: (src.is_line, src.is_transient) for src in model.sources}
    assert types == {
        "LINE": (True, False),
        "TRANS": (False, True),
        "BOTH": (True, True),
        "PLAIN": (False, False),
    }
    assert model.has_transient is True

    # a nulled field is not merely defaulted, it is absent
    assert getattr(model.sources[1], "line_peak", None) is None

    # and the model still predicts: only the transient sources pick up a time axis
    freqs = np.array([1.30e9, 1.42e9])
    unique_times = 1e9 + 60.0 * np.arange(10)
    shapes = [src.get_brightness_matrix(freqs, ncorr=2, unique_times=unique_times).shape for src in model.sources]
    assert shapes == [(2, 2), (2, 10, 2), (2, 10, 2), (2, 2)]


def test_empty_field_in_delimited_catalogue_is_unset(params):
    # A comma-delimited catalogue marks a column unset by leaving it empty
    model = ASCIISkymodel(
        params.write_temp_file(
            "#format: name,ra,dec,stokes_i,line_peak,line_width\nSrc,0.0,-30.0,1.0,,\n",
        ),
        delimiter=",",
    )
    assert model.sources[0].is_line is False


def test_transient_start_accepts_a_bare_number(params):
    # transient_start defaults to seconds like the other transient fields; a bare
    # number used to raise a dimensionless-to-seconds UnitConversionError.
    src = parse_single_source(
        params,
        "#format: ra dec stokes_i transient_start transient_period transient_ingress transient_absorb\n"
        "0 -30 1.0 100 500 50 0.5\n",
    )
    assert src.is_transient is True
    assert src.transient_start == 100.0


def test_parse_error_names_the_offending_row(params):
    with pytest.raises(ASCIISourceError, match="line 3"):
        ASCIISkymodel(
            params.write_temp_file(
                "#format: ra dec stokes_i line_peak line_width\n0 -30 1.0 1.42e9 1e7\n0 -31 1.0 1.42e9 0\n"
            )
        )
