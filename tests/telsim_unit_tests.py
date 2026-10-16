import os
import shutil

import click
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from daskms import xds_from_table
from scipy.optimize import least_squares

from simms.apps.schema import build_options
from simms.apps.telsim import _antenna_selection, inputs_model
from simms.telescope import array_utilities, layouts
from simms.telescope.generate_ms import create_ms
from simms.telescope.layouts import custom_telescopes

from . import InitTest


class InitThisTest(InitTest):
    def __init__(self):
        """Set up common test parameters before each test method runs."""

        # important to have this first, else files/dirs created before it will not be tracked
        self.test_files = []

        self.ms = self.random_named_directory(suffix=".ms")
        self.telescope = "kat-7"
        self.max_bl = 185
        self.min_bl = 26
        self.nant = 7
        self.ntimes = 10
        self.dtime = 1
        self.start_freq = "1420MHz"
        self.dfreq = "1MHz"
        self.nchan = 2
        self.direction = ["J2000", "0h0m20s", "-30deg"]
        self.start_time = "2025-03-06T12:25:00"
        self.sefd = 821
        self.column = "DATA"

        # Store temporary files to be cleaned up
        self.make_ms()

    def make_ms(self):
        create_ms(
            self.ms,
            telescope_name=self.telescope,
            pointing_direction=self.direction,
            dtime=self.dtime,
            ntimes=self.ntimes,
            start_freq=self.start_freq,
            dfreq=self.dfreq,
            nchan=self.nchan,
            correlations=["XX", "YY"],
            row_chunks=10000,
            sefd=self.sefd,
            column=self.column,
            start_time=self.start_time,
            smooth=None,
            fit_order=None,
        )

        self.test_files.append(self.ms)


@pytest.fixture
def params():
    return InitThisTest()


def test_max_bl(params):
    ds = xds_from_table(params.ms)[0]
    uvw = ds.UVW.values
    bl = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2)

    assert np.isclose(bl.max(), params.max_bl, atol=10)
    assert np.isclose(bl.min(), params.min_bl, atol=4)


def test_uv_coverage(params):
    """
    Test to check if the uv coverage shape is what is expected.
    e.g circles at the poles and lines at the equator.

    """

    array = array_utilities.Array(params.telescope)
    array_data_line = array.uvgen(
        pointing_direction=["J2000", "0deg", "0deg"],
        dtime=2,
        ntimes=500,
        start_freq=params.start_freq,
        dfreq="2MHz",
        nchan=10,
    )

    line = check_circle_or_ellipse(array_data_line.uvw[::21, 0], array_data_line.uvw[::21, 1])
    assert line["is_line"]

    array_data_circle = array.uvgen(
        pointing_direction=["J2000", "0deg", "-90deg"],
        dtime=2,
        ntimes=5000,
        start_freq=params.start_freq,
        dfreq="2MHz",
        nchan=10,
    )

    circle = check_circle_or_ellipse(array_data_circle.uvw[::21, 0], array_data_circle.uvw[::21, 1])
    assert circle["is_circle"]

    array_data_circle_n = array.uvgen(
        pointing_direction=["J2000", "0deg", "90deg"],
        dtime=2,
        ntimes=5000,
        start_freq="1420MHz",
        dfreq="2MHz",
        nchan=10,
    )

    circle_n = check_circle_or_ellipse(array_data_circle_n.uvw[::21, 0], array_data_circle_n.uvw[::21, 1])
    assert circle_n["is_circle"]

    array_data_ellipse = array.uvgen(
        pointing_direction=["J2000", "0deg", "-30deg"],
        dtime=2,
        ntimes=5000,
        start_freq=params.start_freq,
        dfreq="2MHz",
        nchan=10,
    )

    ellipse = check_circle_or_ellipse(array_data_ellipse.uvw[::21, 0], array_data_ellipse.uvw[::21, 1])
    assert not ellipse["is_circle"] and ellipse["is_ellipse"]


def test_visdata_configuration_info(params):
    ds = xds_from_table(params.ms)[0]
    ds_spw = xds_from_table(f"{params.ms}::SPECTRAL_WINDOW")[0]
    num_chan = ds_spw.NUM_CHAN.values
    assert np.isclose(num_chan, 2)

    freq = ds_spw.CHAN_FREQ.values[0][0]
    assert np.isclose(freq, 1420e6)

    dfreq = ds_spw.CHAN_FREQ.values[0][1] - freq
    assert np.isclose(dfreq, 1e6)

    # Frequencies are topocentric: MEAS_FREQ_REF must be 5 (TOPO), not the casacore
    # default of 0 (REST/undefined) which breaks CASA imaging.
    assert int(ds_spw.MEAS_FREQ_REF.values[0]) == 5

    time = np.unique(ds.TIME).shape
    assert np.isclose(time, 10)

    dtime = ds.INTERVAL.values[0]
    assert np.isclose(dtime, 1)

    nbl = ds.DATA.shape[0] / time[0]
    tel_nbl = params.nant * (params.nant - 1) // 2
    assert np.isclose(nbl, tel_nbl)

    ds_pol = xds_from_table(f"{params.ms}::POLARIZATION")[0]
    corr = ds_pol.CORR_TYPE.values[0]
    assert np.isclose(corr[0], 9)
    assert np.isclose(corr[1], 12)

    ds_ant = xds_from_table(f"{params.ms}::ANTENNA")[0]
    mount = ds_ant.MOUNT.values[0]
    size = ds_ant.DISH_DIAMETER.values[0]
    assert mount == "ALT-AZ"
    assert np.isclose(size, 12)


def test_pointing_table_is_per_antenna_per_time(params):
    # POINTING is keyed by (ANTENNA_ID, TIME): nant x ntime rows, not one row per
    # main-table row. Getting this wrong repeats each pointing nbl/nant times and leaves
    # every antenna but the first without a pointing record, so the beam centre that
    # skysim reads from POINTING.DIRECTION is only defined for antenna 0.
    ds = xds_from_table(params.ms)[0]
    ds_point = xds_from_table(f"{params.ms}::POINTING")[0]

    main_times = np.unique(ds.TIME.values)
    ntime = main_times.shape[0]
    assert ds_point.sizes["row"] == params.nant * ntime

    ant_ids = np.asarray(ds_point.ANTENNA_ID.values)
    assert set(ant_ids.tolist()) == set(range(params.nant))

    # Exactly one record per (antenna, timestamp) pair, over the main table's timestamps.
    pnt_times = np.asarray(ds_point.TIME.values)
    assert np.array_equal(np.unique(pnt_times), main_times)
    keys = set(zip(ant_ids.tolist(), pnt_times.tolist(), strict=True))
    assert len(keys) == ds_point.sizes["row"]

    # Every antenna points at the field centre, and TARGET (required by MSv2) agrees.
    orig_direction = SkyCoord(*params.direction[1:])
    ra0 = orig_direction.ra.to("rad").value
    dec0 = orig_direction.dec.to("rad").value
    directions = np.asarray(ds_point.DIRECTION.values)[:, 0, :]
    assert np.allclose(directions[:, 0], ra0)
    assert np.allclose(directions[:, 1], dec0)
    assert np.allclose(np.asarray(ds_point.TARGET.values), np.asarray(ds_point.DIRECTION.values))


def test_subarray_options_accept_comma_separated_and_repeated_forms():
    # Both antenna-selection options document a comma-separated syntax, and the CLI renders
    # them as click `multiple=True` options, so the repeated form arrives as a tuple. Accept
    # both, and flatten to the plain list the layout code expects.
    assert _antenna_selection(("M000,M005,M010",)) == ["M000", "M005", "M010"]
    assert _antenna_selection(("M000", "M005")) == ["M000", "M005"]
    assert _antenna_selection(("0,9",), cast=int) == [0, 9]
    assert _antenna_selection(("0", "9"), cast=int) == [0, 9]
    assert _antenna_selection(("0, 9, 2",), cast=int) == [0, 9, 2]
    # A recipe/cab passing a real YAML list of ints reaches the same place.
    assert _antenna_selection([0, 9], cast=int) == [0, 9]
    # Unset (click's empty tuple, or None) stays None so the layout falls back to all antennas.
    assert _antenna_selection(()) is None
    assert _antenna_selection(None) is None


def test_subarray_range_accepts_comma_separated_indices():
    # Guards the string arm of subarray_range: narrowing this to `list[int]` renders an
    # INTEGER option and click rejects "0,9" before the value reaches _antenna_selection.
    option = next(opt for opt in build_options(inputs_model) if opt.name == "subarray_range")
    assert option.multiple
    assert option.type is click.STRING
    assert option.type.convert("0,9", None, None) == "0,9"


@pytest.mark.parametrize(
    "layout,kwargs,expect",
    [
        # A named subarray has no layout file of its own, so subarray selection on top of one
        # used to fail with a missing <name>.geodetic.yaml.
        ("meerkat", {"subarray_range": [0, 9]}, 10),
        ("meerkat", {"subarray_list": ["M000", "M005"]}, 2),
        ("skamid-aa1", {"subarray_range": [0, 3]}, 4),
        # A plain built-in layout keeps working.
        ("skamid", {"subarray_range": [0, 9]}, 10),
    ],
)
def test_subarray_selection_on_named_subarrays(layout, kwargs, expect):
    array = array_utilities.Array(layout=layout, **kwargs)
    assert len(array.layout["antnames"]) == expect


def test_layout_accepts_a_user_supplied_file(params):
    # Layout resolution used to reach for a scabha `File.EXISTS` attribute that a plain str
    # does not have, so passing a layout by path raised AttributeError before reading anything.
    path = params.random_named_file(suffix=".yaml")
    shutil.copy(os.path.join(os.path.dirname(layouts.__file__), "kat-7.geodetic.yaml"), path)

    array = array_utilities.Array(layout=path)
    assert len(array.layout["antnames"]) == params.nant

    array = array_utilities.Array(layout=path, subarray_range=[0, 2])
    assert len(array.layout["antnames"]) == 3


def test_unknown_layout_names_the_known_telescopes():
    with pytest.raises(FileNotFoundError, match="neither a known telescope nor an existing layout file"):
        array_utilities.Array(layout="/no/such/layout.yaml")


def test_subarray_list_rejects_unknown_antenna_by_name():
    # A named subarray belongs to --telescope; passing it to -sublist used to raise a bare
    # KeyError naming only the offending string.
    with pytest.raises(ValueError, match="Unknown antenna"):
        custom_telescopes(layout="skamid", subarray_list=["meerkat"])
    with pytest.raises(ValueError, match="M999"):
        custom_telescopes(layout="skamid", subarray_list=["M000", "M999"])


def test_subarray_range_rejects_out_of_range_indices():
    with pytest.raises(ValueError, match="outside layout"):
        custom_telescopes(layout="skamid", subarray_range=[0, 9999])
    with pytest.raises(ValueError, match="start,end"):
        custom_telescopes(layout="skamid", subarray_range=[5])


def test_antenna_telescope_name(params):
    # The ANTENNA table carries a per-antenna TELESCOPE_NAME column (used by skysim to
    # select a primary beam). For a homogeneous layout it defaults to the array name.
    ds_ant = xds_from_table(f"{params.ms}::ANTENNA")[0]
    assert hasattr(ds_ant, "TELESCOPE_NAME")
    tnames = list(np.asarray(ds_ant.TELESCOPE_NAME.values))
    assert len(tnames) == params.nant
    assert set(tnames) == {"kat-7"}


def test_heterogeneous_telescope_name(params):
    # A mixed subarray (spanning the skamid 13.5 m / 15 m boundary) writes per-antenna
    # TELESCOPE_NAME labels aligned with the dish sizes, so skysim can pick a beam per type.
    ms = params.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=params.direction,
        dtime=params.dtime,
        ntimes=2,
        start_freq=params.start_freq,
        dfreq=params.dfreq,
        nchan=params.nchan,
        correlations=["XX", "YY"],
        row_chunks=10000,
        sefd=params.sefd,
        column=params.column,
        start_time=params.start_time,
        smooth=None,
        fit_order=None,
        subarray_range=[60, 68],  # M060..M063 (MeerKAT) + SKA001..SKA004 (MeerKAT+)
    )
    ds_ant = xds_from_table(f"{ms}::ANTENNA")[0]
    tnames = np.asarray(ds_ant.TELESCOPE_NAME.values).astype(str)
    sizes = np.asarray(ds_ant.DISH_DIAMETER.values)
    assert set(tnames) == {"MKAT-MA", "MKAT-EA"}
    assert np.all(sizes[tnames == "MKAT-MA"] == 13.5)
    assert np.all(sizes[tnames == "MKAT-EA"] == 15.0)


def check_circle_or_ellipse(u, v):
    """
    Check if UV points form a circle, ellipse or line.
    """

    center_u = np.mean(u)
    center_v = np.mean(v)

    distances = np.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
    mean_distance = np.mean(distances)

    def conic_equation(params, x, y):
        a, b, c, d, e, f = params
        return a * x**2 + b * x * y + c * y**2 + d * x + e * y + f

    def residuals(params, x, y):
        return conic_equation(params, x, y)

    initial_guess = [1, 0, 1, 0, 0, -(mean_distance**2)]
    result = least_squares(residuals, initial_guess, args=(u, v))
    a, b, c, _d, _e, _f = result.x  # conic coefficients; only the quadratic terms are tested

    discriminant = b**2 - 4 * a * c
    is_circle = abs(a - c) < 0.01 * abs(a) and abs(b) < 0.01 * abs(a)
    is_ellipse = discriminant < 0 and not is_circle

    coeffs = np.polyfit(u, v, 1)
    m, b = coeffs
    v_pred = m * u + b
    residual_v = v - v_pred
    residual_std = np.std(residual_v)
    ss_tot = np.sum((v - np.mean(v)) ** 2)
    ss_res = np.sum(residual_v**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    u_range = np.max(u) - np.min(u) if len(u) > 1 else 1.0
    v_std = np.std(v)
    is_horizontal_line = abs(m) < 0.01 and v_std < 0.05 * u_range
    is_line = (r_squared > 0.95 and residual_std < 0.05 * np.std(np.sqrt(u**2 + v**2))) or is_horizontal_line

    return {
        "is_circle": is_circle,
        "is_ellipse": is_ellipse,
        "is_line": is_line,
    }
