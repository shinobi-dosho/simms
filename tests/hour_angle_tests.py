"""Recover track angles from UVW, independently of the simulator's time grid."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.utils import iers

from simms.telescope.array_utilities import Array


@pytest.fixture(autouse=True)
def offline_iers():
    with iers.conf.set_temp("auto_download", False):
        yield


def recover_local_ha(array, result):
    i, j = np.triu_indices(array.nant, 1)
    baselines = array.antlocations[j] - array.antlocations[i]
    u_samples = result.uvw[:, 0].reshape(-1, len(i))
    # u = sin(GHA) bx + cos(GHA) by for absolute ITRF baselines.
    coefficients = np.linalg.lstsq(baselines[:, :2], u_samples.T, rcond=None)[0]
    return Angle(np.arctan2(coefficients[0], coefficients[1]) + array.lon0, u.rad).wrap_at(180 * u.deg)


def coverage(array, start_ha=None, ntimes=450):
    return array.uvgen(
        ["J2000", "6h0m0s", "-15deg"],
        32,
        ntimes,
        "1GHz",
        "16MHz",
        4,
        start_time="2024-03-14T12:00:00",
        start_ha=start_ha,
    )


@pytest.mark.parametrize("layout", ["meerkat", "vla-b"])
@pytest.mark.parametrize("start_ha", [-2, 0, 2])
def test_local_start_hours_and_timestamp_consistency(layout, start_ha):
    array = Array(layout)
    result = coverage(array, start_ha)
    local_ha = recover_local_ha(array, result)
    assert local_ha[0].to_value(u.hourangle) == pytest.approx(start_ha, abs=1e-8)
    times = Time(result.time_entries / 86400, format="mjd", scale="utc")
    expected = (times.sidereal_time("mean", longitude=array.lon0 * u.rad) - 6 * u.hourangle).wrap_at(180 * u.deg)
    np.testing.assert_allclose(local_ha.rad, expected.rad, atol=1e-9)
    np.testing.assert_allclose(np.diff(result.time_entries), 32, atol=2e-6, rtol=0)
    assert result.time_entries[-1] - result.time_entries[0] == pytest.approx(449 * 32, abs=2e-6)
    assert (times[0] - Time("2024-03-14T12:00:00")).to_value(u.hour) == pytest.approx(0, abs=12)
    dec = np.deg2rad(-15)
    expected_elevation = np.rad2deg(
        np.arcsin(np.sin(array.lat0) * np.sin(dec) + np.cos(array.lat0) * np.cos(dec) * np.cos(local_ha.rad))
    )
    np.testing.assert_allclose(result.source_elevations, expected_elevation, atol=1e-9)
    if start_ha == -2:
        assert local_ha[-1].to_value(u.hourangle) == pytest.approx(2, abs=0.02)


@pytest.mark.parametrize("ntimes", [1, 450])
def test_timestamp_mode_starts_at_requested_time(ntimes):
    array = Array("meerkat")
    result = coverage(array, ntimes=ntimes)
    start = Time("2024-03-14T12:00:00", scale="utc")
    assert result.time_entries[0] == pytest.approx(start.mjd * 86400, abs=1e-6, rel=0)
    expected = (start.sidereal_time("mean", longitude=array.lon0 * u.rad) - 6 * u.hourangle).wrap_at(180 * u.deg)
    assert recover_local_ha(array, result)[0].rad == pytest.approx(expected.rad, abs=1e-9)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_hour_angle_rejected(invalid):
    with pytest.raises(ValueError, match="finite local hour angle"):
        coverage(Array("meerkat"), invalid)


def test_hour_angle_cli_writes_consistent_ms():
    from subprocess import run

    from casacore.tables import table

    from tests import InitTest, simms_executable

    files = InitTest()
    ms = files.random_named_directory(suffix=".ms")
    run(
        [
            simms_executable(),
            "telsim",
            "--telescope",
            "kat-7",
            "--direction",
            "J2000,6h0m0s,-15deg",
            "--starttime",
            "2024-03-14T12:00:00",
            "--startha=-2",
            "--dtime",
            "32",
            "--ntime",
            "3",
            "--nchan",
            "1",
            ms,
        ],
        check=True,
        capture_output=True,
    )
    with table(ms, ack=False) as main:
        times = np.unique(main.getcol("TIME"))
        uvw = main.getcol("UVW")
    from simms.utilities import ObjDict

    array = Array("kat-7")
    ha = recover_local_ha(array, ObjDict({"uvw": uvw}))
    assert ha[0].hour == pytest.approx(-2, abs=1e-8)
    expected = (
        Time(times / 86400, format="mjd", scale="utc").sidereal_time("mean", longitude=array.lon0 * u.rad)
        - 6 * u.hourangle
    ).wrap_at(180 * u.deg)
    np.testing.assert_allclose(ha.rad, expected.rad, atol=1e-9)
    np.testing.assert_allclose(np.diff(times), 32, atol=2e-6, rtol=0)
