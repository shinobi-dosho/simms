"""``skysim --mode`` combines the simulation with an existing column.

The regression these guard: ``--mode add`` without ``--input-column`` used to fall through
to the plain-overwrite branch, so the mode silently did nothing and the column was replaced.
"""

import numpy as np
import pytest
from daskms import xds_from_ms

from simms.apps import skysim
from simms.telescope.generate_ms import create_ms

from . import InitTest, skysim_opts


class _ModeTest(InitTest):
    def __init__(self):
        self.test_files = []
        self.ms = self.random_named_directory(suffix=".ms")
        create_ms(
            self.ms,
            telescope_name="meerkat",
            pointing_direction=["J2000", "0deg", "-30deg"],
            dtime=600,
            ntimes=3,
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
            subarray_range=[0, 5],
        )
        self.sky = self.random_named_file(suffix=".txt")
        with open(self.sky, "w") as fh:
            fh.write("#format: name ra dec stokes_i\nS 0h0m20s -30d10m0s 3.0\n")


@pytest.fixture
def mt():
    return _ModeTest()


def column(ms, name):
    return getattr(xds_from_ms(ms)[0], name).data.compute()


def test_add_without_input_column_defaults_to_the_output_column(mt):
    skysim.runit(skysim_opts(mt.ms, ascii_sky=mt.sky, column="DATA"))
    once = column(mt.ms, "DATA")
    assert np.abs(once).max() > 0

    skysim.runit(skysim_opts(mt.ms, ascii_sky=mt.sky, column="DATA", mode="add"))
    assert np.allclose(column(mt.ms, "DATA"), 2 * once)


def test_subtract_without_input_column_empties_the_output_column(mt):
    skysim.runit(skysim_opts(mt.ms, ascii_sky=mt.sky, column="DATA"))
    skysim.runit(skysim_opts(mt.ms, ascii_sky=mt.sky, column="DATA", mode="subtract"))
    assert np.allclose(column(mt.ms, "DATA"), 0)


def test_add_reads_an_explicit_input_column(mt):
    skysim.runit(skysim_opts(mt.ms, ascii_sky=mt.sky, column="BASE"))
    base = column(mt.ms, "BASE")

    skysim.runit(skysim_opts(mt.ms, ascii_sky=mt.sky, column="SUM", mode="add", input_column="BASE"))
    assert np.allclose(column(mt.ms, "SUM"), 2 * base)
    # the input column is left alone
    assert np.allclose(column(mt.ms, "BASE"), base)


def test_add_from_a_missing_column_is_an_error(mt):
    with pytest.raises(RuntimeError, match="NOPE"):
        skysim.runit(skysim_opts(mt.ms, ascii_sky=mt.sky, column="DATA", mode="add", input_column="NOPE"))
