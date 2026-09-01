"""Tests for the test helpers themselves (`tests.InitTest`).

The invariant worth guarding is where the temp fixtures live: under pytest's basetemp, never
inside the repository. Cleanup is `InitTest.__del__`, which runs on a normal or interrupted
exit but not on a SIGKILL, an OOM-kill or a crash in casacore -- so if these paths pointed at
`tests/`, a hard-killed run would leave MSs and cubes in the working tree.
"""

import os

from . import TESTDIR, InitTest


def _under(path, parent):
    return os.path.commonpath([os.path.realpath(path), os.path.realpath(parent)]) == os.path.realpath(parent)


def test_temp_paths_live_in_the_pytest_basetemp(tmp_path_factory):
    it = InitTest()
    basetemp = str(tmp_path_factory.getbasetemp())
    for path in (it.random_named_file(suffix=".fits"), it.random_named_directory(suffix=".ms")):
        assert _under(path, basetemp), f"{path} is outside pytest's basetemp"
        assert not _under(path, TESTDIR), f"{path} is inside the repository"


def test_register_tracks_paths_the_helpers_did_not_hand_out(tmp_path_factory):
    # write_beam_fits_cattery expands one prefix into eight files; only the prefix comes from
    # random_named_file, so the derived names have to be registered explicitly or they leak.
    it = InitTest()
    prefix = it.random_named_file(suffix="")
    derived = [f"{prefix}_{corr}_re.fits" for corr in ("xx", "yy")]
    for path in derived:
        with open(path, "w") as fh:
            fh.write("x")

    assert it.register(*derived) == tuple(derived)
    assert it.register(derived[0]) == derived[0]  # single path comes back unwrapped

    it.__del__()  # what the interpreter does when the fixture goes out of scope
    assert not any(os.path.exists(p) for p in derived)
