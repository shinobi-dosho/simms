"""Session-wide wiring for the test helpers.

Keeping `InitTest`'s temp MSs and files inside pytest's own basetemp (rather than next to
the test modules) means a run that dies without unwinding -- SIGKILL, an OOM-kill, a crash
in casacore or dask -- cannot dirty the working tree. `InitTest.__del__` still reclaims each
fixture's space promptly during a run; pytest purges whatever a killed run left behind.
"""

import pytest

from . import set_basedir


@pytest.fixture(scope="session", autouse=True)
def _temp_basedir(tmp_path_factory):
    set_basedir(tmp_path_factory.getbasetemp())
