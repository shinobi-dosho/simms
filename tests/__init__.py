import os.path
import shutil
import sys
import tempfile
from types import SimpleNamespace

TESTDIR = os.path.abspath(os.path.dirname(__file__))


def skysim_opts(ms, ascii_sky=None, column="DATA", **overrides):
    """Baseline ``skysim.runit`` opts for tests; override any field via keyword.

    Keeps the full option set in one place so it does not drift across the test modules
    that drive skysim (each was carrying its own copy of this dict).
    """
    opts = {
        "ms": ms,
        "ascii_sky": ascii_sky,
        "fits_sky": None,
        "wsclean_sky": None,
        "nworkers": 1,
        "row_chunks": 100000,
        "field_id": 0,
        "spw_id": 0,
        "sefd": None,
        "polarisation": False,
        "pol_basis": "linear",
        "chan_chunks": None,
        "source_schema": None,
        "ascii_species": None,
        "ascii_delimiter": None,
        "primary_beam": None,
        "beam_band": "L",
        "beam_pa_step": 1.0,
        "beam_grid_max_gib": 4.0,
        "beam_jones": "diagonal",
        "fits_beam_mode": "aterm",
        "aterm_freq_tol": 1e-3,
        "beam_l_axis": "-X",
        "beam_m_axis": "Y",
        "telescope_name_column": "TELESCOPE_NAME",
        "input_column": None,
        "mode": "sim",
        "column": column,
        "random_seed": None,
        "corruptions": None,
        "log_level": "CRITICAL",
        "pixel_tol": 1e-7,
        "predict_backend": "auto",
        "fits_spectrum": "flat",
        "fits_spi": None,
        "fits_ref_freq": None,
        "fits_spectrum_order": 2,
        "fits_sky_interp": "linear",
        "fft_precision": "double",
        "do_wstacking": True,
    }
    opts.update(overrides)
    return SimpleNamespace(**opts)


def simms_executable() -> str:
    """
    Absolute path to the installed ``simms`` console script.

    Running the suite as ``python -m pytest`` does not put the interpreter's
    script directory on PATH, so look there first and only then fall back to
    PATH.
    """
    candidate = os.path.join(os.path.dirname(sys.executable), "simms")
    if os.path.exists(candidate):
        return candidate

    found = shutil.which("simms")
    if found:
        return found

    raise RuntimeError(
        f"The 'simms' console script was not found next to {sys.executable} or on PATH. Install the package first."
    )


class InitTest:
    def __init__(self):
        self.test_files = []

    def random_named_file(self, suffix: str | None = None):
        if not hasattr(self, "test_files"):
            self.test_files = []

        # Only a uniquely-named path is wanted, not an open handle: the caller writes the
        # file itself (or hands the name to code that creates a directory there). The handle
        # is closed on the next line, so SIM115's context manager would not help.
        file_obj = tempfile.NamedTemporaryFile(suffix=suffix, dir=TESTDIR, delete=False)  # noqa: SIM115
        name = file_obj.name
        file_obj.close()

        self.test_files.append(name)
        return name

    def random_named_directory(self, suffix: str | None = None):
        if not hasattr(self, "test_files"):
            self.test_files = []

        name = tempfile.mkdtemp(suffix=suffix, dir=TESTDIR)

        self.test_files.append(name)
        return name

    def __del__(self):
        for path in getattr(self, "test_files", []):
            if os.path.exists(path):
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError as e:
                        print(f"Error deleting file '{path}': {e}")
                elif os.path.isdir(path):
                    try:
                        shutil.rmtree(path)
                    except OSError as e:
                        print(f"Error deleting directory '{path}': {e}")
