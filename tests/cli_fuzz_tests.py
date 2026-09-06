"""Robustness tests for the CLI: hostile and malformed input on every subcommand.

``cli_contract_tests`` checks the *shape* of the generated option surface -- that every
option is read, that a help string does not promise a syntax the click type rejects. This
module checks what happens when the values are wrong, which is the state a CLI actually
spends most of its life in.

Three properties are asserted, each of which was violated by a shipped release:

* **Nothing reaches the user as a traceback.** Every failure here is a bad *input*, so the
  Python stack that produced it says nothing the user can act on. ``SimmsGroup.invoke``
  folds them into ``Error: ...``; ``--log-level DEBUG`` still gets the traceback.
* **An option is never silently dropped.** The root group is ``chain=True``, which makes
  click stop parsing at the first positional and hand the rest to the next command in the
  chain -- so ``simms telsim obs.ms --telescope meerkat`` used to fail with
  ``No such option '-n'``, every option after the MS quietly reassigned.
* **A value that cannot describe a valid MS is rejected before anything is written.**
  ``--dtime -8`` wrote a negative EXPOSURE and exited 0, and a later ``skysim --sefd``
  turned that into an all-NaN DATA column, also exiting 0.

The cases are driven through ``CliRunner`` rather than a subprocess: it is the same parsing
and dispatch path, and it keeps a ~90-case sweep to a few seconds. Because the commands are
built once at import and shared, that also makes this the place where shared-state
mutation between invocations shows up.
"""

import logging
import os

import click
import pytest
from click.testing import CliRunner

from simms import set_logger
from simms.apps.main import cli
from simms.exceptions import InvalidInputError, SimmsError
from tests import InitTest

init = InitTest()


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


@pytest.fixture(scope="module")
def missing(tmp_path_factory):
    """A path that does not exist and is never created."""
    return str(tmp_path_factory.mktemp("fuzz") / "does-not-exist")


@pytest.fixture(scope="module")
def empty_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("fuzz") / "empty.txt"
    path.write_text("")
    return str(path)


@pytest.fixture(scope="module")
def binary_junk(tmp_path_factory):
    """A file that is neither text nor any format simms reads."""
    path = tmp_path_factory.mktemp("fuzz") / "junk.bin"
    path.write_bytes(bytes(range(256)) * 4)
    return str(path)


@pytest.fixture(scope="module")
def small_ms(tmp_path_factory):
    """A real four-antenna MS, so skysim/primary-beam cases get past opening it."""
    ms = str(tmp_path_factory.mktemp("fuzz") / "fuzz.ms")
    result = CliRunner().invoke(
        cli,
        ["telsim", "-tel", "meerkat", "-subrange", "0,3", "-nt", "2", "-nc", "2", "--nworkers", "1", ms],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, result.output
    return ms


def run(runner, args):
    return runner.invoke(cli, args, catch_exceptions=True)


def assert_clean_failure(result):
    """The invocation failed, and it failed as a reported error rather than a crash."""
    assert result.exit_code != 0, f"expected a failure, got exit 0:\n{result.output}"
    exception = result.exception
    if isinstance(exception, SystemExit):
        exception = None
    assert exception is None or isinstance(exception, (click.ClickException, click.UsageError)), (
        f"raw {type(exception).__name__} reached the user instead of a reported error: {exception}"
    )


# --------------------------------------------------------------------------------------
# Options on either side of the positional argument
# --------------------------------------------------------------------------------------

# (subcommand, the positional value, the options that must survive being placed after it).
TRAILING_OPTION_CASES = [
    ("telsim", "ms", ["-tel", "meerkat", "-nt", "2", "-nc", "2", "-subrange", "0,3", "--nworkers", "1"]),
    ("skysim", "ms", ["-as", "sky", "--column", "DATA", "--nworkers", "1"]),
    ("primary-beam", "to-fits", ["-bp", "meerkat", "--npix", "8", "--nchan", "1", "--nworkers", "1"]),
]


@pytest.mark.parametrize("cmd, positional, options", TRAILING_OPTION_CASES, ids=[c[0] for c in TRAILING_OPTION_CASES])
def test_options_are_accepted_after_the_positional_argument(runner, tmp_path, cmd, positional, options):
    """``simms telsim obs.ms --telescope meerkat`` must mean the same as the reverse order.

    ``chain=True`` on the root group makes click give every subcommand
    ``allow_interspersed_args=False``, so parsing stopped at the positional and everything
    after it was reassigned to a following command that does not exist -- surfacing as
    ``Error: No such option '-n'``, which names neither the option nor the real problem.
    """
    sky = tmp_path / "sky.txt"
    sky.write_text("#format: name ra dec stokes_i\ns0 15.0deg -31.0deg 1.0\n")
    value = {"ms": str(tmp_path / f"{cmd}.ms"), "to-fits": "to-fits"}[positional]
    options = [str(sky) if opt == "sky" else opt for opt in options]
    if cmd == "skysim":
        # skysim needs an MS to write into; build it first, in the working order.
        build = run(runner, ["telsim", "-tel", "meerkat", "-subrange", "0,3", "-nt", "2", "-nc", "2", value])
        assert build.exit_code == 0, build.output
    if cmd == "primary-beam":
        options += ["-o", str(tmp_path / "beam.fits")]

    before = run(runner, [cmd, *options, value])
    after = run(runner, [cmd, value, *options])

    assert before.exit_code == after.exit_code, (
        f"{cmd}: options placed after the positional gave exit {after.exit_code} but "
        f"exit {before.exit_code} before it:\n{after.output}"
    )
    assert "No such option" not in after.output


def test_eager_list_flag_works_after_the_positional(runner, tmp_path):
    """``-ls/--list`` is eager and expose_value=False, so it must not need the MS at all."""
    result = run(runner, ["telsim", str(tmp_path / "unused.ms"), "--list"])
    assert result.exit_code == 0, result.output
    assert "meerkat" in result.output


def test_chained_subcommands_still_split_their_arguments(runner, tmp_path):
    """Re-enabling interspersed args must not break the reason ``chain=True`` is there."""
    ms = str(tmp_path / "chained.ms")
    sky = tmp_path / "sky.txt"
    sky.write_text("#format: name ra dec stokes_i\ns0 15.0deg -31.0deg 1.0\n")
    telsim_args = ["telsim", "-tel", "meerkat", "-subrange", "0,3", "-nt", "2", "-nc", "2", "--nworkers", "1", ms]
    skysim_args = ["skysim", "-as", str(sky), "--column", "DATA", "--nworkers", "1", ms]
    result = run(runner, telsim_args + skysim_args)
    assert result.exit_code == 0, result.output


def test_a_file_named_like_a_subcommand_is_not_treated_as_one(runner, tmp_path):
    """The chain detector skips option *values*, so a sky model called ``skysim`` is a path."""
    ms = str(tmp_path / "named.ms")
    assert run(runner, ["telsim", "-tel", "meerkat", "-subrange", "0,3", "-nt", "2", "-nc", "2", ms]).exit_code == 0
    sky = tmp_path / "skysim"
    sky.write_text("#format: name ra dec stokes_i\ns0 15.0deg -31.0deg 1.0\n")
    result = run(runner, ["skysim", ms, "-as", str(sky), "--column", "DATA", "--nworkers", "1"])
    assert result.exit_code == 0, result.output


# --------------------------------------------------------------------------------------
# Shared state between invocations
# --------------------------------------------------------------------------------------


def test_chain_does_not_strip_the_ms_argument_from_later_invocations(runner):
    """``--chain`` moves ``ms`` up to the group, for that invocation only.

    The commands are built once at import and shared by every invocation in the process --
    a shinobi Recipe, dosho, this test module -- so filtering the parameter used to be done
    by deleting it from ``self.params``, which made one ``--chain`` run strip the MS
    argument from ``telsim`` permanently.
    """
    assert "MS" in run(runner, ["telsim", "--help"]).output
    run(runner, ["--chain", "-ms", "/nonexistent/x.ms", "telsim", "-tel", "nosuchtelescope"])
    assert "MS" in run(runner, ["telsim", "--help"]).output, "--chain stripped MS from the shared command"
    assert "Missing argument 'MS'" in run(runner, ["telsim", "-tel", "meerkat"]).output


def test_chain_hides_the_ms_argument_while_it_is_in_effect(runner):
    """The flip side: under ``--chain`` the MS comes from the group, not the subcommand."""
    result = run(runner, ["--chain", "-ms", "/nonexistent/x.ms", "telsim", "--help"])
    assert result.exit_code == 0, result.output
    assert "[OPTIONS] MS" not in result.output


# --------------------------------------------------------------------------------------
# Values that cannot describe a valid MS
# --------------------------------------------------------------------------------------

# Each of these produced an MS and exited 0.
INVALID_TELSIM_VALUES = [
    pytest.param(["-nc", "0"], "--nchan", id="nchan-zero"),
    pytest.param(["-nc", "-3"], "--nchan", id="nchan-negative"),
    pytest.param(["-nt", "0"], "--ntime", id="ntime-zero"),
    pytest.param(["-dt", "0"], "--dtime", id="dtime-zero"),
    pytest.param(["-dt", "-8"], "--dtime", id="dtime-negative"),
    pytest.param(["-rc", "0"], "--row-chunks", id="rowchunks-zero"),
    pytest.param(["--nworkers", "0"], "--nworkers", id="nworkers-zero"),
    pytest.param(["-sf=-1420MHz"], "--start-freq", id="startfreq-negative"),
    pytest.param(["-df", "0Hz", "-nc", "4"], "--chan-width", id="chanwidth-zero"),
    pytest.param(["-dir", "J2000,1h0m0s,-999d0m0s"], "--direction", id="dec-out-of-range"),
    pytest.param(["-dir", "J2000,1h0m0s"], "--direction", id="direction-too-few-parts"),
    pytest.param(["-corr", "ZZ,QQQ"], "--correlations", id="unknown-correlation"),
    pytest.param(["-corr", ","], "--correlations", id="empty-correlations"),
    pytest.param(["-corr", "XX,YY,XY"], "--correlations", id="three-correlations"),
    pytest.param(["-lsl", "80", "-hsl", "10"], "--low-source-limit", id="elevation-limits-swapped"),
    pytest.param(["-fr", "1GHz,2GHz"], "--freq-range", id="freq-range-too-few-parts"),
    pytest.param(["-fr", "1GHz,2GHz,1"], "--freq-range", id="freq-range-single-channel"),
]


@pytest.mark.parametrize("extra, option", INVALID_TELSIM_VALUES)
def test_telsim_rejects_values_that_cannot_make_a_valid_ms(runner, tmp_path, extra, option):
    """A rejected value must name the option the user typed, and write nothing."""
    ms = str(tmp_path / "rejected.ms")
    result = run(runner, ["telsim", "-tel", "meerkat", "-subrange", "0,3", *extra, ms])
    assert_clean_failure(result)
    assert option in result.output, f"the error does not name {option}:\n{result.output}"
    assert not os.path.exists(ms), f"{option} was rejected but an MS was still written"


def test_a_late_failure_does_not_destroy_the_existing_ms(runner, tmp_path):
    """``create_ms`` used to ``remove_ms`` on entry, so a ``--telescope`` typo deleted the MS
    already at the target path and then exited with an error, leaving nothing behind."""
    ms = str(tmp_path / "precious.ms")
    assert run(runner, ["telsim", "-tel", "meerkat", "-subrange", "0,3", "-nt", "2", "-nc", "2", ms]).exit_code == 0

    for bad in (["-tel", "meerkatt"], ["-tel", "meerkat", "-sublist", "NOSUCHANT"]):
        result = run(runner, ["telsim", *bad, ms])
        assert_clean_failure(result)
        assert os.path.exists(ms), f"{bad} destroyed the existing MS before failing"

    # ...while a successful run still overwrites it, which is the documented behaviour.
    assert run(runner, ["telsim", "-tel", "meerkat", "-subrange", "0,3", "-nt", "3", "-nc", "2", ms]).exit_code == 0


def test_a_valid_ms_never_carries_a_non_positive_exposure(runner, tmp_path):
    """The concrete damage ``--dtime -8`` did: a negative EXPOSURE is written straight into
    the MS, and the thermal-noise ``1/sqrt(2 dnu dt)`` then fills DATA with NaNs -- with
    both telsim and skysim reporting success."""
    import numpy as np
    from daskms import xds_from_ms

    ms = str(tmp_path / "exposure.ms")
    assert run(runner, ["telsim", "-tel", "meerkat", "-subrange", "0,3", "-nt", "2", "-nc", "2", ms]).exit_code == 0
    dataset = xds_from_ms(ms)[0]
    assert (dataset.EXPOSURE.values > 0).all()
    assert (dataset.INTERVAL.values > 0).all()

    sky = tmp_path / "sky.txt"
    sky.write_text("#format: name ra dec stokes_i\ns0 15.0deg -31.0deg 1.0\n")
    result = run(runner, ["skysim", "-as", str(sky), "--sefd", "400", "--column", "DATA", "--nworkers", "1", ms])
    assert result.exit_code == 0, result.output
    assert np.isfinite(xds_from_ms(ms)[0].DATA.values).all(), "the predicted DATA column is not finite"


# --------------------------------------------------------------------------------------
# Malformed input on every subcommand: never a traceback
# --------------------------------------------------------------------------------------


def _malformed_cases(ms, missing, empty_file, binary_junk):
    """(id, argv) for input that is wrong in every way a user manages to get wrong."""
    return [
        # telsim
        ("telsim-unknown-telescope", ["telsim", "-tel", "nosuchtelescope", missing]),
        ("telsim-empty-telescope", ["telsim", "-tel", "", missing]),
        ("telsim-telescope-path-traversal", ["telsim", "-tel", "../../etc/passwd", missing]),
        ("telsim-bad-direction", ["telsim", "-tel", "meerkat", "-dir", "garbage", missing]),
        ("telsim-bad-starttime", ["telsim", "-tel", "meerkat", "-st", "not-a-time", missing]),
        ("telsim-bad-startfreq", ["telsim", "-tel", "meerkat", "-sf", "notafreq", missing]),
        ("telsim-startfreq-wrong-unit", ["telsim", "-tel", "meerkat", "-sf", "5Jy", missing]),
        ("telsim-subrange-nonnumeric", ["telsim", "-tel", "meerkat", "-subrange", "a,b", missing]),
        ("telsim-subrange-reversed", ["telsim", "-tel", "meerkat", "-subrange", "10,2", missing]),
        ("telsim-subrange-out-of-range", ["telsim", "-tel", "meerkat", "-subrange", "0,100000", missing]),
        ("telsim-sublist-unknown", ["telsim", "-tel", "meerkat", "-sublist", "NOSUCHANT", missing]),
        ("telsim-subfile-missing", ["telsim", "-tel", "meerkat", "-subfile", missing, missing]),
        ("telsim-subfile-empty", ["telsim", "-tel", "meerkat", "-subfile", empty_file, missing]),
        ("telsim-subfile-binary", ["telsim", "-tel", "meerkat", "-subfile", binary_junk, missing]),
        ("telsim-sensitivity-missing", ["telsim", "-tel", "meerkat", "-sfile", missing, missing]),
        ("telsim-sensitivity-binary", ["telsim", "-tel", "meerkat", "-sfile", binary_junk, missing]),
        ("telsim-freq-range-nonnumeric", ["telsim", "-tel", "meerkat", "-fr", "a,b,c", missing]),
        ("telsim-unwritable-path", ["telsim", "-tel", "meerkat", "-nt", "1", "-nc", "1", "/proc/nope/x.ms"]),
        # skysim
        ("skysim-no-sky-model", ["skysim", ms]),
        ("skysim-missing-ms", ["skysim", "-as", empty_file, missing]),
        ("skysim-ascii-missing", ["skysim", "-as", missing, ms]),
        ("skysim-ascii-empty", ["skysim", "-as", empty_file, ms]),
        ("skysim-ascii-binary", ["skysim", "-as", binary_junk, ms]),
        ("skysim-fits-missing", ["skysim", "-fs", missing, ms]),
        ("skysim-fits-binary", ["skysim", "-fs", binary_junk, ms]),
        ("skysim-two-sky-models", ["skysim", "-as", empty_file, "-fs", binary_junk, ms]),
        ("skysim-bad-mode", ["skysim", "-as", empty_file, "--mode", "nonsense", ms]),
        ("skysim-bad-smearing", ["skysim", "-as", empty_file, "--smearing", "nonsense", ms]),
        ("skysim-subsamples-zero", ["skysim", "-as", empty_file, "--smearing-subsamples", "0", ms]),
        ("skysim-rowchunks-zero", ["skysim", "-as", empty_file, "--row-chunks", "0", ms]),
        ("skysim-rowchunks-negative", ["skysim", "-as", empty_file, "--row-chunks", "-5", ms]),
        ("skysim-nworkers-zero", ["skysim", "-as", empty_file, "--nworkers", "0", ms]),
        ("skysim-field-id-out-of-range", ["skysim", "-as", empty_file, "--field-id", "999", ms]),
        ("skysim-field-id-negative", ["skysim", "-as", empty_file, "--field-id", "-1", ms]),
        ("skysim-spw-out-of-range", ["skysim", "-as", empty_file, "--spw-id", "999", ms]),
        ("skysim-corruptions-missing", ["skysim", "-as", empty_file, "--corruptions", missing, ms]),
        ("skysim-corruptions-binary", ["skysim", "-as", empty_file, "--corruptions", binary_junk, ms]),
        ("skysim-beam-missing", ["skysim", "-as", empty_file, "--primary-beam", missing, ms]),
        ("skysim-schema-missing", ["skysim", "-as", empty_file, "--source-schema", missing, ms]),
        ("skysim-schema-binary", ["skysim", "-as", empty_file, "--source-schema", binary_junk, ms]),
        ("skysim-seed-not-an-int", ["skysim", "-as", empty_file, "--seed-noise", "abc", ms]),
        # primary-beam
        ("pb-unknown-mode", ["primary-beam", "nonsense"]),
        ("pb-to-fits-no-pattern", ["primary-beam", "to-fits"]),
        ("pb-to-fits-missing-pattern", ["primary-beam", "-bp", missing, "to-fits"]),
        ("pb-to-fits-binary-pattern", ["primary-beam", "-bp", binary_junk, "to-fits"]),
        ("pb-to-fits-npix-zero", ["primary-beam", "-bp", "meerkat", "--npix", "0", "to-fits"]),
        ("pb-to-fits-npix-negative", ["primary-beam", "-bp", "meerkat", "--npix", "-4", "to-fits"]),
        ("pb-to-fits-bad-pixel-size", ["primary-beam", "-bp", "meerkat", "--pixel-size", "junk", "to-fits"]),
        ("pb-to-fits-nchan-zero", ["primary-beam", "-bp", "meerkat", "--nchan", "0", "to-fits"]),
        ("pb-to-fits-bad-band", ["primary-beam", "-bp", "meerkat", "--beam-band", "Q", "to-fits"]),
        ("pb-to-fits-nworkers-zero", ["primary-beam", "-bp", "meerkat", "--nworkers", "0", "to-fits"]),
        ("pb-tag-ms-no-ms", ["primary-beam", "tag-ms"]),
        ("pb-tag-ms-missing-ms", ["primary-beam", "--ms", missing, "--label", "X", "tag-ms"]),
        ("pb-tag-ms-no-label", ["primary-beam", "--ms", ms, "tag-ms"]),
        ("pb-apply-no-sky", ["primary-beam", "--ms", ms, "-bp", "meerkat", "apply"]),
        (
            "pb-apply-two-skies",
            ["primary-beam", "--ms", ms, "-bp", "meerkat", "-fits", binary_junk, "-ascii", empty_file, "apply"],
        ),
        ("pb-correct-missing-sky", ["primary-beam", "--ms", ms, "-bp", "meerkat", "-ascii", missing, "correct"]),
        # root group
        ("group-unknown-command", ["nosuchcommand"]),
        ("group-chain-without-ms", ["--chain", "telsim", "-tel", "meerkat"]),
        ("group-bad-log-level", ["--log-level", "LOUD", "telsim", "--help"]),
    ]


def test_malformed_input_is_reported_not_raised(runner, small_ms, missing, empty_file, binary_junk):
    """No malformed input reaches the user as a Python traceback.

    Run as one test over every case rather than a parametrised sweep so a regression reports
    the whole set at once: these share a single cause (the absence of a top-level handler),
    and 60 near-identical failures are harder to read than one list.
    """
    crashed = []
    for case_id, args in _malformed_cases(small_ms, missing, empty_file, binary_junk):
        result = run(runner, args)
        exception = result.exception
        if isinstance(exception, SystemExit):
            exception = None
        if result.exit_code == 0:
            crashed.append(f"{case_id}: exited 0 on invalid input")
        elif exception is not None and not isinstance(exception, (click.ClickException, click.UsageError)):
            crashed.append(f"{case_id}: raw {type(exception).__name__}: {exception}")
    assert not crashed, "malformed input did not fail cleanly:\n  " + "\n  ".join(crashed)


def test_the_traceback_is_still_available_under_debug(runner, missing, small_ms):
    """Folding the stack away must not make a real bug undebuggable."""
    args = ["skysim", "-as", missing, small_ms]
    assert isinstance(run(runner, ["--log-level", "DEBUG", *args]).exception, FileNotFoundError)
    assert not isinstance(run(runner, args).exception, FileNotFoundError)


def test_simms_errors_are_reported_without_their_type(runner, tmp_path):
    """simms' own errors are written for the user, so the class name is noise; anything else
    keeps its type, which is all that is left of the traceback."""
    ms = str(tmp_path / "typed.ms")
    own = run(runner, ["telsim", "-tel", "meerkat", "-nt", "0", ms])
    assert "InvalidInputError" not in own.output
    assert "Error: --ntime must be at least 1" in own.output

    foreign = run(runner, ["telsim", "-tel", "nosuchtelescope", ms])
    assert "FileNotFoundError" in foreign.output


# --------------------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------------------


def test_debug_is_an_accepted_log_level(runner):
    """``set_logger`` has always honoured DEBUG; the click.Choice omitted it, so the one
    level worth asking for was the one level the CLI rejected."""
    assert run(runner, ["--log-level", "DEBUG", "telsim", "--help"]).exit_code == 0


def test_log_levels_are_case_insensitive(runner):
    assert run(runner, ["--log-level", "debug", "telsim", "--help"]).exit_code == 0
    assert run(runner, ["-ll", "warning", "telsim", "--help"]).exit_code == 0


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_every_offered_log_level_is_a_real_level(runner, level):
    """A level the CLI offers but ``set_logger`` does not know would silently land on the
    fallback, quietly logging at some other verbosity than the one asked for."""
    assert run(runner, ["--log-level", level, "telsim", "--help"]).exit_code == 0
    assert set_logger(f"fuzz-{level}", level).level == getattr(logging, level)


def test_an_unknown_log_level_falls_back_to_info():
    """The fallback used to be 10 (DEBUG), so a typo turned the logging all the way *up*."""
    assert set_logger("fuzz-unknown", "NOSUCHLEVEL").level == logging.INFO


# --------------------------------------------------------------------------------------
# Exception hierarchy
# --------------------------------------------------------------------------------------


def test_simms_exceptions_share_a_base():
    """``SimmsGroup`` tells its own errors apart by this base, so an exception added outside
    the hierarchy would print with a redundant class name."""
    from simms import exceptions

    own = [
        value
        for name, value in vars(exceptions).items()
        if isinstance(value, type) and issubclass(value, Exception) and not name.startswith("_")
    ]
    strays = [cls.__name__ for cls in own if not issubclass(cls, SimmsError)]
    assert not strays, f"exceptions outside the SimmsError hierarchy: {strays}"
    assert issubclass(InvalidInputError, ValueError), "InvalidInputError must stay catchable as a ValueError"
