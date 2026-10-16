"""Contract tests for the generated CLI surface.

The options of every subcommand are generated from its pydantic ``inputs_model`` by
simms' ``build_options``, so ``--help`` cannot drift from the model. It can still drift
from the *code*: an option can be declared and never read, and a help string can promise a
value syntax that the rendered click type rejects. Both have happened. These tests check the
whole option surface at once, so a new option cannot reintroduce either.
"""

import inspect
import pathlib
import re

import click
import pytest

from simms.apps import primary_beam, skysim, telsim
from simms.apps.main import FLAG_ALIASES, cli
from simms.apps.schema import build_options
from simms.skymodel import pb_ops
from simms.telescope import generate_ms

SRC = pathlib.Path(inspect.getfile(telsim)).parent.parent

STEPS = {
    "telsim": telsim,
    "skysim": skysim,
    "primary-beam": primary_beam,
}

# Modules a subcommand hands its `opts` (or the values pulled off it) to. An option is
# "consumed" if it is read as `opts.<name>` somewhere on this path.
CONSUMER_MODULES = {
    "telsim": [telsim, generate_ms],
    "skysim": [skysim],
    "primary-beam": [primary_beam, pb_ops],
}

# Read on a subcommand's path but not via `opts.<name>`; `ms` is the click Argument, and
# `log_level` is threaded in by the root group rather than the step callback.
NOT_READ_VIA_OPTS = {"log_level"}


def consumer_source(cmd):
    text = "\n".join(inspect.getsource(module) for module in CONSUMER_MODULES[cmd])
    if cmd in ("skysim", "primary-beam"):
        # These fan out across the sky-model package rather than a single module.
        text += "\n".join(path.read_text() for path in (SRC / "skymodel").rglob("*.py"))
    return text


def options(cmd):
    return [opt for opt in build_options(STEPS[cmd].inputs_model) if opt.name not in NOT_READ_VIA_OPTS]


@pytest.mark.parametrize("cmd", sorted(STEPS))
def test_every_option_is_consumed(cmd):
    """No declared-but-ignored flags.

    `--nworkers` was accepted, documented and defaulted on telsim and primary-beam while
    nothing read it, so asking for more workers silently did nothing.
    """
    text = consumer_source(cmd)
    unused = [opt.name for opt in options(cmd) if not re.search(rf"\bopts\.{opt.name}\b", text)]
    assert not unused, f"{cmd}: options declared but never read as opts.<name>: {unused}"


@pytest.mark.parametrize("cmd", sorted(STEPS))
def test_list_options_render_as_strings(cmd):
    """List-typed fields become click ``multiple=True`` options.

    A bare ``list[int]`` renders as INTEGER and rejects a comma-separated value
    before any simms code runs. Including a string arm keeps both the repeated
    and the comma form working.
    """
    bad = [
        (opt.name, opt.type.name)
        for opt in build_options(STEPS[cmd].inputs_model)
        if opt.multiple and opt.type is not click.STRING
    ]
    assert not bad, (
        f"{cmd}: list options must render as STRING so a comma-separated value parses; "
        f"annotate as `list[str | ...]`. Offenders: {bad}"
    )


@pytest.mark.parametrize("cmd", sorted(STEPS))
def test_help_promising_comma_values_can_accept_them(cmd):
    """An option whose help shows a comma-separated example must actually take one: either a
    ``multiple=True`` STRING option (split downstream) or a plain string the app splits itself.
    """
    text = consumer_source(cmd)
    offenders = []
    for opt in options(cmd):
        help_text = opt.help or ""
        # A comma-joined example value, e.g. "XX,YY" or "start,end,step" -- but not prose
        # commas ("fast, but approximate") which always have a space after the comma.
        if not re.search(r"[\w.\-]+,[\w.\-]+", help_text):
            continue
        if opt.multiple and opt.type is click.STRING:
            continue
        if re.search(rf"opts\.{opt.name}\s*\.split\(|{opt.name}\s*=\s*\w+\.split\(", text):
            continue
        # The value names a file whose *contents* are a list; the flag itself takes a path.
        if opt.type.name == "text" and "file" in opt.name:
            continue
        offenders.append(opt.name)
    assert not offenders, (
        f"{cmd}: help text shows a comma-separated example for {offenders}, but the option "
        f"neither splits on commas nor renders as a repeatable STRING option"
    )


@pytest.mark.parametrize("cmd", sorted(STEPS))
def test_no_duplicate_flags(cmd):
    """Two fields claiming the same flag or abbreviation would let one silently shadow the
    other, since click resolves by flag string. Checked on the *assembled* command, so an
    alias in FLAG_ALIASES colliding with a real flag fails here too.
    """
    seen = {}
    for param in cli.commands[cmd].params:
        for flag in param.opts + param.secondary_opts:
            assert flag not in seen, f"{cmd}: {flag} claimed by both {seen[flag]} and {param.name}"
            seen[flag] = param.name


def test_flag_aliases_name_real_fields():
    """An alias for a field that no longer exists would silently do nothing."""
    for cmd, aliases in FLAG_ALIASES.items():
        fields = set(STEPS[cmd].inputs_model.model_fields)
        unknown = sorted(set(aliases) - fields)
        assert not unknown, f"{cmd}: FLAG_ALIASES names fields that do not exist: {unknown}"


def test_flag_aliases_resolve_to_the_same_field():
    """Each alias is a second spelling of one option, not a separate parameter: both flags
    must land on the same callback kwarg, or one of them would silently be discarded.
    """
    for cmd, aliases in FLAG_ALIASES.items():
        params = {param.name: param for param in cli.commands[cmd].params}
        for field, alias in aliases.items():
            param = params[field]
            assert alias in param.opts, f"{cmd}: {alias} is not registered on --{field.replace('_', '-')}"
            assert param.name == field, f"{cmd}: {alias} changed the callback kwarg to {param.name!r}"


def test_drifted_spellings_are_accepted_on_every_subcommand():
    """The point of the aliases: a spelling that works on one subcommand is not rejected by
    another. telsim/skysim/primary-beam each named these three concepts differently.
    """
    for spellings in (
        ("--rowchunks", "--row-chunks"),
        ("--startfreq", "--start-freq"),
        ("--dfreq", "--chan-width"),
    ):
        commands = [
            cmd for cmd in STEPS if any(flag in spellings for param in cli.commands[cmd].params for flag in param.opts)
        ]
        for cmd in commands:
            flags = {flag for param in cli.commands[cmd].params for flag in param.opts}
            missing = [spelling for spelling in spellings if spelling not in flags]
            assert not missing, f"{cmd} accepts {set(spellings) - set(missing)} but not {missing}"


@pytest.mark.parametrize("cmd", sorted(STEPS))
def test_every_option_has_help(cmd):
    missing = [opt.name for opt in build_options(STEPS[cmd].inputs_model) if not opt.help]
    assert not missing, f"{cmd}: options with no help text: {missing}"
