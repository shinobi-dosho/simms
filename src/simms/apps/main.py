from __future__ import annotations

import os

import click
from shinobi.clickutil import build_options, unflatten_kwargs
from shinobi.steps.dispatch import _dispatch

from simms import BIN, __version__
from simms.apps import primary_beam, skysim, telsim
from simms.exceptions import SimmsError


class SimmsCommand(click.Command):
    """A subcommand that accepts its options on either side of the positional argument.

    The root group is ``chain=True`` so that ``simms telsim ... skysim ...`` runs both in
    one go. click implements that by giving *every* subcommand context
    ``allow_interspersed_args=False`` (``Group.invoke``), so parsing stops at the first
    non-option token and the rest is handed to the next command in the chain. For a single
    subcommand -- the overwhelmingly common case -- that turns the natural
    ``simms telsim obs.ms --telescope meerkat`` into ``Error: No such option '-n'``: every
    option after the MS is quietly reassigned to a command that does not exist.

    So interspersing is re-enabled whenever nothing in the remaining arguments names another
    subcommand, and left off when something does, where the args really do have to be split.
    """

    def _chains_into_another_command(self, args, ctx):
        """True when a later subcommand name appears in ``args`` as a command, not a value."""
        commands = set(ctx.parent.command.commands) if ctx.parent else set()
        if not commands:
            return False
        # Flags that consume the token after them, so a file called "skysim" passed as
        # `--ascii-sky skysim` is not mistaken for the start of a chained command.
        takes_value = {
            flag
            for param in self.get_params(ctx)
            if isinstance(param, click.Option) and not param.is_flag
            for flag in param.opts + param.secondary_opts
        }
        index = 0
        while index < len(args):
            token = args[index]
            if token in takes_value:
                index += 2
            elif token.startswith("-"):
                index += 1
            elif token in commands:
                return True
            else:
                index += 1
        return False

    def make_context(self, info_name, args, parent=None, **extra):
        ctx = click.Context(self, info_name=info_name, parent=parent)
        if not self._chains_into_another_command(args, ctx):
            extra["allow_interspersed_args"] = True
        return super().make_context(info_name, args, parent=parent, **extra)


class RemoveMSIfChained(SimmsCommand):
    """A subcommand whose ``ms`` argument moves up to the group under ``--chain``.

    The parameter is filtered out of ``get_params`` rather than deleted from ``self.params``:
    the commands are built once at import and shared by every invocation in the process (a
    shinobi Recipe, dosho, an in-process test), so mutating them made one ``--chain`` run
    strip the MS argument from ``telsim`` for good. ``get_params`` is consulted by parsing
    and by ``--help`` alike, so both stay consistent without touching shared state.
    """

    def get_params(self, ctx):
        params = super().get_params(ctx)
        if ctx.parent and ctx.parent.params.get("chain", False):
            return [param for param in params if param.name != "ms"]
        return params


class SimmsGroup(click.Group):
    """The root group, with user-facing errors reported as errors rather than tracebacks.

    Everything simms raises below the CLI -- a missing sky model, an unreadable layout, an
    option value that cannot describe an MS -- used to reach the terminal as a bare Python
    traceback. The stack is noise to someone who mistyped a path, so it is folded into a
    one-line ``Error:`` and kept available on demand: ``--log-level DEBUG``, or
    ``SIMMS_TRACEBACK=1`` for a failure that happens before the log level is parsed.
    """

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.exceptions.Abort, click.exceptions.Exit, SystemExit):
            # click.exceptions.Exit is how --help and --version end; it subclasses
            # RuntimeError, so it has to be named explicitly or a successful --help
            # would be reported as a failure.
            raise
        except Exception as exc:
            if ctx.params.get("log_level") == "DEBUG" or os.environ.get("SIMMS_TRACEBACK"):
                raise
            # simms' own errors describe the input and read fine alone; anything else keeps
            # its type, which is the only clue left once the traceback is gone.
            detail = str(exc) if isinstance(exc, SimmsError) else f"{type(exc).__name__}: {exc}"
            raise click.ClickException(f"{detail}\n(re-run with --log-level DEBUG for the full traceback)") from exc


# Long-flag aliases, keyed by subcommand and field name.
#
# telsim predates skysim and primary-beam and spells three shared concepts differently:
# --rowchunks vs --row-chunks, --startfreq vs --start-freq, --dfreq vs --chan-width. Renaming
# either side would break existing scripts and Stimela recipes, and telsim's spellings are
# internally symmetric (--startfreq/--starttime, --dfreq/--dtime), so neither side is simply
# wrong. Each spelling is instead accepted on both commands and resolves to the same field, so
# a flag that works on one subcommand is never rejected by another. `--help` lists both.
FLAG_ALIASES = {
    "telsim": {
        "rowchunks": "--row-chunks",
        "startfreq": "--start-freq",
        "dfreq": "--chan-width",
    },
    "skysim": {
        "row_chunks": "--rowchunks",
    },
    "primary-beam": {
        "start_freq": "--startfreq",
        "chan_width": "--dfreq",
    },
}


def _add_alias(opt, alias):
    """Add a second long flag to an already-built click Option.

    click's parser reads `Parameter.opts` when the command is invoked, while the callback
    kwarg name was derived from the primary flag at construction, so appending here adds a
    spelling without touching the round-trip back to the model field.
    """
    if alias not in opt.opts:
        opt.opts.append(alias)


def _make_command(step, *, positional, chained, extra_options=()):
    """Build a `click.Command` for a `@shinobi.pystep` StepRef.

    Options come from the step's pydantic ``inputs_model`` via shinobi's
    ``build_options`` (choices, abbreviations, bool/list handling). The
    ``log_level`` field is dropped -- the root group's ``--log-level``
    controls it -- and the ``positional`` field is rendered as a
    ``click.Argument`` (build_options only emits ``--options``), and any
    field in ``FLAG_ALIASES`` gains a second long flag. The
    callback re-nests the flat kwargs and dispatches the step in-process
    via shinobi, exactly as ``shinobi.cli``'s ``run`` command does.
    """
    options = [opt for opt in build_options(step.step.inputs_model) if opt.name != "log_level"]

    aliases = FLAG_ALIASES.get(step.step.name, {})
    for opt in options:
        if opt.name in aliases:
            _add_alias(opt, aliases[opt.name])

    params = list(extra_options)
    for opt in options:
        if opt.name == positional:
            params.append(click.Argument([positional], required=True, type=opt.type))
        else:
            params.append(opt)

    def _callback(**raw):
        ctx = click.get_current_context()
        kwargs = unflatten_kwargs(step.step.inputs_model, raw)
        kwargs["log_level"] = ctx.obj["log_level"]
        if chained and ctx.obj["chain"]:
            kwargs["ms"] = ctx.obj["ms"]
        result = _dispatch(step.step, step.func, **kwargs)
        if not result.success:
            raise click.ClickException(f"{step.step.name!r} failed (returncode {result.returncode}).")

    cls = RemoveMSIfChained if chained else SimmsCommand
    return cls(
        name=step.step.name,
        params=params,
        callback=_callback,
        help=step.step.info,
        no_args_is_help=True,
    )


@click.group(cls=SimmsGroup, chain=True, no_args_is_help=True)
@click.version_option(str(__version__))
@click.option("--ms", "-ms", help="MS to create and then populate; required with --chain.")
@click.option(
    "--log-level",
    "-ll",
    help="Log level. DEBUG also restores the full traceback on an unexpected failure.",
    # DEBUG was missing while set_logger has always honoured it, so the most useful level was
    # the one level the CLI could not ask for. Case-insensitive so `-ll debug` works too.
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="INFO",
)
@click.option(
    "--chain",
    is_flag=True,
    help="Chain telsim and skysim for an End-to-End simulation."
    " When this option is set, the -ms/--ms has to be given in the "
    "main command and excluded from both sub-commands",
)
@click.pass_context
def cli(ctx, ms, log_level, chain):
    """
    Tools for simulating radio interferometry observations. 'telsim' creates a simulated observation
    (Measurement Set; MS), and 'skysim' populates an MS with visibilities generated from a given skymodel
    (FITS or ASCII format). For more info on the tools run:

        simms telsim --help

        simms skysim --help

    """

    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    ctx.obj["chain"] = chain
    if chain:
        if ms:
            ctx.obj["ms"] = ms
        else:
            raise click.exceptions.MissingParameter(
                "The --ms/-ms option is required when --chain is set", param_type="Option", param_hint="--ms"
            )


_list_option = click.Option(
    ["--list", "-ls"],
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=telsim.print_data_database,
    help="Displays a list of available telescope array layouts",
)

cli.add_command(
    _make_command(telsim.telsim, positional="ms", chained=True, extra_options=[_list_option]), name=BIN.telsim
)
cli.add_command(_make_command(skysim.skysim, positional="ms", chained=True), name=BIN.skysim)
cli.add_command(_make_command(primary_beam.primary_beam, positional="mode", chained=False), name=BIN.primary_beam)
