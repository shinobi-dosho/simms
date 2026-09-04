from __future__ import annotations

from types import SimpleNamespace
from typing import Annotated

import shinobi
from dask import config as dask_config
from pydantic import BaseModel, Field
from shinobi.steps.schema import ParamMeta

from simms import BIN, set_logger


class PrimaryBeamOutputs(BaseModel):
    """Passthrough paths of the primary-beam operation, so it can be wired into a
    shinobi Recipe or dosho: `to-fits`/`apply`/`correct` write `output` (beam FITS
    or beamed sky model), while `tag-ms` mutates `ms` in place and echoes it back --
    the only handle a dependent step has to chain onto the tagged MS.

    `output` is the path the run *resolved*, not the `--output` it was given: the modes
    default the filename when `--output` is omitted, so echoing the raw option back would
    hand a dependent step `None` in exactly that case. For `--fits-format cattery`, which
    writes eight files, `output` is the bare prefix -- what `from_cattery` and DDFacet's
    `--Beam-FITSFile` consume, so it round-trips into a later `--beam-pattern`.
    `files` lists every file written (empty for `tag-ms`, which writes none)."""

    ms: str | None = None
    output: str | None = None
    files: list[str] = []


def _require(opts, field):
    if not getattr(opts, field, None):
        raise RuntimeError(f"--{field.replace('_', '-')} is required for mode {opts.mode!r}.")


def runit(opts) -> PrimaryBeamOutputs:
    """Run one primary-beam mode and report the paths it actually wrote."""
    set_logger(BIN.primary_beam, opts.log_level)
    from simms.skymodel import pb_ops

    # pb_ops builds and computes dask graphs, so --nworkers has to reach the scheduler the
    # same way skysim sets it; without this the option was accepted and silently ignored.
    dask_config.set(scheduler="threads", num_workers=opts.nworkers)

    mode = opts.mode
    if mode == "to-fits":
        _require(opts, "beam_pattern")
        output, files = pb_ops.to_fits(opts)
        return PrimaryBeamOutputs(ms=opts.ms, output=output, files=files)
    elif mode == "tag-ms":
        _require(opts, "ms")
        pb_ops.tag_ms(opts)
        # tag-ms edits ANTENNA in place and writes no new file, so the echoed MS is the handle.
        return PrimaryBeamOutputs(ms=opts.ms, output=None, files=[])
    elif mode in ("apply", "correct"):
        _require(opts, "ms")
        _require(opts, "beam_pattern")
        if sum(bool(x) for x in (opts.fits_sky, opts.ascii_sky)) != 1:
            raise RuntimeError("apply/correct needs exactly one of --fits-sky or --ascii-sky.")
        invert = mode == "correct"
        run = pb_ops.apply_correct_image if opts.fits_sky else pb_ops.apply_correct_ascii
        output = run(opts, invert)
        return PrimaryBeamOutputs(ms=opts.ms, output=output, files=[output])
    else:
        raise RuntimeError(f"Unknown primary-beam mode {mode!r}.")


@shinobi.pystep(
    name=BIN.primary_beam, info="Primary-beam utilities (build/tag/apply/correct); no visibility simulation."
)
def primary_beam(
    mode: Annotated[str, ParamMeta(choices=["to-fits", "tag-ms", "apply", "correct"])] = Field(
        ..., description="Operation to perform."
    ),
    beam_pattern: str | None = Field(
        None,
        description="Beam model: a cosine-taper CSV path, a built-in name, a band shorthand (L/UHF), or a FITS cube.",
        json_schema_extra={"abbreviation": "bp"},
    ),
    beam_band: Annotated[str, ParamMeta(choices=["UHF", "L"])] = Field(
        "L", description="Default band for a built-in beam when beam-pattern omits one."
    ),
    beam_pa_step: float = Field(
        1.0, description="Parallactic-angle sampling step (degrees) for the time-averaged beam."
    ),
    fits_format: Annotated[str, ParamMeta(choices=["simms", "cattery"])] = Field(
        "simms",
        description="to-fits output layout: simms's own single-file 4-plane HH/VV cube, or "
        "the Cattery/DDFacet 8-file per-Jones-element schema (--Beam-Model FITS).",
        json_schema_extra={"abbreviation": "ff"},
    ),
    pol_basis: Annotated[str, ParamMeta(choices=["linear", "circular"])] = Field(
        "linear",
        description="Correlation basis for the cattery fits-format output (xx/xy/yx/yy vs "
        "rr/rl/lr/ll); must match the target MS's feed basis.",
        json_schema_extra={"abbreviation": "pol"},
    ),
    beam_l_axis: Annotated[str, ParamMeta(choices=["-X", "X"])] = Field(
        "-X",
        description="Sign convention for the cattery fits-format L axis, matching DDFacet's "
        "--Beam-FITSLAxis (pass the same value to both).",
        json_schema_extra={"abbreviation": "bla"},
    ),
    beam_m_axis: Annotated[str, ParamMeta(choices=["Y", "-Y"])] = Field(
        "Y",
        description="Sign convention for the cattery fits-format M axis, matching DDFacet's "
        "--Beam-FITSMAxis (pass the same value to both).",
        json_schema_extra={"abbreviation": "bma"},
    ),
    ms: str | None = Field(
        None,
        description="Measurement set (time/PA range, array position and frequencies). "
        "Required for tag-ms/apply/correct.",
    ),
    fits_sky: str | None = Field(
        None, description="Input FITS image sky model (apply/correct).", json_schema_extra={"abbreviation": "fits"}
    ),
    ascii_sky: str | None = Field(
        None,
        description="Input ASCII component sky model (apply/correct).",
        json_schema_extra={"abbreviation": "ascii"},
    ),
    ascii_delimiter: str | None = Field(
        None,
        description="Delimiter used in the ascii-sky file. Defaults to whitespace.",
        json_schema_extra={"abbreviation": "ad"},
    ),
    source_schema: str | None = Field(
        None,
        description="Custom source schema (YAML) mapping the ascii-sky columns to the fields simms expects.",
    ),
    output: str | None = Field(
        None,
        description="Output path - FITS beam (to-fits, or filename prefix when --fits-format cattery) "
        "or beamed/corrected sky model (apply/correct).",
        json_schema_extra={"abbreviation": "o"},
    ),
    telescope_name_column: str = Field(
        "TELESCOPE_NAME",
        description="ANTENNA-table column holding the per-antenna telescope/type label (tag-ms).",
        json_schema_extra={"abbreviation": "tnc"},
    ),
    label: str | None = Field(None, description="Single telescope-name label applied to all antennas (tag-ms)."),
    label_map: str | None = Field(None, description="YAML mapping antenna NAME -> telescope-name label (tag-ms)."),
    from_layout: str | None = Field(
        None,
        description="simms layout whose per-antenna telescope_name is matched to the MS antenna names (tag-ms).",
    ),
    pb_cutoff: float = Field(0.1, description="In correct mode, blank (NaN) where the beam is below this level."),
    field_id: int = Field(
        0,
        description="FIELD_ID whose phase centre and time span define the beam (apply/correct).",
        json_schema_extra={"abbreviation": "fi"},
    ),
    spw_id: int = Field(
        0, description="Spectral-window (DATA_DESC_ID) whose frequencies define the beam (apply/correct)."
    ),
    pixel_size: str = Field("1arcmin", description="Angular pixel size for the to-fits grid, e.g. '1arcmin'."),
    npix: int = Field(256, description="Number of pixels per side for the to-fits grid."),
    start_freq: str | None = Field(
        None,
        description="Start frequency of the to-fits cube, e.g. '856MHz'. Defaults to the beam's first frequency.",
        json_schema_extra={"abbreviation": "sf"},
    ),
    chan_width: str | None = Field(
        None,
        description="Channel width of the to-fits cube, e.g. '10MHz'. "
        "Defaults to spanning the beam range across nchan.",
        json_schema_extra={"abbreviation": "cw"},
    ),
    nchan: int | None = Field(
        None,
        description="Number of output channels in the to-fits cube. Defaults to the beam's channel count.",
        json_schema_extra={"abbreviation": "nc"},
    ),
    nworkers: int = Field(4, description="Number of worker threads."),
    log_level: str = Field("INFO", description="Logging verbosity."),
) -> PrimaryBeamOutputs:
    opts = SimpleNamespace(**locals())
    return runit(opts)
