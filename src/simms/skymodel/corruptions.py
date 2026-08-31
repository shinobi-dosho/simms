"""RIME-style visibility corruptions applied after prediction."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Literal

import astropy.units as u
import dask
import dask.array as da
import numpy as np
import yaml

from simms import BIN

log = logging.getLogger(BIN.skysim)

DIMENSION_UNITS = {
    "time": u.s,
    "frequency": u.Hz,
}

TermType = Literal["scalar", "diagonal", "full"]

#: Feeds each term type draws independent random phases for. ``scalar`` and
#: ``full`` share one phase per antenna; ``diagonal`` gets one per feed.
TERM_FEEDS: dict[str, int] = {"scalar": 1, "diagonal": 2, "full": 1}

#: Correlation counts each term type can be represented in.
TERM_NCORR: dict[str, tuple[int, ...]] = {"scalar": (1, 2, 4), "diagonal": (2, 4), "full": (4,)}


@dataclass
class TermSpec:
    """Description of a single Jones corruption term.

    ``type`` selects the Jones form:

    ``scalar``
        ``g * I`` -- one gain per antenna, both polarisations identical.
    ``diagonal``
        ``diag(g_x, g_y)`` -- independent per-feed gains, no leakage. Needs at
        least 2 correlations.
    ``full``
        a dense 2x2 Jones with leakage. Needs a 4-correlation MS.

    Left unset it is ``diagonal``, falling back to ``scalar`` only on a
    single-correlation MS. An explicit type that the MS cannot carry is an error
    rather than a silent downgrade.

    ``diagonal`` is the deprecated boolean spelling: ``true`` means ``scalar``
    and ``false`` means ``full`` (note it never meant the ``diagonal`` type).
    """

    label: str
    type: TermType | None = None
    diagonal: bool | None = None
    complex: bool = True
    axes: list[Literal["time", "frequency"]] = field(default_factory=list)
    period: float | int | str | dict[str, float | int | str] | None = None
    amplitude: float = 0.0


@dataclass
class CorruptionSpec:
    """A full corruption specification loaded from YAML."""

    terms: list[str]
    spec: list[TermSpec]


def _stable_label_hash(label: str) -> int:
    """Return a deterministic 32-bit integer hash for ``label``."""
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def _term_seed(random_seed: int | None, label: str) -> int | None:
    """Seed for a corruption term, derived from the global seed and label."""
    label_hash = _stable_label_hash(label)
    if random_seed is None:
        return label_hash
    return random_seed + label_hash


def _parse_period(value: float | int | str, axis: str) -> float:
    """Convert a period value to base units (seconds or Hz)."""
    if isinstance(value, str):
        quantity = u.Quantity(value)
        return float(quantity.to_value(DIMENSION_UNITS[axis]))
    return float(value)


def _normalise_period(
    period: float | int | str | dict[str, float | int | str] | None,
    axes: list[str],
) -> dict[str, float]:
    """Return a ``{axis: period_in_base_units}`` mapping."""
    if period is None:
        raise ValueError("'period' is required for every corruption term")

    if isinstance(period, dict):
        if set(period.keys()) != set(axes):
            raise ValueError(f"'period' keys {sorted(period.keys())} do not match axes {sorted(axes)}")
        return {axis: _parse_period(period[axis], axis) for axis in axes}

    if len(axes) != 1:
        raise ValueError(f"'period' must be a mapping when multiple axes are present; got {period!r}")
    return {axes[0]: _parse_period(period, axes[0])}


def load_corruption_spec(path: str) -> CorruptionSpec:
    """Load a corruption specification from a YAML file.

    Everything hangs off a top-level ``gains`` block. A file without one used to
    load as an empty spec, so a misspelled key -- or the wrong file entirely --
    ran to completion having corrupted nothing.

    ``gains.terms`` is normally a YAML list, but a plain string is accepted as
    shorthand and split on commas and whitespace, so ``terms: "G, B"`` and
    ``terms: [G, B]`` are the same thing.
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise RuntimeError(f"Corruption spec '{path}' is empty or is not a YAML mapping")

    gains = data.get("gains")
    if gains is None:
        raise RuntimeError(
            f"Corruption spec '{path}' has no top-level 'gains' block; found {sorted(data) or 'nothing'}"
        )
    if not isinstance(gains, dict):
        raise RuntimeError(f"Corruption spec '{path}': 'gains' must be a mapping, got {type(gains).__name__}")

    raw_terms = gains.get("terms", [])
    if isinstance(raw_terms, str):
        terms = [t.strip() for t in raw_terms.replace(",", " ").split() if t.strip()]
    else:
        terms = [str(t).strip() for t in raw_terms]

    specs = []
    for index, item in enumerate(gains.get("spec", [])):
        if not isinstance(item, dict):
            raise RuntimeError(f"Corruption spec '{path}': entry {index} of 'gains.spec' is not a mapping")
        try:
            specs.append(TermSpec(**item))
        except TypeError as exc:
            # The dataclass raises TypeError for a missing 'label' or an unknown
            # key; re-raise as the error type the rest of the loader uses, and
            # say which entry it was.
            described = item.get("label", f"entry {index}")
            raise RuntimeError(f"Corruption spec '{path}', term '{described}': {exc}") from exc

    for term in specs:
        if term.diagonal is not None:
            log.warning(
                "Corruption term '%s': 'diagonal: %s' is deprecated; use 'type: %s'. Note the new "
                "'diagonal' type is diag(g_x, g_y), which the boolean never meant.",
                term.label,
                str(bool(term.diagonal)).lower(),
                "scalar" if term.diagonal else "full",
            )

    return CorruptionSpec(terms=terms, spec=specs)


def resolve_type(spec: TermSpec, ncorr: int) -> str:
    """Resolve a term's Jones form against the MS correlation count.

    An unset type is ``diagonal`` -- independent per-feed gains, the usual
    starting point for a calibration-style corruption -- falling back to
    ``scalar`` only on a single-correlation MS, which has no second feed to give
    its own gain. Leakage (``full``) is never implied; it has to be asked for.
    """
    if spec.type is not None and spec.diagonal is not None:
        raise RuntimeError(
            f"Corruption term '{spec.label}' sets both 'type' and the deprecated 'diagonal'; use 'type' only"
        )

    if spec.type is not None:
        if spec.type not in TERM_FEEDS:
            raise RuntimeError(
                f"Corruption term '{spec.label}' has unknown type '{spec.type}'; expected one of {sorted(TERM_FEEDS)}"
            )
        return spec.type

    if spec.diagonal is not None:
        # The deprecation is warned about once, in load_corruption_spec: this is
        # called per term by both validate_spec and _build_term_params, so
        # warning here repeats it for every term on every run.
        return "scalar" if spec.diagonal else "full"

    return "diagonal" if ncorr in TERM_NCORR["diagonal"] else "scalar"


def validate_spec(spec: CorruptionSpec, ncorr: int) -> None:
    """Raise RuntimeError if the specification is unusable.

    Two rules, deliberately different in reach:

    *Structural* validity -- axes, period, amplitude sign, and the type
    declaration itself -- is required of every entry in ``gains.spec``, listed
    in ``gains.terms`` or not, so a typo in a term you have not enabled yet
    still fails now rather than the day you enable it.

    *MS compatibility* -- whether the term's Jones form fits this MS's
    correlation count -- applies only to the terms actually listed in
    ``gains.terms``. One ``spec`` block can then hold a library of terms and
    ``terms`` select among them, without an unused ``full`` entry vetoing a
    scalar-only run on a 2-correlation MS. :func:`needs_feed_basis` and the
    all-zero-amplitude check below draw the same line.
    """
    labels = [s.label for s in spec.spec]
    if len(labels) != len(set(labels)):
        raise RuntimeError(f"Corruption spec contains duplicate labels: {labels}")

    if not spec.terms:
        raise RuntimeError("Corruption spec lists no terms in 'gains.terms', so it would corrupt nothing")

    label_set = set(labels)
    seen_terms = set()
    for term in spec.terms:
        if not term:
            raise RuntimeError("Empty term label in corruption 'terms' list")
        if term in seen_terms:
            raise RuntimeError(f"Corruption term '{term}' is duplicated in 'terms'")
        seen_terms.add(term)
        if term not in label_set:
            raise RuntimeError(f"Corruption term '{term}' is not described in the spec")

    for s in spec.spec:
        if not s.axes:
            raise RuntimeError(f"Corruption term '{s.label}' has no axes")
        for axis in s.axes:
            if axis not in DIMENSION_UNITS:
                raise RuntimeError(f"Corruption term '{s.label}' has unknown axis '{axis}'")

        try:
            periods = _normalise_period(s.period, s.axes)
        except ValueError as exc:
            raise RuntimeError(f"Invalid period for corruption term '{s.label}': {exc}") from exc

        for axis, period in periods.items():
            if period <= 0:
                raise RuntimeError(f"Corruption term '{s.label}' period for '{axis}' must be positive")

        if s.amplitude < 0:
            raise RuntimeError(f"Corruption term '{s.label}' amplitude must be non-negative")

    # A single zero-amplitude term is legitimate (it is the identity, useful
    # alongside others); a spec where *every* term is zero corrupts nothing and
    # is a mistake worth reporting rather than a run that quietly does nothing.
    label_to_spec = {s_.label: s_ for s_ in spec.spec}
    if all(label_to_spec[term].amplitude == 0 for term in spec.terms):
        raise RuntimeError(
            "Every corruption term has amplitude 0, so the spec would leave the visibilities "
            "unchanged; give at least one term a non-zero amplitude"
        )

    # Structural validity is checked for every entry, listed or not -- resolving
    # the type raises on a type/diagonal conflict or an unknown type, and a typo
    # in a term you have not enabled yet is worth catching now.
    resolved = {s_.label: resolve_type(s_, ncorr) for s_ in spec.spec}

    # Compatibility with *this* MS applies only to the terms this run actually
    # multiplies in, so an unused entry cannot veto a run that never touches it.
    # Resolve before testing: a falsy non-bool from YAML (`diagonal: 0` parses as
    # int) would slip past an identity check and only fail inside the block
    # function, losing the clear up-front error.
    for term in spec.terms:
        term_type = resolved[term]
        allowed = TERM_NCORR[term_type]
        if ncorr not in allowed:
            raise RuntimeError(
                f"Corruption term '{term}' is type '{term_type}', which needs "
                f"{' or '.join(str(n) for n in allowed)} correlations; this MS has {ncorr}"
            )


def needs_feed_basis(spec: CorruptionSpec, ncorr: int) -> bool:
    """True if any listed term tells the two feeds apart.

    ``scalar`` terms are ``g * I``: the same gain reaches every correlation, so
    they do not care how ``POLARIZATION.CORR_TYPE`` orders them. ``diagonal``
    and ``full`` terms map correlation index to feed index positionally, so they
    only make sense on a standard linear (XX..YY) or circular (RR..LL) ordering.

    Only terms listed in ``gains.terms`` count -- an unused spec entry is never
    applied and so cannot impose a basis requirement.
    """
    label_to_spec = {s.label: s for s in spec.spec}
    return any(resolve_type(label_to_spec[term], ncorr) != "scalar" for term in spec.terms if term in label_to_spec)


def _build_term_params(spec: TermSpec, nant: int, random_seed: int | None, ncorr: int) -> dict:
    """Build deterministic numpy parameters for a single term."""
    term_type = resolve_type(spec, ncorr)
    nfeed = TERM_FEEDS[term_type]
    rng = np.random.default_rng(_term_seed(random_seed, spec.label))
    params = {
        "label": spec.label,
        "type": term_type,
        "nfeed": nfeed,
        "complex": spec.complex,
        "axes": spec.axes,
        "period": _normalise_period(spec.period, spec.axes),
        "amplitude": float(spec.amplitude),
        # (nant, nfeed): one phase per antenna, or one per antenna per feed.
        "phases": {axis: rng.random((nant, nfeed)) for axis in spec.axes},
    }

    if term_type == "full":
        if spec.complex:
            magnitude = rng.random((nant, 2, 2))
            phase = 2.0 * np.pi * rng.random((nant, 2, 2))
            matrix = magnitude * np.exp(1j * phase)
        else:
            matrix = 2.0 * rng.random((nant, 2, 2)) - 1.0
        params["matrix"] = matrix.astype(np.complex128 if spec.complex else np.float64)

    return params


def _build_all_term_params(spec: CorruptionSpec, nant: int, random_seed: int | None, ncorr: int) -> list[dict]:
    """Build parameters for every term in multiplication order."""
    label_to_spec = {s.label: s for s in spec.spec}
    return [_build_term_params(label_to_spec[label], nant, random_seed, ncorr) for label in spec.terms]


def _compute_oscillation(
    params: dict, time: np.ndarray, freqs: np.ndarray, t0: float, freq0: float, ant: np.ndarray
) -> np.ndarray:
    """Compute a term's scalar oscillation for the antenna observing each row.

    ``t0`` and ``freq0`` are the phase references for the caller's whole
    selection, passed in so every block evaluates the same phase regardless of
    where chunk boundaries fall.

    ``ant`` is the per-row antenna index (``ANTENNA1`` or ``ANTENNA2``), so the
    result is ``(nrow, nchan)``. Evaluating per row rather than building the
    full ``(nant, nrow, nchan)`` cube and indexing it afterwards gives the same
    numbers for ``nant`` times less memory -- only the two indexed slices were
    ever used.

    Returns an array of shape ``(nrow, nchan, nfeed)``, where ``nfeed`` is 1 for
    ``scalar`` and ``full`` terms and 2 for ``diagonal`` ones.
    """
    oscillation = np.ones((time.size, freqs.size, params["nfeed"]), dtype=np.complex128)

    for axis in params["axes"]:
        period = params["period"][axis]
        phases = params["phases"][axis][ant]  # (nrow, nfeed)

        if axis == "time":
            phase = 2.0 * np.pi * ((time[:, None] - t0) / period + phases)  # (nrow, nfeed)
            trig = np.cos(phase) + 1j * np.sin(phase) if params["complex"] else np.cos(phase)
            oscillation *= trig[:, None, :]
        else:
            # (nrow, nchan, nfeed)
            phase = 2.0 * np.pi * ((freqs[None, :, None] - freq0) / period + phases[:, None, :])
            trig = np.cos(phase) + 1j * np.sin(phase) if params["complex"] else np.cos(phase)
            oscillation *= trig

    return oscillation


def _antenna_gain(
    term_params: list[dict], time: np.ndarray, freqs: np.ndarray, t0: float, freq0: float, ant: np.ndarray
) -> np.ndarray:
    """Combined scalar gain for the antenna observing each row: ``(nrow, nchan)``.

    Only valid when every term is ``scalar``; this is the common case and avoids
    carrying a feed axis that would hold two copies of the same number.
    """
    gain = np.ones((time.size, freqs.size), dtype=np.complex128)
    for params in term_params:
        oscillation = _compute_oscillation(params, time, freqs, t0, freq0, ant)
        gain *= 1.0 + params["amplitude"] * oscillation[..., 0]
    return gain


def _antenna_feed_gain(
    term_params: list[dict], time: np.ndarray, freqs: np.ndarray, t0: float, freq0: float, ant: np.ndarray
) -> np.ndarray:
    """Combined per-feed gain for the antenna observing each row: ``(nrow, nchan, 2)``.

    Valid when no term is ``full``, so the combined Jones stays diagonal and only
    its two diagonal entries need carrying. A ``scalar`` term contributes the
    same gain to both feeds.
    """
    gain = np.ones((time.size, freqs.size, 2), dtype=np.complex128)
    for params in term_params:
        oscillation = _compute_oscillation(params, time, freqs, t0, freq0, ant)
        # A scalar term has nfeed == 1 and broadcasts across both feeds.
        gain *= 1.0 + params["amplitude"] * oscillation
    return gain


def _antenna_jones(
    term_params: list[dict], time: np.ndarray, freqs: np.ndarray, t0: float, freq0: float, ant: np.ndarray
) -> np.ndarray:
    """Combined 2x2 Jones for the antenna observing each row: ``(nrow, nchan, 2, 2)``."""
    jones = np.broadcast_to(np.eye(2), (time.size, freqs.size, 2, 2)).astype(np.complex128)
    for params in term_params:
        oscillation = _compute_oscillation(params, time, freqs, t0, freq0, ant)
        amp = params["amplitude"]
        if params["type"] == "scalar":
            jones *= (1.0 + amp * oscillation[..., 0])[:, :, None, None]
        elif params["type"] == "diagonal":
            # Right-multiply, matching the full branch's einsum below, so terms
            # compose left-to-right in the order listed. Scaling *columns* is
            # `jones @ diag(g)`; scaling rows would be `diag(g) @ jones`, which
            # silently reverses a diagonal term that follows a full one.
            jones = jones * (1.0 + amp * oscillation)[:, :, None, :]
        else:
            matrix = params["matrix"][ant][:, None, :, :]  # (nrow, 1, 2, 2)
            term_jones = np.eye(2) + amp * oscillation[..., 0][:, :, None, None] * matrix
            jones = np.einsum("rfij,rfjk->rfik", jones, term_jones)
    return jones


def _corrupt_block(
    vis: np.ndarray,
    time: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    freqs: np.ndarray,
    t0: float,
    freq0: float,
    term_params: list[dict],
    out_dtype: np.dtype,
) -> np.ndarray:
    """Apply RIME corruptions to one (row, chan) chunk.

    ``t0`` and ``freq0`` span the caller's whole selection, and the per-antenna
    phases in ``term_params`` are sized for its whole array: deriving either
    from this block would make the result depend on where the chunk boundaries
    fall.

    Note the references are per *selection*, not per MS -- ``skysim`` calls this
    for one field and one SPW at a time, so separate runs over different fields
    or SPWs of the same MS do not share a time or frequency origin.

    Gains are evaluated per row, never per antenna: an ``(nant, nrow, nchan)``
    cube is ``nant`` times larger than the two slices that get used, which at
    the default chunking runs to tens of GiB on a real array.
    """
    vis = np.asarray(vis)
    time = np.asarray(time)
    antenna1 = np.asarray(antenna1)
    antenna2 = np.asarray(antenna2)
    freqs = np.asarray(freqs)
    ncorr = vis.shape[-1]
    nrow = vis.shape[0]

    types = {p["type"] for p in term_params}

    if "full" in types:
        if ncorr != 4:
            # This should have been caught by validation, but keep a clear message.
            raise RuntimeError("Full Jones corruptions require a 4-correlation MS")

        # Apply J_p V J_q^H per row/chan.
        jp = _antenna_jones(term_params, time, freqs, t0, freq0, antenna1)
        jq = _antenna_jones(term_params, time, freqs, t0, freq0, antenna2)
        jq_h = np.conj(jq.swapaxes(-2, -1))

        vmat = np.empty((nrow, freqs.size, 2, 2), dtype=np.complex128)
        vmat[..., 0, 0] = vis[..., 0]
        vmat[..., 0, 1] = vis[..., 1]
        vmat[..., 1, 0] = vis[..., 2]
        vmat[..., 1, 1] = vis[..., 3]
        vout = np.einsum("rcij,rcjk,rckl->rcil", jp, vmat, jq_h)
        out = np.empty_like(vis)
        out[..., 0] = vout[..., 0, 0]
        out[..., 1] = vout[..., 0, 1]
        out[..., 2] = vout[..., 1, 0]
        out[..., 3] = vout[..., 1, 1]
    elif "diagonal" in types:
        # Combined Jones is diag(g_x, g_y), so V' = diag(g_p) V diag(g_q)^H picks
        # one feed from each antenna per correlation. Carrying the two diagonal
        # entries is enough; the off-diagonal entries stay zero.
        gp = _antenna_feed_gain(term_params, time, freqs, t0, freq0, antenna1)
        gq = _antenna_feed_gain(term_params, time, freqs, t0, freq0, antenna2)
        gq_conj = np.conj(gq)

        out = np.empty_like(vis)
        if ncorr == 4:
            # [XX, XY, YX, YY] <-> feeds [(x,x), (x,y), (y,x), (y,y)].
            for corr, (fp, fq) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                out[..., corr] = vis[..., corr] * gp[..., fp] * gq_conj[..., fq]
        elif ncorr == 2:
            # [XX, YY]: each parallel hand sees its own feed on both antennas.
            for corr in (0, 1):
                out[..., corr] = vis[..., corr] * gp[..., corr] * gq_conj[..., corr]
        else:
            # This should have been caught by validation, but keep a clear message.
            raise RuntimeError("Diagonal Jones corruptions require at least 2 correlations")
    else:
        # Every term is scalar: one gain per antenna, identical on both feeds.
        gp = _antenna_gain(term_params, time, freqs, t0, freq0, antenna1)
        gq = _antenna_gain(term_params, time, freqs, t0, freq0, antenna2)
        out = vis * (gp * np.conj(gq))[:, :, None]

    return out.astype(out_dtype, copy=False)


def apply_corruptions(
    vis: da.Array,
    time: da.Array,
    antenna1: da.Array,
    antenna2: da.Array,
    freqs: np.ndarray,
    spec: CorruptionSpec,
    random_seed: int | None = None,
    nant: int | None = None,
    time_ref: float | None = None,
    freq_ref: float | None = None,
) -> da.Array:
    """Apply RIME corruptions to a lazy visibility array.

    Parameters
    ----------
    vis : dask.array.Array
        Visibility array of shape ``(nrow, nchan, ncorr)``.
    time, antenna1, antenna2 : dask.array.Array
        MS columns of shape ``(nrow,)``.
    freqs : numpy.ndarray
        Channel centre frequencies of shape ``(nchan,)``.
    spec : CorruptionSpec
        Parsed corruption specification.
    random_seed : int or None, optional
        Global seed for reproducible randomness. ``None`` is *not* fresh entropy:
        each term falls back to a deterministic hash of its label, so an
        unseeded run is reproducible (and two same-hash terms share phases).
    nant : int or None, optional
        Number of antennas to size the per-antenna gains for. Defaults to the
        highest index present in ``antenna1``/``antenna2``, which is only the
        antennas in this selection; pass the ``ANTENNA`` table's row count so
        every field of an MS draws the same per-antenna phases.
    time_ref, freq_ref : float or None, optional
        Phase origins for the ``time`` and ``frequency`` axes. Default to the
        earliest time and lowest frequency in this selection, which for a
        multi-field or multi-SPW MS differs between runs; pass the MS-wide
        values so every run shares one origin.

    Returns
    -------
    dask.array.Array
        Corrupted visibilities with the same shape and chunking as ``vis``.
    """
    # Phase origins and array size, fixed across the whole computation so chunk
    # boundaries cannot shift them. The fallbacks describe this selection only --
    # `freqs` is one SPW and `time` one field -- so a caller with the MS in hand
    # should pass the MS-wide values instead.
    if time_ref is None or nant is None:
        needed = [time.min()] if time_ref is None else []
        if nant is None:
            needed += [antenna1.max(), antenna2.max()]
        computed = list(dask.compute(*needed))
        if time_ref is None:
            time_ref = float(computed.pop(0))
        if nant is None:
            nant = max(int(computed[0]), int(computed[1])) + 1

    t0 = float(time_ref)
    freq0 = float(freqs.min()) if freq_ref is None else float(freq_ref)

    ncorr = vis.shape[-1]
    validate_spec(spec, ncorr=ncorr)
    term_params = _build_all_term_params(spec, nant, random_seed, ncorr)

    return da.blockwise(
        _corrupt_block,
        ("row", "chan", "corr"),
        vis,
        ("row", "chan", "corr"),
        time,
        ("row",),
        antenna1,
        ("row",),
        antenna2,
        ("row",),
        freqs,
        ("chan",),
        dtype=vis.dtype,
        t0=t0,
        freq0=freq0,
        term_params=term_params,
        out_dtype=vis.dtype,
        meta=np.empty((0, 0, vis.shape[-1]), dtype=vis.dtype),
    )
