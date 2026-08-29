"""RIME-style visibility corruptions applied after prediction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import astropy.units as u
import dask.array as da
import numpy as np
import yaml

DIMENSION_UNITS = {
    "time": u.s,
    "frequency": u.Hz,
}


@dataclass
class TermSpec:
    """Description of a single Jones corruption term."""

    label: str
    diagonal: bool = True
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
    """Load a corruption specification from a YAML file."""
    with open(path) as fh:
        data = yaml.safe_load(fh)

    gains = data.get("gains", {}) if isinstance(data, dict) else {}
    raw_terms = gains.get("terms", [])
    if isinstance(raw_terms, str):
        terms = [t.strip() for t in raw_terms.replace(",", " ").split() if t.strip()]
    else:
        terms = [str(t).strip() for t in raw_terms]

    specs = [TermSpec(**item) for item in gains.get("spec", [])]
    return CorruptionSpec(terms=terms, spec=specs)


def validate_spec(spec: CorruptionSpec, ncorr: int) -> None:
    """Raise RuntimeError if the specification is inconsistent with the MS."""
    labels = [s.label for s in spec.spec]
    if len(labels) != len(set(labels)):
        raise RuntimeError(f"Corruption spec contains duplicate labels: {labels}")

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

    if any(not s.diagonal for s in spec.spec) and ncorr != 4:
        raise RuntimeError("Non-diagonal (full) Jones corruptions require a 4-correlation MS")


def _build_term_params(spec: TermSpec, nant: int, random_seed: int | None) -> dict:
    """Build deterministic numpy parameters for a single term."""
    rng = np.random.default_rng(_term_seed(random_seed, spec.label))
    params = {
        "label": spec.label,
        "diagonal": spec.diagonal,
        "complex": spec.complex,
        "axes": spec.axes,
        "period": _normalise_period(spec.period, spec.axes),
        "amplitude": float(spec.amplitude),
        "phases": {axis: rng.random(nant) for axis in spec.axes},
    }

    if not spec.diagonal:
        if spec.complex:
            magnitude = rng.random((nant, 2, 2))
            phase = 2.0 * np.pi * rng.random((nant, 2, 2))
            matrix = magnitude * np.exp(1j * phase)
        else:
            matrix = 2.0 * rng.random((nant, 2, 2)) - 1.0
        params["matrix"] = matrix.astype(np.complex128 if spec.complex else np.float64)

    return params


def _build_all_term_params(spec: CorruptionSpec, nant: int, random_seed: int | None) -> list[dict]:
    """Build parameters for every term in multiplication order."""
    label_to_spec = {s.label: s for s in spec.spec}
    return [_build_term_params(label_to_spec[label], nant, random_seed) for label in spec.terms]


def _compute_oscillation(params: dict, time: np.ndarray, freqs: np.ndarray, t0: float) -> np.ndarray:
    """Compute the scalar oscillation for a term.

    Returns an array of shape ``(nant, nrow, nchan)``.
    """
    nant = next(iter(params["phases"].values())).size
    nrow = time.size
    nchan = freqs.size
    oscillation = np.ones((nant, nrow, nchan), dtype=np.complex128)

    reference = {"time": t0, "frequency": float(freqs[0])}
    x_values = {"time": time, "frequency": freqs}

    for axis in params["axes"]:
        period = params["period"][axis]
        phases = params["phases"][axis]  # (nant,)
        x = x_values[axis]
        x0 = reference[axis]

        phase = 2.0 * np.pi * ((x[None, :] - x0) / period + phases[:, None])

        if axis == "time":
            trig = np.cos(phase) + 1j * np.sin(phase) if params["complex"] else np.cos(phase)
            oscillation *= trig[:, :, None]
        else:
            trig = np.cos(phase) + 1j * np.sin(phase) if params["complex"] else np.cos(phase)
            oscillation *= trig[:, None, :]

    return oscillation


def _corrupt_block(
    vis: np.ndarray,
    time: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    freqs: np.ndarray,
    t0: float,
    term_params: list[dict],
    out_dtype: np.dtype,
) -> np.ndarray:
    """Apply RIME corruptions to one (row, chan) chunk."""
    vis = np.asarray(vis)
    time = np.asarray(time)
    antenna1 = np.asarray(antenna1)
    antenna2 = np.asarray(antenna2)
    freqs = np.asarray(freqs)
    ncorr = vis.shape[-1]
    nrow = vis.shape[0]
    rows = np.arange(nrow)

    nant = max(int(antenna1.max()), int(antenna2.max())) + 1

    any_full = any(not p["diagonal"] for p in term_params)

    if any_full:
        # Build combined 2x2 Jones per antenna/time/freq.
        jones = np.stack([np.eye(2) for _ in range(nant * nrow * freqs.size)], axis=0)
        jones = jones.reshape((nant, nrow, freqs.size, 2, 2)).astype(np.complex128)

        for params in term_params:
            oscillation = _compute_oscillation(params, time, freqs, t0)
            amp = params["amplitude"]
            if params["diagonal"]:
                scalar = 1.0 + amp * oscillation  # (nant, nrow, nchan)
                jones *= scalar[:, :, :, None, None]
            else:
                matrix = params["matrix"][:, None, None, :, :]  # (nant, 1, 1, 2, 2)
                term_jones = np.eye(2) + amp * oscillation[:, :, :, None, None] * matrix
                jones = np.einsum("anfij,anfjk->anfik", jones, term_jones)

        # Apply J_p V J_q^H per row/chan.
        jp = jones[antenna1, rows, :, :, :]  # (nrow, nchan, 2, 2)
        jq = jones[antenna2, rows, :, :, :]
        jq_h = np.conj(jq.swapaxes(-2, -1))

        if ncorr == 4:
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
        else:
            # This should have been caught by validation, but keep a clear message.
            raise RuntimeError("Non-diagonal Jones corruptions require a 4-correlation MS")
    else:
        # All terms diagonal: scalar gain per antenna/time/freq.
        combined = np.ones((nant, nrow, freqs.size), dtype=np.complex128)
        for params in term_params:
            oscillation = _compute_oscillation(params, time, freqs, t0)
            combined *= 1.0 + params["amplitude"] * oscillation

        factor = combined[antenna1, rows, :] * np.conj(combined[antenna2, rows, :])
        out = vis * factor[:, :, None]

    return out.astype(out_dtype, copy=False)


def apply_corruptions(
    vis: da.Array,
    time: da.Array,
    antenna1: da.Array,
    antenna2: da.Array,
    freqs: np.ndarray,
    spec: CorruptionSpec,
    random_seed: int | None = None,
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
        Global seed for reproducible randomness.

    Returns
    -------
    dask.array.Array
        Corrupted visibilities with the same shape and chunking as ``vis``.
    """
    t0 = float(time.min().compute())
    nant = max(int(antenna1.max().compute()), int(antenna2.max().compute())) + 1

    validate_spec(spec, ncorr=vis.shape[-1])
    term_params = _build_all_term_params(spec, nant, random_seed)

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
        term_params=term_params,
        out_dtype=vis.dtype,
        meta=np.empty((0, 0, vis.shape[-1]), dtype=vis.dtype),
    )
