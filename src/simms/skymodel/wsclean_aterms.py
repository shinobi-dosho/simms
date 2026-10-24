"""WSClean IDG diagonal FITS screens, sampled in absolute sky coordinates.

Files contain RA, DEC, MATRIX, ANTENNA, FREQ[, TIME] axes. MATRIX stores
Re(XX), Im(XX), Re(YY), Im(YY). Time is MS/MJD seconds. Screens use nearest
(time, frequency), bilinear sky interpolation and zero padding outside the image.
Only file names and metadata survive preparation: FITS handles are opened per slab.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy import units
from astropy.io import fits
from astropy.wcs import WCS

log = logging.getLogger(__name__)


def nearest_indices(grid, values):
    """Nearest sorted sample, clamped at the edges; ties select the earlier sample."""
    grid = np.asarray(grid)
    values = np.asarray(values)
    hi = np.clip(np.searchsorted(grid, values), 0, len(grid) - 1)
    lo = np.maximum(hi - 1, 0)
    return np.where(np.abs(values - grid[lo]) <= np.abs(grid[hi] - values), lo, hi).astype(np.int64)


def _list(value):
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError("WSClean parset lists must be enclosed in [...].")
    lexer = shlex.shlex(value[1:-1], posix=True)
    lexer.whitespace += ","
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def read_parset(path):
    """Read the documented single diagonal term; resolve images relative to the parset."""
    path = Path(path)
    config = {}
    for raw in path.read_text().splitlines():
        # Strip comments without destroying quotes (needed for image paths with spaces).
        quote = None
        end = len(raw)
        escaped = False
        for i, char in enumerate(raw):
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif quote:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "#":
                end = i
                break
        line = raw[:end].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Invalid WSClean parset line: {raw!r}")
        key, value = (part.strip() for part in line.split("=", 1))
        if key in config:
            raise ValueError(f"Duplicate WSClean parset option {key!r}.")
        config[key] = value
    terms = _list(config.get("aterms", "[]"))
    if len(terms) != 1:
        raise ValueError("WSClean import requires exactly one diagonal a-term.")
    term = terms[0]
    kind = config.get(f"{term}.type", term).strip("\"'")
    if kind != "diagonal":
        raise ValueError(f"Unsupported WSClean a-term type {kind!r}; only diagonal FITS screens are supported.")
    allowed = {"aterms", f"{term}.type", f"{term}.images", f"{term}.window"}
    unknown = config.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported WSClean parset option(s): {', '.join(sorted(unknown))}.")
    if f"{term}.window" in config:
        window = config[f"{term}.window"].strip("\"'").lower()
        if window not in {"tukey", "hann", "raised_hann", "rectangular", "gaussian"}:
            raise ValueError(f"Unsupported WSClean a-term window {window!r}.")
        log.warning(
            "WSClean window=%s is a gridding control; native bilinear sampling does not emulate "
            "IDG window/lowpass resampling.",
            window,
        )
    images = _list(config.get(f"{term}.images", "[]"))
    if not images:
        raise ValueError("The WSClean diagonal term requires a nonempty images list.")
    return [str((path.parent / image).resolve()) for image in images]


def _axis(header, axis, unit=None):
    keys = [f"{key}{axis}" for key in ("CRVAL", "CRPIX", "CDELT")]
    try:
        crval, crpix, cdelt = (float(header[key]) for key in keys)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Screen axis {axis} requires finite CRVAL, CRPIX and nonzero CDELT.") from exc
    if not np.isfinite([crval, crpix, cdelt]).all() or cdelt == 0:
        raise ValueError(f"Screen axis {axis} requires finite metadata and nonzero CDELT.")
    scale = units.Unit(header.get(f"CUNIT{axis}", unit or "")).to(unit) if unit else 1.0
    values = (crval + (np.arange(header[f"NAXIS{axis}"]) + 1 - crpix) * cdelt) * scale
    if not np.isfinite(values).all():
        raise ValueError(f"Screen axis {axis} coordinates must be finite.")
    return values


@dataclass(frozen=True)
class ScreenFile:
    path: str
    times: np.ndarray | None
    freqs: np.ndarray
    shape: tuple
    celestial: WCS


class DiagonalScreens:
    """Validated, lazy collection of compatible diagonal Jones FITS screens."""

    def __init__(self, paths, nant):
        files = []
        reference = None
        for path in paths:
            with fits.open(path, memmap=True) as hdus:
                h = hdus[0].header
                ndim = h.get("NAXIS", 0)
                expected = ["RA", "DEC", "MATRIX", "ANTENNA", "FREQ", "TIME"]
                if ndim not in (5, 6) or any(
                    str(h.get(f"CTYPE{i + 1}", "")).upper().split("-")[0] != expected[i] for i in range(ndim)
                ):
                    raise ValueError(
                        "WSClean screens require axes RA, DEC, MATRIX, ANTENNA, FREQ[, TIME] in that order."
                    )
                shape = tuple(int(h[f"NAXIS{i}"]) for i in range(1, ndim + 1))
                if min(shape) < 1 or shape[2] != 4 or shape[3] not in (1, nant):
                    raise ValueError(
                        f"Screen MATRIX must have 4 planes; ANTENNA must be 1 or the whole MS antenna count ({nant})."
                    )
                if h.get("BITPIX") not in (8, 16, 32, 64, -32, -64):
                    raise ValueError("Screen pixels must be real numeric values.")
                if h.get("BSCALE", 1) != 1 or h.get("BZERO", 0) != 0:
                    raise ValueError("Scaled integer WSClean screens are unsupported; store unscaled real pixels.")
                # Validate each axis before constructing WCS: astropy otherwise repairs some invalid metadata.
                for axis in (3, 4):
                    # WSClean's categorical axes are positional and examples omit CDELT.
                    for key, default in (("CRVAL", 0), ("CRPIX", 1), ("CDELT", 1)):
                        h.setdefault(f"{key}{axis}", default)
                for axis in range(1, ndim + 1):
                    _axis(h, axis)
                if not np.allclose(_axis(h, 3), np.arange(4)) and not np.allclose(_axis(h, 3), np.arange(1, 5)):
                    raise ValueError(
                        "Screen MATRIX coordinates must enumerate the four diagonal real/imaginary planes."
                    )
                for i in range(1, ndim + 1):
                    for j in range(1, ndim + 1):
                        if max(i, j) >= 3 and h.get(f"PC{i}_{j}", float(i == j)) != float(i == j):
                            raise ValueError(
                                "Screen categorical, frequency and time axes must be independent linear axes."
                            )
                        if max(i, j) >= 3 and f"CD{i}_{j}" in h:
                            raise ValueError("Screen frequency/time axes use CRVAL/CRPIX/CDELT, not a CD matrix.")
                ant_coords = _axis(h, 4)
                if not (
                    np.allclose(ant_coords, np.arange(shape[3])) or np.allclose(ant_coords, np.arange(1, shape[3] + 1))
                ):
                    raise ValueError("Screen ANTENNA coordinates must enumerate MS antenna row indices.")
                frame = str(h.get("RADESYS", "ICRS")).strip().upper()
                if frame not in ("ICRS", "FK5") or float(h.get("EQUINOX", 2000)) != 2000:
                    raise ValueError("WSClean screens require ICRS or J2000/FK5 celestial coordinates.")
                if not (h["CTYPE1"].upper().startswith("RA---") and h["CTYPE2"].upper().startswith("DEC--")):
                    raise ValueError("Screen RA/DEC axes require a celestial projection.")
                freqs = _axis(h, 5, "Hz")
                if np.any(freqs <= 0):
                    raise ValueError("Screen frequencies must be positive.")
                times = _axis(h, 6, "s") if ndim == 6 else None
                cel = WCS(h).celestial
                if cel.naxis != 2 or cel.wcs.lng != 0 or cel.wcs.lat != 1:
                    raise ValueError("Screen needs an RA/DEC celestial WCS.")
                signature = (shape[:4], tuple(freqs), cel.to_header().tostring())
                if reference is not None and signature != reference:
                    raise ValueError(
                        "WSClean screen files must have identical spatial WCS, frequency and antenna axes."
                    )
                reference = signature
                # Read only one plane at a time, including integer/scaled FITS data.
                for ti in range(shape[5] if ndim == 6 else 1):
                    for fi in range(shape[4]):
                        for ai in range(shape[3]):
                            data = hdus[0].data[ti, fi, ai] if ndim == 6 else hdus[0].data[fi, ai]
                            if not np.isfinite(data).all():
                                raise ValueError(f"Screen {path!r} contains non-finite pixel data.")
                files.append(ScreenFile(str(path), times, freqs, shape, cel))
        if not files:
            raise ValueError("No WSClean screen files supplied.")
        if any(f.times is None for f in files) and len(files) != 1:
            raise ValueError("Timeless WSClean screens cannot be combined with other files.")
        self.files = files
        self.nant = nant
        self.celestial = files[0].celestial
        self.freq_order = np.argsort(files[0].freqs)
        self.freqs = files[0].freqs[self.freq_order]
        entries = [
            (float(t), fi, ti)
            for fi, f in enumerate(files)
            for ti, t in enumerate(f.times if f.times is not None else [0.0])
        ]
        entries.sort()
        self.times = np.array([e[0] for e in entries])
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("Duplicate WSClean screen times are not allowed.")
        self.entries = [(e[1], e[2]) for e in entries]

    @classmethod
    def from_parset(cls, path, nant):
        return cls(read_parset(path), nant)

    @property
    def nant_screen(self):
        return self.files[0].shape[3]

    def sample(self, time_index, freq_index, antenna, ra, dec):
        """Sample one complex diagonal map at absolute RA/Dec (radians)."""
        file_index, ti = self.entries[time_index]
        f = self.files[file_index]
        ai = 0 if self.nant_screen == 1 else int(antenna)
        fi = self.freq_order[freq_index]
        x, y = self.celestial.all_world2pix(np.degrees(ra), np.degrees(dec), 0)
        shape = np.shape(x)
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        nx, ny = f.shape[:2]
        good = (
            np.isfinite(x) & np.isfinite(y) & (x >= -1e-7) & (y >= -1e-7) & (x <= nx - 1 + 1e-7) & (y <= ny - 1 + 1e-7)
        )
        xx, yy = np.clip(np.where(good, x, 0), 0, nx - 1), np.clip(np.where(good, y, 0), 0, ny - 1)
        x0, y0 = np.floor(xx).astype(int), np.floor(yy).astype(int)
        x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
        wx, wy = xx - x0, yy - y0
        with fits.open(f.path, memmap=True) as hdus:
            data = hdus[0].data[ti, fi, ai] if f.times is not None else hdus[0].data[fi, ai]
            planes = (
                (1 - wx) * (1 - wy) * data[:, y0, x0]
                + wx * (1 - wy) * data[:, y0, x1]
                + (1 - wx) * wy * data[:, y1, x0]
                + wx * wy * data[:, y1, x1]
            )
            result = np.stack((planes[0] + 1j * planes[1], planes[2] + 1j * planes[3]), axis=-1)
        result[~good] = 0
        return result.reshape(*shape, 2).astype(np.complex64)


def attach_component_screens(prepared, screens, ra0, dec0, max_gib):
    from simms.skymodel.beams import corr_feed_maps
    from simms.skymodel.fits_skies import lm_to_radec
    from simms.skymodel.mstools import to_full_corr

    ntime = max(2, len(screens.times))
    dims = (screens.nant_screen, ntime, len(prepared.lmn), len(prepared.freqs), 2)
    needed = np.prod(dims, dtype=np.float64) * 8 / 2**30
    if not np.isfinite(max_gib) or max_gib <= 0 or needed > max_gib:
        raise MemoryError(f"WSClean screen grid requires {needed:.3f} GiB, above --beam-grid-max-gib {max_gib}.")
    ra, dec = lm_to_radec(prepared.lmn[:, 0], prepared.lmn[:, 1], ra0, dec0)
    grid = np.empty(dims, dtype=np.complex64)
    fids = nearest_indices(screens.freqs, prepared.freqs)
    for a in range(screens.nant_screen):
        for t in range(ntime):
            for ch, fi in enumerate(fids):
                grid[a, t, :, ch] = screens.sample(min(t, len(screens.times) - 1), fi, a, ra, dec)
    p, q = corr_feed_maps(prepared.ncorr)
    tgrid = screens.times if len(screens.times) > 1 else np.repeat(screens.times, 2)
    return replace(
        to_full_corr(prepared),
        beam_enabled=True,
        beam_grid=grid,
        tgrid=tgrid,
        beam_time_mode="nearest",
        ant_type=np.zeros(screens.nant, dtype=np.int64)
        if screens.nant_screen == 1
        else np.arange(screens.nant, dtype=np.int64),
        corr_feed_p=p,
        corr_feed_q=q,
    )


@dataclass
class ScreenATerm:
    screens: DiagonalScreens
    ra: np.ndarray
    dec: np.ndarray
    cache: object
    full_jones: bool = False

    def voltage_map(self, antenna, time_index, freq_index):
        key = (0 if self.screens.nant_screen == 1 else antenna, time_index, freq_index)
        hit = self.cache.get(key)
        if hit is None:
            hit = self.screens.sample(time_index, freq_index, antenna, self.ra, self.dec)
            self.cache.put(key, hit)
        return hit


def attach_fits_screens(prepared, screens, ra0, dec0, max_gib):
    from simms.skymodel.aterms import _MapCache
    from simms.skymodel.fits_skies import lm_to_radec

    needed = 2 * prepared.npix_l * prepared.npix_m * 2 * 8 / 2**30
    if not np.isfinite(max_gib) or max_gib <= 0 or needed > max_gib:
        raise MemoryError(f"WSClean screen cache requires {needed:.3f} GiB, above --beam-grid-max-gib {max_gib}.")
    ii, jj = np.meshgrid(np.arange(prepared.npix_l), np.arange(prepared.npix_m), indexing="ij")
    lmn = prepared.grid.pixel_lmn(ii.ravel(), jj.ravel())
    ra, dec = lm_to_radec(lmn[:, 0], lmn[:, 1], ra0, dec0)
    return replace(prepared, aterm=ScreenATerm(screens, ra, dec, _MapCache(int(max_gib * 2**30))))


def predict_screen_block(prepared, uvw, times, antenna1, antenna2, epsilon, do_wgridding, nthreads):
    """Group exact screen selections and predict complex apparent correlation images."""
    from simms.skymodel.aterms import _corr_planes, _grid_pass
    from simms.skymodel.beams import corr_feed_maps
    from simms.skymodel.fits_spectrum import SpectralKind, evaluate_scale

    if times is None or antenna1 is None or antenna2 is None:
        raise ValueError("WSClean screen prediction needs TIME, ANTENNA1 and ANTENNA2.")
    at = prepared.aterm
    tids = nearest_indices(at.screens.times, times)
    fids = nearest_indices(at.screens.freqs, prepared.chan_freqs)
    groups = np.column_stack((tids, antenna1, antenna2))
    unique, inverse = np.unique(groups, axis=0, return_inverse=True)
    vis = np.zeros((len(uvw), len(fids), prepared.ncorr), dtype=np.complex128)
    fp, fq = corr_feed_maps(prepared.ncorr)
    ducc = dict(
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
        **prepared.grid.ducc_kwargs(prepared.npix_l, prepared.npix_m),
    )
    flat = prepared.spectrum.kind is SpectralKind.FLAT
    base = _corr_planes(prepared, None) if prepared.spectrum.kind is not SpectralKind.CUBE else None
    for g, (ti, a1, a2) in enumerate(unique):
        rows = np.flatnonzero(inverse == g)
        for fi in np.unique(fids):
            channels = np.flatnonzero(fids == fi)
            gp, gq = at.voltage_map(a1, ti, fi), at.voltage_map(a2, ti, fi)
            batches = [channels] if flat else [np.array([ch]) for ch in channels]
            for chs in batches:
                brightness = base
                if prepared.spectrum.kind is SpectralKind.CUBE:
                    brightness = _corr_planes(prepared, int(chs[0]))
                elif prepared.spectrum.kind is SpectralKind.POLY:
                    scale = evaluate_scale(
                        prepared.spectrum.coeffs, prepared.chan_freqs[chs], prepared.spectrum.ref_freq
                    )[0]
                    brightness = [b * scale for b in base]
                for c, b in enumerate(brightness):
                    image = b * (gp[:, fp[c]] * np.conj(gq[:, fq[c]])).reshape(prepared.npix_l, prepared.npix_m)
                    value = _grid_pass(
                        image,
                        np.ascontiguousarray(uvw[rows], dtype=np.float64),
                        np.ascontiguousarray(prepared.chan_freqs[chs], dtype=np.float64),
                        ducc,
                        0.0,
                    )
                    if value is not None:
                        vis[rows[:, None], chs[None, :], c] = value
    return vis
