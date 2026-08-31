"""Fast image-domain a-term (primary-beam) correction for gridded prediction.

The FITS-image path predicts ``V = sum_pix I(x) K(u, x)`` with ``ducc0``'s
wgridder. With direction-dependent effects the quantity is instead

    V_pq(t, nu) = sum_pix  [ E_p(t, nu, x) . B(nu, x) . E_q(t, nu, x)^H ]_c  K(u, x, nu)

where ``E_p`` is antenna p's 2x2 voltage beam (the *a-term*) and ``B`` the sky
coherency built from the Stokes planes. The exact answer is a per-source DFT
(``predict_vis_beam``/``predict_vis_jones``), which costs
``O(nrow * nchan * ncomp * ncorr)`` and is hopeless for a filled image
(``ncomp ~ npix``). WSClean's IDG solves this by applying a-terms to gridded
subgrids per time/frequency chunk; DDFacet's facet scheme applies a piecewise-
constant (per-facet) a-term and grids per facet. This module uses the same core
idea -- *apply the a-term in the image domain, amortised over blocks of
visibilities within which it barely changes* -- with one simplification
available to a simulator that only degrids: the a-term is multiplied into the
**full image** (no spatial facet/subgrid approximation), so the *only*
approximation relative to the DFT is interpolation of the a-term in time and
frequency, both with controlled error and both exact in the limit of dense
knots.

Decomposition and cost rationale
--------------------------------
Every optimisation below is either exact or carries an explicit error control:

1. **Baseline classes (exact).** ``E_p`` depends on antenna p only through its
   beam *type* (the ``TELESCOPE_NAME``-keyed provider). Rows are partitioned by
   the ordered type pair ``(type_p, type_q)``; within a class the a-term is
   baseline-independent, so one apparent image serves the whole class. This is
   an exact regrouping -- heterogeneous arrays cost ``n_used_type_pairs`` image
   passes instead of being (wrongly) averaged away.

2. **Time: per-antenna linear interpolation on the PA grid (error O(dt^2),
   same semantics as the DFT kernel).** The beam rotates with parallactic
   angle. The exact-DFT path samples each type's beam on the uniform-in-time
   grid of :func:`simms.skymodel.beams.pa_sample_grid` and interpolates each
   *antenna's* voltage linearly between the bracketing knots. We reproduce
   that contraction identically in the image domain: with per-row weight
   ``w`` toward knot ``k+1``,

       g_p g_q^* = (1-w)^2 G_k G_k^* + w^2 G_{k+1} G_{k+1}^*
                   + w(1-w) (G_k G_{k+1}^* + G_{k+1} G_k^*)

   so a row's visibility is a *quadratic blend of three image products*: the
   two "diagonal" knot images and one "cross" image per time bin (the two
   cross terms share a weight, and prediction is linear in the image, so they
   are summed into a single image). Diagonal-knot passes are shared by the two
   bins adjacent to a knot, so a block spanning ``nbin`` bins costs
   ``nbin + 1`` diagonal + ``nbin`` cross passes per class -- not ``3 nbin``.
   Because the blend is algebraically identical to the DFT kernel's
   interpolation, the two paths agree to gridder accuracy (this is asserted in
   the tests), and the existing ``--beam-pa-step`` knob controls the error of
   both paths in the same way.

3. **Frequency: adaptive knots (error O(dnu^2), tolerance-driven).** The beam
   also scales with frequency. A flat-spectrum image otherwise needs only
   *one* gridder pass for the whole band (the gridder scales ``uvw`` per
   channel), so evaluating the beam per channel would multiply the FFT count
   by ``nchan``. Instead the band is split at knot channels chosen greedily so
   that a linear interpolation of every type's *voltage* beam between adjacent
   knots is within ``freq_tol`` (in voltage units, beam peak ~1) of the true
   beam on a probe grid covering the image. Each channel's visibility is then
   a linear blend of the two bracketing knot passes -- by linearity of the
   prediction this equals predicting with the frequency-interpolated apparent
   image. First-order, a voltage error ``e`` on each of the two antennas gives
   a fractional visibility error ``<= ~2 e`` on that source's contribution.
   Models that already pay a per-channel FFT (spectral cubes, per-pixel
   log-polynomials -- the sky itself changes per channel) interpolate the
   a-term *image* per channel instead, which is the same approximation without
   extra FFTs. Setting ``freq_tol <= 0`` forces a knot at every channel
   (exact).

4. **Per-type voltage maps, cached (exact).** The pixel maps ``G_type(t_knot,
   f_knot)`` -- not per-antenna, not per-class -- are the only beam
   evaluations. They are cached in a byte-capped LRU shared by all row blocks
   (dask's thread scheduler shares the prepared model), keyed time-independent
   for non-ALT-AZ mounts. Class images are cheap pixelwise products of cached
   maps with the Stokes planes.

5. **Negligible-pass skipping (error below the gridder's own floor).** A pass
   whose apparent image peaks below ``epsilon * peak_brightness`` is dropped:
   its contribution is smaller than the accuracy the gridder is already
   allowed to miss on the brightest emission. This is what silently removes
   the imaginary-part passes for real (e.g. cosine-taper) beams and the
   cross-hand passes of unpolarised skies, instead of hard-coding either
   special case. At most a few tens of passes touch one (row, channel) cell,
   so the worst-case accumulated skip error stays ``O(10 epsilon)``.

Complex images cost two real gridder passes (Re and Im) because the wgridder
consumes real dirty images; skipping (5) makes the common real-beam case pay
one.

Total cost per (row-block, channel-segment):
``(n_knots_touched + n_bins) * n_classes * ncorr * {1|2}`` FFT+degrid passes,
each degridding only its own rows/channels, versus one pass with no beam and
``O(nrow * nchan * npix_support * ncorr)`` for the exact DFT. For a 4k-channel,
8h MeerKAT-like observation with a 1-degree PA step and default ``freq_tol``,
that is tens of image FFTs -- orders of magnitude below the DFT.

**That formula holds for a FLAT sky only.** A per-channel model (spectral CUBE,
or a per-pixel log-polynomial) pays those passes *per channel* rather than per
channel-segment, since its image changes with frequency: the frequency-knot
saving in (3) buys nothing there, and only the negligible-pass skip in (5) keeps
empty line channels free. Such a model also gives up the per-channel backend
choice ``_perchan_visibilities`` makes without a beam, where a channel holding a
handful of bright pixels is predicted by DFT instead of gridded. A spectral-line
cube under a beam can therefore cost far more than "tens of image FFTs"; the
planned pass count is logged at DEBUG before the first pass so the bill is
visible up front.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np
from ducc0.wgridder import dirty2vis

from simms import BIN
from simms.skymodel.beams import EVAL_SLAB_PIXELS, corr_feed_maps, pa_sample_grid, reproject_lm
from simms.skymodel.fits_spectrum import SpectralKind, evaluate_scale

log = logging.getLogger(BIN.skysim)

# Probe grid side for frequency-knot selection: the beam is compared against its
# linear-in-frequency interpolation on a probe_side x probe_side grid spanning
# the image. The beam is smooth on the scale of the primary lobe, so a coarse
# spatial probe suffices; the cost is ~probe_side^2 * nchan * ntype evaluations.
FREQ_PROBE_SIDE = 17

# Hard ceiling on frequency knots (beyond this the greedy split degenerates
# toward per-channel knots anyway, which callers get exactly with freq_tol<=0).
MAX_FREQ_KNOTS = 257

# Voltage maps touched by a single prediction pass: two antenna types x two time
# knots x two frequency knots. The cache must hold at least this many or it
# thrashes on every pass.
MAPS_PER_PASS = 8

# Fraction of the map-cache budget lent to the per-segment cache of apparent-beam
# products in the per-channel path (see _predict_segment_perchan). Bounded so the
# hoist cannot reintroduce unbounded transient growth on large images.
PRODUCT_CACHE_FRACTION = 0.25


def aterm_cache_min_gib(npix_l: int, npix_m: int, full_jones: bool) -> float:
    """GiB the voltage-map cache must hold for one prediction pass at this image size.

    Public so callers can pre-flight the ceiling and degrade gracefully (as
    ``skysim`` does, falling back to the PA-averaged power beam) rather than
    meeting :class:`MemoryError` from :func:`attach_fits_aterm` mid-run.
    """
    fold = 4 if full_jones else 2
    return MAPS_PER_PASS * npix_l * npix_m * fold * 8 / 2**30  # complex64 == 8 B


class _MapCache:
    """Thread-safe, byte-capped LRU for per-type voltage pixel maps.

    Values are numpy arrays. A concurrent miss on the same key may compute the
    map twice (the lock guards only the dict), which wastes work but never
    correctness; serialising beam evaluation across dask threads would cost
    more than the rare duplicate.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self._data: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key, value: np.ndarray):
        with self._lock:
            if key in self._data:
                return
            self._data[key] = value
            self._bytes += value.nbytes
            while self._bytes > self.max_bytes and len(self._data) > 1:
                _, old = self._data.popitem(last=False)
                self._bytes -= old.nbytes

    # The cache holds a lock and possibly GiB of maps; neither should travel
    # through a pickle (e.g. a process-based dask scheduler). Rebuild empty.
    def __getstate__(self):
        return {"max_bytes": self.max_bytes}

    def __setstate__(self, state):
        self.__init__(state["max_bytes"])


@dataclass
class ATermModel:
    """Everything needed to apply per-antenna a-terms to a gridded FITS sky.

    Built once by :func:`attach_fits_aterm` and shared by every prediction
    block. ``tgrid``/``chi_grid`` are the PA knot grid (a single knot when no
    mount rotates the beam); ``fknot_chan``/``fknot_freq`` the frequency knots
    as global channel indices and their frequencies.
    """

    ant_type: np.ndarray  # (nant,) type index per antenna
    providers: list  # one BeamProvider per type
    type_is_altaz: np.ndarray  # (ntype,) bool
    tgrid: np.ndarray  # (n_t,) knot times (MS seconds)
    chi_grid: np.ndarray  # (n_t,) parallactic angle at the knots
    fknot_chan: np.ndarray  # (n_f,) ascending global channel indices
    fknot_freq: np.ndarray  # (n_f,) frequencies at the knots (Hz)
    ell: np.ndarray  # (npix,) pixel l, relative to the *pointing* centre
    emm: np.ndarray  # (npix,) pixel m, likewise
    npix_l: int
    npix_m: int
    ncorr: int
    full_jones: bool
    basis_transform: np.ndarray | None  # folded into the Jones maps (full_jones only)
    corr_feed_p: np.ndarray | None  # (ncorr,) feed index maps (diagonal mode)
    corr_feed_q: np.ndarray | None
    peak_brightness: float  # reference for the negligible-pass threshold
    cache: _MapCache = field(repr=False)

    @property
    def ntype(self) -> int:
        return len(self.providers)

    # -- time ---------------------------------------------------------------

    def row_time_bins(self, times: np.ndarray):
        """Per-row PA-grid bin ``k`` and weight ``wt`` toward knot ``k+1``.

        Identical to the indexing in :func:`simms.skymodel.mstools.predict_block`,
        so the two beam paths interpolate in time the same way.
        """
        nrow = times.shape[0]
        n_t = self.tgrid.size
        if n_t < 2:
            return np.zeros(nrow, dtype=np.int64), np.zeros(nrow, dtype=np.float64)
        dt = self.tgrid[1] - self.tgrid[0]
        if dt > 0:
            gpos = np.clip((np.asarray(times, dtype=np.float64) - self.tgrid[0]) / dt, 0.0, n_t - 1)
        else:
            gpos = np.zeros(nrow, dtype=np.float64)
        k = np.clip(np.floor(gpos).astype(np.int64), 0, n_t - 2)
        wt = np.clip(gpos - k, 0.0, 1.0)
        return k, wt

    # -- voltage maps -------------------------------------------------------

    def _eval_map(self, type_idx: int, chi: float, freq: float) -> np.ndarray:
        """Evaluate one type's voltage map over the image pixels, in slabs.

        Returns ``(npix, 2)`` complex64 feed voltages (diagonal mode) or
        ``(npix, 2, 2)`` complex64 Jones with the basis transform folded in
        (full-Jones mode). complex64 for the same reason the DFT path's beam
        grid is: voltages are O(1), single precision is ample, and the maps
        dominate the cache footprint; the image products still accumulate in
        double.
        """
        prov = self.providers[type_idx]
        npix = self.ell.size
        freqs = np.array([freq], dtype=np.float64)
        chis = np.array([chi], dtype=np.float64)
        out = np.empty((npix, 2, 2), dtype=np.complex64) if self.full_jones else np.empty((npix, 2), dtype=np.complex64)
        for start in range(0, npix, EVAL_SLAB_PIXELS):
            sl = slice(start, min(start + EVAL_SLAB_PIXELS, npix))
            if self.full_jones:
                jones = prov.jones(self.ell[sl], self.emm[sl], freqs, chis)[0, :, 0]  # (ns, 2, 2)
                out[sl] = np.einsum("ij,sjk->sik", self.basis_transform, jones)
            else:
                out[sl] = prov.voltage(self.ell[sl], self.emm[sl], freqs, chis)[0, :, 0]  # (ns, 2)
        return out

    def voltage_map(self, type_idx: int, t_idx: int, f_idx: int) -> np.ndarray:
        """Cached voltage map for (type, time knot, frequency knot).

        Non-ALT-AZ mounts do not rotate, so their maps are keyed
        time-independently and computed once per frequency knot.
        """
        altaz = bool(self.type_is_altaz[type_idx])
        key = (type_idx, t_idx if altaz else -1, f_idx)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        chi = float(self.chi_grid[t_idx]) if altaz else 0.0
        value = self._eval_map(type_idx, chi, float(self.fknot_freq[f_idx]))
        self.cache.put(key, value)
        return value


# --------------------------------------------------------------------------- knots


def select_freq_knots(providers, type_is_altaz, chi_grid, ell, emm, freqs, tol) -> np.ndarray:
    """Choose channel indices at which to sample the a-term in frequency.

    Greedy: starting from the first channel, each segment is extended (by
    galloping + bisection) while the worst absolute deviation between the true
    voltage beam and its linear-in-frequency interpolation across the segment
    stays within ``tol``, over a probe grid spanning ``(ell, emm)``, all
    provider types, both feeds, and (for ALT-AZ types) the mid-observation
    parallactic angle -- rotation only re-orients the pattern on the sky, it
    does not change how the pattern deforms with frequency, so one probe angle
    is representative. ``tol <= 0``, or fewer than 3 channels, selects every
    channel (exact).
    """
    nchan = freqs.size
    if nchan <= 2 or tol <= 0:
        return np.arange(nchan, dtype=np.int64)

    # Probe grid over the image's bounding box in (l, m).
    side = FREQ_PROBE_SIDE
    pl = np.linspace(ell.min(), ell.max(), side)
    pm = np.linspace(emm.min(), emm.max(), side)
    pll, pmm = (a.ravel() for a in np.meshgrid(pl, pm, indexing="ij"))

    chi_mid = float(chi_grid[chi_grid.size // 2])
    samples = []
    for ti, prov in enumerate(providers):
        chi = chi_mid if type_is_altaz[ti] else 0.0
        v = prov.voltage(pll, pmm, freqs, np.array([chi]))[0]  # (npts, nchan, 2)
        samples.append(v.reshape(-1, nchan, 2))
    volt = np.concatenate(samples, axis=0).transpose(0, 2, 1).reshape(-1, nchan)  # (npts*ntype*2, nchan)

    def segment_ok(a: int, b: int) -> bool:
        if b - a < 2:
            return True
        w = (freqs[a + 1 : b] - freqs[a]) / (freqs[b] - freqs[a])
        interp = volt[:, a, None] * (1.0 - w) + volt[:, b, None] * w
        return float(np.abs(volt[:, a + 1 : b] - interp).max()) <= tol

    knots = [0]
    a = 0
    while a < nchan - 1:
        # Gallop to an upper bound, then bisect for the furthest valid end.
        step = 1
        b = a + 1
        while b < nchan - 1 and segment_ok(a, min(a + 2 * step, nchan - 1)):
            step *= 2
            b = min(a + step, nchan - 1)
        lo, hi = b, min(a + 2 * step, nchan - 1)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if segment_ok(a, mid):
                lo = mid
            else:
                hi = mid - 1
        knots.append(lo)
        a = lo
        if len(knots) >= MAX_FREQ_KNOTS and a < nchan - 1:
            knots.append(nchan - 1)
            break
    return np.unique(np.array(knots, dtype=np.int64))


# --------------------------------------------------------------------------- attach


def attach_fits_aterm(
    prepared,
    ant_type: np.ndarray,
    providers: list,
    type_is_altaz: np.ndarray,
    ra0: float,
    dec0: float,
    lon: float,
    lat: float,
    t_start: float,
    duration: float,
    pa_step: float,
    freq_tol: float = 1e-3,
    full_jones: bool = False,
    basis_transform: np.ndarray | None = None,
    phase_ra0: float | None = None,
    phase_dec0: float | None = None,
    max_gib: float = 4.0,
):
    """Return a copy of a gridder-backend ``PreparedFitsSky`` with a-terms attached.

    ``ra0``/``dec0`` are the beam (antenna pointing) centre; ``phase_ra0``/
    ``phase_dec0`` the phase centre the image grid is referenced to, so the
    beam is sampled at each pixel's offset from where the dishes point (the
    same convention as the DFT path). ``pa_step`` sizes the PA knot grid via
    :func:`~simms.skymodel.beams.pa_sample_grid`; ``freq_tol`` drives
    :func:`select_freq_knots`; ``max_gib`` caps the voltage-map cache.
    """
    if prepared.backend not in ("fft", "perchan"):
        raise ValueError(f"a-terms attach to a gridder backend, not {prepared.backend!r}")
    if prepared.planes is None:
        # A gridder backend always carries planes; None here is a caller bug, not
        # an empty sky (which is a planes array of zeros and predicts zeros).
        raise ValueError("a-terms need a gridder-backend model with image planes, but 'planes' is None")
    if not prepared.linear_basis:
        # _corr_planes builds the coherency in the linear feed basis, which is what
        # the per-feed/Jones beam maps multiply. A circular-basis model would get
        # silently wrong cross-hands.
        raise ValueError(
            "a-terms require a sky prepared in the linear feed basis (prepare_fits_sky(linear_basis=True)); "
            "for a circular-correlation MS use --beam-jones full, which folds the basis transform into the "
            "beam Jones, or --fits-beam-mode average."
        )

    npix_l, npix_m = prepared.npix_l, prepared.npix_m
    i_pix, j_pix = (
        a.ravel().astype(np.float64) for a in np.meshgrid(np.arange(npix_l), np.arange(npix_m), indexing="ij")
    )
    lmn = prepared.grid.pixel_lmn(i_pix, j_pix)
    ell, emm = lmn[:, 0].copy(), lmn[:, 1].copy()
    if phase_ra0 is not None:
        ell, emm = reproject_lm(ell, emm, phase_ra0, phase_dec0, ra0, dec0)
    # Stored float32: these are two npix arrays living on a long-lived object (1 GiB
    # at 8k^2 in float64). Direction cosines are O(0.1) and the beam varies on degree
    # scales, so float32 (~1e-8 rad here) is far finer than any beam model resolves;
    # the providers upcast to float64 for their own arithmetic anyway.
    ell = np.ascontiguousarray(ell, dtype=np.float32)
    emm = np.ascontiguousarray(emm, dtype=np.float32)

    if np.any(type_is_altaz):
        tgrid, chi_grid = pa_sample_grid(t_start, duration, ra0, dec0, lon, lat, pa_step)
    else:
        # Nothing rotates: the a-term is time-independent and the whole time
        # axis collapses to a single knot (the quadratic blend reduces to one
        # diagonal pass with weight 1).
        tgrid, chi_grid = np.array([t_start]), np.array([0.0])

    freqs = np.asarray(prepared.chan_freqs, dtype=np.float64)
    fknot_chan = select_freq_knots(providers, type_is_altaz, chi_grid, ell, emm, freqs, freq_tol)
    fknot_freq = freqs[fknot_chan]
    log.info(
        "A-term correction: %d type(s), %d PA knot(s), %d frequency knot(s) over %d channel(s).",
        len(providers),
        tgrid.size,
        fknot_chan.size,
        freqs.size,
    )

    max_bytes = int(max_gib * 2**30)
    needed_gib = aterm_cache_min_gib(npix_l, npix_m, full_jones)
    if needed_gib > max_gib:
        raise MemoryError(
            f"A-term voltage-map cache of {max_gib:.2f} GiB cannot hold the {MAPS_PER_PASS} maps "
            f"({needed_gib:.2f} GiB) one prediction pass touches for a {npix_l}x{npix_m} image. "
            f"Raise --beam-grid-max-gib, shrink the image, or use --fits-beam-mode average "
            f"(the PA-averaged power beam, which has no such ceiling)."
        )

    if full_jones:
        if prepared.ncorr != 4:
            raise ValueError("full-Jones a-terms require 4 correlations")
        corr_feed_p = corr_feed_q = None
        if basis_transform is None:
            basis_transform = np.eye(2, dtype=np.complex128)
    else:
        corr_feed_p, corr_feed_q = corr_feed_maps(prepared.ncorr)
        basis_transform = None

    # Bound on |B_c| anywhere: correlations are +/-1, +/-i combinations of the
    # Stokes planes, so the pixelwise sum of |Stokes| bounds every correlation.
    peak = float(np.abs(prepared.planes).sum(axis=0).max())

    aterm = ATermModel(
        ant_type=np.ascontiguousarray(ant_type, dtype=np.int64),
        providers=list(providers),
        type_is_altaz=np.asarray(type_is_altaz, dtype=bool),
        tgrid=tgrid,
        chi_grid=chi_grid,
        fknot_chan=fknot_chan,
        fknot_freq=fknot_freq,
        ell=ell,
        emm=emm,
        npix_l=npix_l,
        npix_m=npix_m,
        ncorr=prepared.ncorr,
        full_jones=full_jones,
        basis_transform=basis_transform,
        corr_feed_p=corr_feed_p,
        corr_feed_q=corr_feed_q,
        peak_brightness=peak,
        cache=_MapCache(max_bytes),
    )
    return replace(prepared, aterm=aterm)


# --------------------------------------------------------------------------- image assembly


def _corr_planes(prepared, chan: int | None):
    """The sky's correlation images ``B_c`` in the linear feed basis.

    ``chan`` indexes the model's channel axis (``None`` for the single
    reference plane of FLAT/POLY). Returns a list of ``ncorr`` complex
    ``(npix_l, npix_m)`` images (4 in full-Jones mode, ordered XX, XY, YX, YY).
    """
    from simms.skymodel.fits_skies import _stokes_getter, stokes_to_correlations

    planes = prepared.planes[..., 0 if chan is None else chan]  # (nstokes, npix_l, npix_m)
    ncorr = 4 if prepared.aterm.full_jones else prepared.ncorr
    return stokes_to_correlations(
        _stokes_getter(planes, prepared.stokes_names), ncorr, prepared.polarisation, linear_basis=True
    )


def _diag_products(at: ATermModel, gp_a, gq_a, gp_b=None, gq_b=None):
    """Yield pixelwise per-correlation beam products for the diagonal (per-feed) model.

    With one map pair: ``G_p[fp(c)] conj(G_q[fq(c)])``. With two (the cross
    term of the quadratic time blend): the symmetrised sum
    ``G_p^a conj(G_q^b) + G_p^b conj(G_q^a)``. Yields ``ncorr`` flat complex128
    arrays one at a time -- each is consumed by one gridder pass, so
    materialising the whole list only multiplies the transient footprint by
    ``ncorr``.
    """
    for c in range(at.ncorr):
        fp, fq = at.corr_feed_p[c], at.corr_feed_q[c]
        if gp_b is None:
            yield gp_a[:, fp].astype(np.complex128) * np.conj(gq_a[:, fq])
        else:
            yield gp_a[:, fp].astype(np.complex128) * np.conj(gq_b[:, fq]) + gp_b[:, fp].astype(
                np.complex128
            ) * np.conj(gq_a[:, fq])


def _jones_products(bmats, ep, eq):
    """Yield per-correlation images of ``E_p B E_q^H`` for pixelwise 2x2 maps.

    ``bmats`` is ``(b00, b01, b10, b11)`` flat coherency images; ``ep``/``eq``
    are ``(npix, 2, 2)`` voltage maps. Yields 4 flat images ordered
    (0,0), (0,1), (1,0), (1,1) -- the MS correlation order once the basis
    transform has been folded into the maps.

    One row of ``m = E_p B`` at a time, and the ``E_q^H`` entries indexed
    individually rather than upcasting the whole ``(npix, 2, 2)`` map: peak is
    ~5 pixel images instead of the 12 a fully-materialised list needs.
    """
    b00, b01, b10, b11 = bmats
    for row in (0, 1):
        e0 = ep[:, row, 0].astype(np.complex128)
        e1 = ep[:, row, 1].astype(np.complex128)
        m0 = e0 * b00 + e1 * b10
        m1 = e0 * b01 + e1 * b11
        del e0, e1
        for col in (0, 1):
            cq0 = np.conj(eq[:, col, 0].astype(np.complex128))
            cq1 = np.conj(eq[:, col, 1].astype(np.complex128))
            yield m0 * cq0 + m1 * cq1


def _beam_products(at: ATermModel, tp: int, tq: int, t_a: int, f_idx: int, bmats, t_b: int | None = None):
    """Yield the ``ncorr`` apparent-beam-product images for one pass, one at a time.

    Diagonal mode ignores ``bmats`` (the sky is multiplied in later, once per
    correlation); full-Jones mode folds the coherency here because the matrix
    product mixes correlations. ``t_b`` selects the symmetrised cross term.
    """
    if at.full_jones:
        ep_a = at.voltage_map(tp, t_a, f_idx)
        eq_a = at.voltage_map(tq, t_a, f_idx)
        if t_b is None:
            yield from _jones_products(bmats, ep_a, eq_a)
            return
        ep_b = at.voltage_map(tp, t_b, f_idx)
        eq_b = at.voltage_map(tq, t_b, f_idx)
        for first, second in zip(_jones_products(bmats, ep_a, eq_b), _jones_products(bmats, ep_b, eq_a), strict=True):
            yield first + second
        return
    gp_a = at.voltage_map(tp, t_a, f_idx)
    gq_a = at.voltage_map(tq, t_a, f_idx)
    if t_b is None:
        yield from _diag_products(at, gp_a, gq_a)
        return
    gp_b = at.voltage_map(tp, t_b, f_idx)
    gq_b = at.voltage_map(tq, t_b, f_idx)
    yield from _diag_products(at, gp_a, gq_a, gp_b, gq_b)


# --------------------------------------------------------------------------- prediction


def _grid_pass(image, uvw, freqs, ducc_kwargs, skip_below):
    """Predict one complex apparent image: up to two real wgridder passes.

    Skips a real/imaginary part whose peak is below ``skip_below`` (see module
    docstring, point 5). Returns ``(nrow, nchan)`` complex128, or ``None`` when
    both parts are negligible.
    """
    vis = None
    for part, factor in ((image.real, 1.0), (image.imag, 1.0j)):
        if np.abs(part).max() <= skip_below:
            continue
        part_vis = dirty2vis(uvw=uvw, freq=freqs, dirty=np.ascontiguousarray(part), **ducc_kwargs)
        part_vis = part_vis * factor if factor != 1.0 else part_vis
        vis = part_vis if vis is None else vis + part_vis
    return vis


def predict_aterm_block(prepared, uvw, times, antenna1, antenna2, epsilon, do_wgridding, nthreads):
    """Predict one block of rows with per-antenna a-terms applied.

    ``prepared`` must carry an :class:`ATermModel` (see
    :func:`attach_fits_aterm`) and, when channel-chunked, the global channel
    indices of this block (``prepared.chan_ids``). Rows are grouped by
    (time-bin, baseline-class); channels by frequency-knot segment; each group
    contributes the quadratic-in-time, linear-in-frequency blend of image
    passes derived in the module docstring.
    """
    at: ATermModel = prepared.aterm
    if times is None or antenna1 is None or antenna2 is None:
        raise ValueError("a-term prediction requires 'times', 'antenna1' and 'antenna2'")

    uvw = np.ascontiguousarray(uvw, dtype=np.float64)
    nrow = uvw.shape[0]
    freqs = np.asarray(prepared.chan_freqs, dtype=np.float64)
    nchan = freqs.size
    ncorr = prepared.ncorr
    vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    if nrow == 0 or nchan == 0 or at.peak_brightness == 0.0:
        return vis

    chan_gids = np.arange(nchan, dtype=np.int64) if prepared.chan_ids is None else np.asarray(prepared.chan_ids)

    ducc_kwargs = dict(
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
        **prepared.grid.ducc_kwargs(prepared.npix_l, prepared.npix_m),
    )
    skip_below = epsilon * at.peak_brightness
    shape = (prepared.npix_l, prepared.npix_m)

    # --- row groups: (time bin, ordered type pair) -> row indices
    kbin, wt = at.row_time_bins(np.asarray(times, dtype=np.float64))
    a1 = np.asarray(antenna1)
    a2 = np.asarray(antenna2)
    tp_row, tq_row = at.ant_type[a1], at.ant_type[a2]
    ntype = at.ntype
    group_key = (kbin * ntype + tp_row) * ntype + tq_row
    order = np.argsort(group_key, kind="stable")
    keys, starts = np.unique(group_key[order], return_index=True)
    bins = {}  # (k, tp, tq) -> row index array
    for key, start, stop in zip(keys, starts, np.append(starts[1:], order.size), strict=True):
        k, rem = divmod(int(key), ntype * ntype)
        tp, tq = divmod(rem, ntype)
        bins[(k, tp, tq)] = order[start:stop]

    single_knot = at.tgrid.size < 2

    # Diagonal-knot passes serve the two adjacent bins; collect each knot's rows
    # and their squared blend weights once.
    #
    # Load-bearing invariant: no row index may repeat within a group's row array,
    # because _accumulate writes through fancy indexing, which drops duplicates
    # instead of summing them. It holds because `bins` partitions the rows, and a
    # given (t_knot, tp, tq) entry merges only bins t_knot-1 and t_knot (a row's
    # bin is unique), which are disjoint. Preserve it in any regrouping.
    diag_groups = {}  # (t_knot, tp, tq) -> (rows, weights)
    for (k, tp, tq), rows in bins.items():
        w = wt[rows]
        for t_knot, w_sq in ((k, (1.0 - w) ** 2), (k + 1, w**2)):
            if single_knot and t_knot > 0:
                continue
            if not np.any(w_sq):
                continue
            prev = diag_groups.get((t_knot, tp, tq))
            if prev is None:
                diag_groups[(t_knot, tp, tq)] = (rows, w_sq)
            else:
                diag_groups[(t_knot, tp, tq)] = (np.concatenate([prev[0], rows]), np.concatenate([prev[1], w_sq]))

    # --- channel segments between frequency knots
    n_f = at.fknot_chan.size
    if n_f < 2:
        seg_of_chan = np.zeros(nchan, dtype=np.int64)
    else:
        seg_of_chan = np.clip(np.searchsorted(at.fknot_chan, chan_gids, side="right") - 1, 0, n_f - 2)

    per_channel_sky = prepared.spectrum.kind is not SpectralKind.FLAT

    # Quote the bill before a long run rather than after: a per-channel sky costs
    # this many passes per channel, not per segment (see the module docstring).
    if log.isEnabledFor(logging.DEBUG):
        ngroup = len(diag_groups) + (0 if single_knot else len(bins))
        nseg = np.unique(seg_of_chan).size
        npass = ngroup * ncorr * (nchan if per_channel_sky else nseg)
        log.debug(
            "A-term block: %d row group(s) x %d corr x %s -> up to %d gridder pass(es) "
            "(each x2 for a complex apparent image, minus negligible-pass skips).",
            ngroup,
            ncorr,
            f"{nchan} channel(s)" if per_channel_sky else f"{nseg} channel segment(s)",
            npass,
        )

    for seg in np.unique(seg_of_chan):
        chs = np.nonzero(seg_of_chan == seg)[0]
        seg_freqs = np.ascontiguousarray(freqs[chs])
        j0 = int(seg)
        j1 = min(j0 + 1, n_f - 1)
        if j1 > j0 and at.fknot_freq[j1] > at.fknot_freq[j0]:
            wf = (seg_freqs - at.fknot_freq[j0]) / (at.fknot_freq[j1] - at.fknot_freq[j0])
        else:
            wf = np.zeros(chs.size)

        if per_channel_sky:
            _predict_segment_perchan(
                prepared,
                at,
                uvw,
                vis,
                chs,
                seg_freqs,
                wf,
                j0,
                j1,
                diag_groups,
                bins,
                wt,
                ducc_kwargs,
                skip_below,
                shape,
            )
        else:
            _predict_segment_flat(
                prepared,
                at,
                uvw,
                vis,
                chs,
                seg_freqs,
                wf,
                j0,
                j1,
                diag_groups,
                bins,
                wt,
                ducc_kwargs,
                skip_below,
                shape,
            )

    return vis


def _knot_products(at, tp, tq, t_a, t_b, f_idx, bmats, cache):
    """Yield one knot's apparent-beam products, memoised in ``cache`` when given.

    ``cache`` is only ever passed in diagonal mode, where the products are
    independent of both the channel and the sky; the entry is stored stacked so
    the LRU can weigh it, and the yielded rows are read-only views into it.
    """
    if cache is None:
        yield from _beam_products(at, tp, tq, t_a, f_idx, bmats, t_b=t_b)
        return
    key = (tp, tq, t_a, -1 if t_b is None else t_b, f_idx)
    hit = cache.get(key)
    if hit is None:
        hit = np.stack(list(_beam_products(at, tp, tq, t_a, f_idx, bmats, t_b=t_b)))
        cache.put(key, hit)
    yield from hit


def _lerped_products(at, tp, tq, t_a, t_b, j0, j1, w, bmats, cache):
    """Yield the frequency-lerped apparent-beam product for each correlation."""
    lo = _knot_products(at, tp, tq, t_a, t_b, j0, bmats, cache)
    if not (j1 > j0 and w > 0):
        yield from lo
        return
    hi = _knot_products(at, tp, tq, t_a, t_b, j1, bmats, cache)
    for lo_c, hi_c in zip(lo, hi, strict=True):
        yield (1.0 - w) * lo_c + w * hi_c


def _accumulate(vis, rows, chs, c, w_row, w_chan, pass_vis):
    # ``+=`` through fancy indexing does NOT accumulate over repeated indices, so
    # this relies on ``rows`` holding each row index at most once. That invariant
    # is established where the groups are built (see predict_aterm_block).
    vis[rows[:, None], chs[None, :], c] += (w_row[:, None] * w_chan[None, :]) * pass_vis


def _predict_segment_flat(
    prepared, at, uvw, vis, chs, seg_freqs, wf, j0, j1, diag_groups, bins, wt, ducc_kwargs, skip_below, shape
):
    """Flat-spectrum sky: one image serves all channels of a pass.

    Channel weights implement the linear-in-frequency blend between the two
    knot images (visibility-domain, exact by linearity of the prediction).
    """
    bmats_flat = [np.ascontiguousarray(b).ravel() for b in _corr_planes(prepared, None)]
    single_knot = at.tgrid.size < 2

    f_edges = [(j0, 1.0 - wf)]
    if j1 > j0:
        f_edges.append((j1, wf))

    def run(rows, w_row, tp, tq, t_a, t_b):
        sub_uvw = np.ascontiguousarray(uvw[rows])
        for f_idx, w_chan in f_edges:
            if not np.any(w_chan):
                continue
            products = _beam_products(at, tp, tq, t_a, f_idx, bmats_flat, t_b=t_b)
            for c, product in enumerate(products):
                image = product if at.full_jones else product * bmats_flat[c]
                image = image.reshape(shape)
                if np.abs(image.real).max() <= skip_below and np.abs(image.imag).max() <= skip_below:
                    continue
                pass_vis = _grid_pass(image, sub_uvw, seg_freqs, ducc_kwargs, skip_below)
                if pass_vis is not None:
                    _accumulate(vis, rows, chs, c, w_row, np.asarray(w_chan, dtype=np.float64), pass_vis)

    for (t_knot, tp, tq), (rows, w_sq) in diag_groups.items():
        run(rows, w_sq, tp, tq, t_knot, None)
    if not single_knot:
        for (k, tp, tq), rows in bins.items():
            w_cross = wt[rows] * (1.0 - wt[rows])
            if not np.any(w_cross):
                continue
            run(rows, w_cross, tp, tq, k, k + 1)


def _predict_segment_perchan(
    prepared, at, uvw, vis, chs, seg_freqs, wf, j0, j1, diag_groups, bins, wt, ducc_kwargs, skip_below, shape
):
    """Per-channel sky (spectral CUBE or per-pixel POLY): one image per channel.

    The sky itself changes per channel, so a per-channel FFT is already the
    price of the model; the a-term is therefore interpolated in the *image*
    domain (identical blend, no extra passes). Channels with no emission are
    skipped outright.
    """
    is_poly = prepared.spectrum.kind is SpectralKind.POLY
    single_knot = at.tgrid.size < 2

    # POLY's reference-plane correlation images are channel-independent -- only the
    # per-pixel spectral scale varies -- so build them once for the whole segment.
    poly_base = [np.ascontiguousarray(b).ravel() for b in _corr_planes(prepared, None)] if is_poly else None

    # In diagonal mode the apparent-beam product depends on neither the channel nor
    # the sky, so one build per (group, knot) serves every channel of the segment
    # instead of being rebuilt per (channel x group). Full Jones folds the sky into
    # the product, so there it genuinely varies per channel and is not cached.
    product_cache = None
    if not at.full_jones:
        budget = int(PRODUCT_CACHE_FRACTION * at.cache.max_bytes)
        if budget >= at.ncorr * at.ell.size * 16:  # complex128
            product_cache = _MapCache(budget)

    # Unit channel weight: each pass here grids exactly one channel, so there is no
    # frequency blend to weight (unlike the flat-spectrum path, which blends knots
    # across the band). Loop-invariant, and `_accumulate` only reads it, so it is
    # built once -- keeping it out of the loop also keeps `run` below from closing
    # over a loop variable.
    one = np.ones(1)

    for local, (ch, freq, w) in enumerate(zip(chs, seg_freqs, wf, strict=True)):
        if is_poly:
            scale = evaluate_scale(prepared.spectrum.coeffs, np.array([freq]), prepared.spectrum.ref_freq)[0].ravel()
            bmats_corrs = [b * scale for b in poly_base]
        else:
            bmats_corrs = [np.ascontiguousarray(b).ravel() for b in _corr_planes(prepared, int(ch))]
        if max(np.abs(b).max() for b in bmats_corrs) <= skip_below:
            continue
        chan_freq = np.ascontiguousarray(seg_freqs[local : local + 1])
        chan_idx = np.array([ch])

        def run(rows, w_row, tp, tq, t_a, t_b, w=w, bmats_corrs=bmats_corrs, chan_freq=chan_freq, chan_idx=chan_idx):
            sub_uvw = np.ascontiguousarray(uvw[rows])
            products = _lerped_products(at, tp, tq, t_a, t_b, j0, j1, w, bmats_corrs, product_cache)
            for c, product in enumerate(products):
                image = product if at.full_jones else product * bmats_corrs[c]
                image = image.reshape(shape)
                if np.abs(image.real).max() <= skip_below and np.abs(image.imag).max() <= skip_below:
                    continue
                pass_vis = _grid_pass(image, sub_uvw, chan_freq, ducc_kwargs, skip_below)
                if pass_vis is not None:
                    _accumulate(vis, rows, chan_idx, c, w_row, one, pass_vis)

        for (t_knot, tp, tq), (rows, w_sq) in diag_groups.items():
            run(rows, w_sq, tp, tq, t_knot, None)
        if not single_knot:
            for (k, tp, tq), rows in bins.items():
                w_cross = wt[rows] * (1.0 - wt[rows])
                if not np.any(w_cross):
                    continue
                run(rows, w_cross, tp, tq, k, k + 1)
