from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
from daskms import xds_from_ms, xds_from_table

from simms import BIN
from simms.constants import FWHM_TO_GAUSS_SCALE
from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.skymodel.kernels import (
    NO_BEAM_GRAD,
    NO_BEAM_GRAD_JONES,
    NO_POINTING,
    NO_POINTING_JONES,
    NO_SMEAR_UVW,
    is_uniform_grid,
    predict_vis,
    predict_vis_beam,
    predict_vis_jones,
)
from simms.skymodel.smearing import Smearing
from simms.utilities import radec2lm

if TYPE_CHECKING:
    from simms.skymodel.corruptions import PointingModel

log = logging.getLogger(BIN.skysim)

# Largest relative error of the pointing Taylor model, against the exact offset
# beam at the worst-case offset, before attach_beam warns.
POINTING_TAYLOR_WARN = 1e-2

DEFAULT_ROW_CHUNK_CAP = 10000
"""Default upper bound on rows per chunk (the ``--row-chunks`` default)."""

ROW_TASKS_PER_WORKER = 4
"""Row chunks aimed for per worker, so the pool stays fed and load stays even."""

MIN_ROW_CHUNK = 256
"""Never chunk finer than this; below it dask's per-task overhead starts to dominate."""


def auto_row_chunks(
    nrows: int,
    nworkers: int,
    cap: int = DEFAULT_ROW_CHUNK_CAP,
    tasks_per_worker: int = ROW_TASKS_PER_WORKER,
    min_chunk: int = MIN_ROW_CHUNK,
) -> int:
    """Rows per chunk, sized so that every worker has several chunks to work on.

    A *fixed* chunk size makes the task count depend only on the length of the MS,
    so a short observation yields fewer chunks than there are workers and most of
    them sit idle: 76608 rows (a 5-minute MeerKAT track) at the 10000-row default
    is 8 chunks, which pins a ``--nworkers 32`` run to ~8 cores however many are
    asked for.

    `cap` (the ``--row-chunks`` value) is therefore treated as an *upper* bound:
    the size is reduced until each worker has roughly `tasks_per_worker` chunks,
    but never below `min_chunk` rows, so a small MS is not shattered into tasks
    whose overhead outweighs the work they do. The chunk size is never increased
    beyond `cap`, so this can only add parallelism, never memory per task.

    Parameters
    ----------
    nrows : int
        Number of rows the prediction will run over.
    nworkers : int
        Worker count the graph will be scheduled on (``--nworkers``).
    cap : int, optional
        Upper bound on rows per chunk.
    tasks_per_worker : int, optional
        Chunks aimed for per worker.
    min_chunk : int, optional
        Lower bound on rows per chunk.

    Returns
    -------
    int
        Rows per chunk to hand to ``xds_from_ms``.
    """
    if cap <= 0:
        cap = DEFAULT_ROW_CHUNK_CAP
    if nrows <= 0 or nworkers <= 1:
        return cap
    target = -(-nrows // (tasks_per_worker * nworkers))  # ceil division
    return max(min_chunk, min(cap, target))


def vis_noise_from_sefd_and_ms(ms: str, sefd: float, spw_id: int = 0, field_id: int = 0):
    """
    Compute per-visibility thermal noise from an SEFD and an MS.

    Parameters
    ----------
    ms : MS or str
        Measurement Set path.
    sefd : float
        Antenna System Equivalent Flux Density in Jy.
    spw_id : int, optional
        DATA_DESC_ID (spectral window) to use. Default is 0.
    field_id : int, optional
        FIELD_ID to filter rows. Default is 0.

    Returns
    -------
    float
        RMS noise per visibility (Jy).
    """
    spw_ds = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    msds = xds_from_ms(ms, group_cols=["DATA_DESC_ID"], taql_where=f"FIELD_ID=={field_id}")[spw_id]

    df = spw_ds.CHAN_WIDTH.data[spw_id][0]
    dt = msds.EXPOSURE.data[0]
    # Reduce to a plain float here. A lazy dask scalar would otherwise be carried
    # into every predict task, where evaluating it re-opens the MS.
    noise_vis = float((sefd / np.sqrt(2 * dt * df)).compute())
    return noise_vis


def sim_noise(dshape: list | tuple, vis_noise: float, dtype: np.dtype = np.complex128, seed=None) -> np.ndarray:
    """
    Simulate complex Gaussian visibility noise.

    Each of the real and imaginary parts has RMS ``vis_noise``.

    Parameters
    ----------
    dshape : list or tuple
        Desired output shape (e.g., (nrows, nchan, ncorr)).
    vis_noise : float
        RMS per visibility (Jy).
    dtype : numpy.dtype, optional
        Complex dtype of the output. Default complex128.
    seed : int, numpy.random.SeedSequence, numpy.random.Generator or None, optional
        Anything :func:`numpy.random.default_rng` accepts. ``None`` (the default) draws
        fresh entropy, so the result is **not** reproducible.

        Calling this once per block with the *same integer* seed gives every block an
        identical noise realisation, which is not what block-wise noise should look like.
        To get noise that is both reproducible and independent across blocks, build one
        ``numpy.random.default_rng(seed)`` and pass that same Generator to every call:
        ``default_rng`` returns a Generator unchanged, so its state advances from block
        to block.

    Returns
    -------
    numpy.ndarray
        Complex noise array of shape ``dshape``.
    """
    rng = np.random.default_rng(seed)
    real_dtype = np.finfo(dtype).dtype
    # Draw the real and imaginary parts as one array and reinterpret it as
    # complex, so the noise costs a single allocation of the output size.
    noise = rng.standard_normal((*dshape, 2), dtype=real_dtype)
    noise *= vis_noise
    return noise.view(dtype).reshape(dshape)


def noise_visibilities(shape, chunks, vis_noise: float, dtype: np.dtype, seed=None):
    """
    A lazy array of complex Gaussian visibility noise.

    Reproducible for a given ``seed`` **at a given chunking**. ``dask.array.random``
    spawns an independent bit generator per chunk, keyed to that chunk's position in the
    grid, so rechunking ``shape`` re-keys every block and changes the whole realisation --
    it is not merely offset. Pin `chunks` alongside `seed` if you need to reproduce a run.

    Each of the real and imaginary parts has RMS ``vis_noise``.

    Parameters
    ----------
    shape : tuple
        Output shape, ``(nrow, nchan, ncorr)``.
    chunks : tuple
        Dask chunking for `shape`.
    vis_noise : float
        RMS per visibility (Jy).
    dtype : numpy.dtype
        Complex output dtype.
    seed : int or None
        Base seed. ``None`` draws fresh entropy (not reproducible).

    Returns
    -------
    dask.array.Array
    """
    rng = da.random.default_rng(seed)
    real = rng.standard_normal(shape, chunks=chunks)
    imag = rng.standard_normal(shape, chunks=chunks)
    return ((real + 1j * imag) * vis_noise).astype(dtype)


def add_noise(vis: np.ndarray, vis_noise: float, seed=None):
    """
    Add complex Gaussian noise to visibilities in place.

    Parameters
    ----------
    vis : numpy.ndarray
        Visibility data.
    vis_noise : float
        RMS per visibility (Jy).
    seed : optional
        Passed to :func:`sim_noise`; see the reproducibility note there. ``None`` (the
        default) draws fresh entropy.

    Returns
    -------
    numpy.ndarray
        `vis`, with noise added.
    """
    vis += sim_noise(vis.shape, vis_noise, dtype=vis.dtype, seed=seed)
    return vis


def stack_unpolarised_vis(vis: np.ndarray, ncorr: int) -> np.ndarray:
    """
    Replicate unpolarised visibilities across correlation dimension.

    Parameters
    ----------
    vis : numpy.ndarray
        Array of shape (nrows, nchan) with Stokes I or single correlation.
    ncorr : int
        Number of output correlations (2 or 4).

    Returns
    -------
    numpy.ndarray
        Stacked array of shape (nrows, nchan, ncorr).

    Raises
    ------
    ValueError
        If `ncorr` is not 2 or 4.
    """
    if ncorr == 2:
        vis = np.stack([vis, vis], axis=2)
    elif ncorr == 4:
        vis = np.stack([vis, np.zeros_like(vis), np.zeros_like(vis), vis], axis=2)
    else:
        raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
    return vis


@dataclass
class PreparedSky:
    """An ASCII sky model reduced to flat arrays ready for :func:`predict_vis`.

    Built once per simulation rather than once per row block, and shared by all
    blocks. Its memory footprint is dominated by ``bmat``, which is
    ``nsrc * nspec * nchan`` complex values.
    """

    lmn: np.ndarray
    gauss_shape: np.ndarray
    is_gauss: np.ndarray
    bmat: np.ndarray
    lightcurve: np.ndarray
    unique_times: np.ndarray | None
    freqs: np.ndarray
    uniform_freqs: bool
    ncorr: int
    polarisation: bool
    # Primary-beam fields, all None/False unless a beam is attached (see attach_beam).
    beam_enabled: bool = False
    beam_full_jones: bool = False  # True -> beam_grid is (...,2,2) and predict_vis_jones is used
    ant_type: np.ndarray | None = None
    beam_grid: np.ndarray | None = None  # (ntype, n_pa, nsrc, nchan, 2[, 2]) complex
    tgrid: np.ndarray | None = None  # (n_pa,) PA-grid sample times (MS seconds)
    corr_feed_p: np.ndarray | None = None
    corr_feed_q: np.ndarray | None = None
    # Time/bandwidth smearing, applied when set (see attach_smearing).
    smearing: Smearing | None = None
    # Antenna pointing errors, applied in the beam kernels when set (see attach_beam).
    # The derivative grids are shaped like beam_grid; beam_lap only at pointing_order 2.
    pointing: PointingModel | None = None
    pointing_order: int = 0
    beam_dl: np.ndarray | None = None
    beam_dm: np.ndarray | None = None
    beam_lap: np.ndarray | None = None

    @property
    def nspec(self) -> int:
        """Number of correlations actually carried through the kernel."""
        return self.bmat.shape[1]

    def select_channels(self, chan_ids: np.ndarray) -> PreparedSky:
        """Restrict the model to a subset of channels, for channel-chunked prediction."""
        freqs = self.freqs[chan_ids]

        # Advanced-index the chan axis (3); trailing feed/Jones axes are kept as-is, so this
        # works for both the diagonal (...,2) and full-Jones (...,2,2) grids.
        beam_grid = self.beam_grid[:, :, :, chan_ids] if self.beam_enabled else self.beam_grid

        # np.take, unlike the indexing above, returns C-contiguous grids, matching the
        # kernels' C-contiguous placeholders so each kernel keeps a single signature.
        def chans(grid):
            return None if grid is None else np.take(grid, chan_ids, axis=3)

        return replace(
            self,
            freqs=freqs,
            bmat=self.bmat[:, :, chan_ids],
            uniform_freqs=is_uniform_grid(freqs),
            beam_grid=beam_grid,
            beam_dl=chans(self.beam_dl),
            beam_dm=chans(self.beam_dm),
            beam_lap=chans(self.beam_lap),
        )


def prepare_skymodel(
    skymodel: ASCIISkymodel,
    freqs: np.ndarray,
    ra0: float,
    dec0: float,
    ncorr: int = 2,
    polarisation: bool = False,
    linear_basis: bool = True,
    unique_times: np.ndarray = None,
    dtype: np.dtype = np.complex128,
) -> PreparedSky:
    """
    Flatten an ASCII sky model into the arrays the prediction kernel consumes.

    Parameters
    ----------
    skymodel : ASCIISkymodel
        Parsed sky model object.
    freqs : numpy.ndarray
        Channel centre frequencies (Hz).
    ra0, dec0 : float
        Phase centre (radians).
    ncorr : int, optional
        Number of correlations (2 or 4). Default 2.
    polarisation : bool, optional
        If True, carry every correlation. If False, only Stokes I is predicted
        and the remaining correlations are filled in afterwards. Default False.
    linear_basis : bool, optional
        Use linear (True) or circular (False) basis. Default True.
    unique_times : numpy.ndarray, optional
        Sorted unique time stamps spanning the *whole* observation. Required if
        the model contains transient sources: the lightcurve is referenced to
        the start of the observation, not to the start of a row block.
    dtype : numpy.dtype, optional
        Complex dtype of the brightness matrix (and hence of the visibilities).

    Returns
    -------
    PreparedSky

    Raises
    ------
    ValueError
        If transient sources are present and `unique_times` is None, or if
        `ncorr` is not 2 or 4.
    """
    if ncorr not in (2, 4):
        raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")

    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    has_transient = skymodel.has_transient
    if has_transient and unique_times is None:
        raise ValueError("parameter 'unique_times' must be provided for skymodels with transient sources")

    sources = skymodel.sources
    nsrc = len(sources)
    nchan = freqs.size
    # Unpolarised runs only ever need Stokes I; the other correlations are
    # derived from it once, after the sources have been summed.
    nspec = ncorr if polarisation else 1
    ntime = unique_times.size if has_transient else 1

    lmn = np.zeros((nsrc, 3), dtype=np.float64)
    gauss_shape = np.zeros((nsrc, 3), dtype=np.float64)
    is_gauss = np.zeros(nsrc, dtype=np.bool_)
    bmat = np.zeros((nsrc, nspec, nchan), dtype=dtype)
    lightcurve = np.ones((nsrc, ntime), dtype=np.float64)

    for i, source in enumerate(sources):
        el, em = radec2lm(ra0, dec0, source.ra, source.dec)
        lmn[i] = el, em, np.sqrt(1 - el * el - em * em) - 1

        emaj = source.value_or_default("emaj")
        emin = source.value_or_default("emin")
        if emaj or emin:
            pa = source.value_or_default("pa")
            is_gauss[i] = True
            # emaj, emin are FWHM angles (radians); scale to the kernel's shape.
            axis_major = emaj * FWHM_TO_GAUSS_SCALE
            axis_minor = emin * FWHM_TO_GAUSS_SCALE
            gauss_shape[i] = (
                axis_major * np.sin(pa),
                axis_major * np.cos(pa),
                axis_minor / (1.0 if axis_major == 0.0 else axis_major),
            )

        bmat[i] = source.get_brightness_matrix(freqs, ncorr, linear_basis=linear_basis)[:nspec]

        if source.is_transient:
            lightcurve[i] = source.get_lightcurve(unique_times)

    return PreparedSky(
        lmn=lmn,
        gauss_shape=gauss_shape,
        is_gauss=is_gauss,
        bmat=bmat,
        lightcurve=lightcurve,
        unique_times=unique_times if has_transient else None,
        freqs=freqs,
        uniform_freqs=is_uniform_grid(freqs),
        ncorr=ncorr,
        polarisation=polarisation,
    )


def to_full_corr(prepared: PreparedSky) -> PreparedSky:
    """Expand a Stokes-I-only model (``nspec == 1``) to the full ``ncorr`` width.

    The primary-beam kernel applies a per-feed voltage to every correlation, so it needs
    the parallel hands carried explicitly (cross-hands zero for an unpolarised source).
    No-op when the model already carries all correlations.
    """
    if prepared.nspec == prepared.ncorr:
        return prepared
    ncorr = prepared.ncorr
    nsrc, _, nchan = prepared.bmat.shape
    full = np.zeros((nsrc, ncorr, nchan), dtype=prepared.bmat.dtype)
    stokes_i = prepared.bmat[:, 0, :]
    # Parallel hands = Stokes I; cross-hands stay zero. (Linear basis; beams refuse circular.)
    full[:, 0, :] = stokes_i
    full[:, -1, :] = stokes_i
    return replace(prepared, bmat=full)


def attach_beam(
    prepared: PreparedSky,
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
    ncorr: int,
    full_jones: bool = False,
    basis_transform: np.ndarray | None = None,
    phase_ra0: float | None = None,
    phase_dec0: float | None = None,
    beam_grid_max_gib: float | None = None,
    pointing: PointingModel | None = None,
    pointing_step: float | None = None,
) -> PreparedSky:
    """Return a copy of ``prepared`` with a primary-beam grid attached.

    Samples each type's beam on a parallactic-angle grid spanning the observation
    (built once, sliced per channel-chunk by :meth:`PreparedSky.select_channels`).
    ``ra0``/``dec0`` are the beam (antenna pointing) centre; ``phase_ra0``/``phase_dec0``
    are the phase centre the source ``l/m`` were prepared for, so the beam is sampled at each
    source's offset from where the dish points. ``prepared`` must carry the full-width
    brightness (``nspec == ncorr``). With ``full_jones`` the grid holds 2x2 Jones (folding
    ``basis_transform``) and the ``predict_vis_jones`` kernel is used; otherwise the diagonal
    per-feed grid.

    With a ``pointing`` model the beam derivatives are sampled on the same grid, and the
    kernels perturb each antenna's beam by its per-row pointing offset (see
    :mod:`simms.skymodel.kernels`): 2 extra grids for ``taylor: first``, 3 for
    ``laplacian``, all within ``beam_grid_max_gib``. ``pointing_step`` is the
    finite-difference step (radians), defaulting to
    :data:`~simms.skymodel.beams.POINTING_FD_STEP`.
    """
    from simms.skymodel.beams import (
        BEAM_GRID_MAX_GIB_DEFAULT,
        POINTING_FD_STEP,
        _beam_grid_gib,
        _check_beam_grid_footprint,
        build_beam_derivative_grids,
        build_beam_derivative_grids_jones,
        build_beam_grid,
        build_beam_grid_jones,
        corr_feed_maps,
        pa_sample_grid,
        reproject_lm,
    )

    max_gib = BEAM_GRID_MAX_GIB_DEFAULT if beam_grid_max_gib is None else beam_grid_max_gib
    tgrid, chi_grid = pa_sample_grid(t_start, duration, ra0, dec0, lon, lat, pa_step)
    ell, emm = prepared.lmn[:, 0], prepared.lmn[:, 1]
    if phase_ra0 is not None:
        ell, emm = reproject_lm(ell, emm, phase_ra0, phase_dec0, ra0, dec0)

    fold = 4 if full_jones else 2
    dims = (len(providers), chi_grid.size, ell.size, prepared.freqs.size)
    order = 0
    build_gib = max_gib
    if pointing is not None:
        order = _pointing_order(pointing, providers)
        if pointing.nant < np.size(ant_type):
            raise ValueError(
                f"The pointing model covers {pointing.nant} antennas but the beam has {np.size(ant_type)}."
            )
        # The whole set -- E plus its 2 or 3 derivative grids -- has to fit, so check it
        # once, before allocating any of it. The builders would each re-check their own
        # share against the same ceiling, repeating the warning with smaller numbers.
        _check_beam_grid_footprint(*dims, fold, max_gib, pointing_grids=order + 1)
        build_gib = np.inf

    if full_jones:
        beam_grid = build_beam_grid_jones(
            providers, type_is_altaz, ell, emm, prepared.freqs, chi_grid, basis_transform, max_gib=build_gib
        )
        corr_feed_p = corr_feed_q = None
    else:
        beam_grid = build_beam_grid(providers, type_is_altaz, ell, emm, prepared.freqs, chi_grid, max_gib=build_gib)
        corr_feed_p, corr_feed_q = corr_feed_maps(ncorr)

    pointing_fields = {}
    if pointing is not None:
        step = POINTING_FD_STEP if pointing_step is None else pointing_step
        laplacian = order == 2
        if full_jones:
            beam_dl, beam_dm, beam_lap = build_beam_derivative_grids_jones(
                providers,
                type_is_altaz,
                ell,
                emm,
                prepared.freqs,
                chi_grid,
                basis_transform,
                laplacian,
                step=step,
                max_gib=build_gib,
            )
        else:
            beam_dl, beam_dm, beam_lap = build_beam_derivative_grids(
                providers, type_is_altaz, ell, emm, prepared.freqs, chi_grid, laplacian, step=step, max_gib=build_gib
            )
        pointing_fields = dict(
            pointing=pointing, pointing_order=order, beam_dl=beam_dl, beam_dm=beam_dm, beam_lap=beam_lap
        )
        _log_pointing(
            pointing,
            order,
            providers,
            type_is_altaz,
            ell,
            emm,
            prepared.freqs,
            chi_grid,
            step,
            _beam_grid_gib(*dims, fold) * (order + 1),
        )
        _warn_taylor_error(
            pointing,
            order,
            providers,
            type_is_altaz,
            ell,
            emm,
            prepared.freqs,
            chi_grid,
            basis_transform if full_jones else None,
            beam_grid,
            beam_dl,
            beam_dm,
            beam_lap,
        )

    return replace(
        prepared,
        beam_enabled=True,
        beam_full_jones=full_jones,
        ant_type=np.ascontiguousarray(ant_type, dtype=np.int64),
        beam_grid=beam_grid,
        tgrid=tgrid,
        corr_feed_p=corr_feed_p,
        corr_feed_q=corr_feed_q,
        **pointing_fields,
    )


def _pointing_order(pointing: PointingModel, providers: list) -> int:
    """The Taylor order a pointing model runs at on these beam types.

    ``laplacian`` needs a smooth pattern; a bilinear FITS cube has no meaningful
    curvature, so those types get first-order pointing (a zero ``L`` slab), and a run
    where no type has curvature is first order outright.
    """
    from simms.skymodel.corruptions import TAYLOR_ORDERS

    requested = TAYLOR_ORDERS[pointing.taylor]
    curved = [bool(getattr(p, "smooth_curvature", True)) for p in providers]
    if requested == 2 and not all(curved):
        flat = [_provider_label(p, i) for i, (p, c) in enumerate(zip(providers, curved, strict=True)) if not c]
        everywhere = not any(curved)
        log.warning(
            "Pointing errors with taylor: laplacian, but beam type(s) %s are bilinear FITS cubes whose "
            "Laplacian is undefined; those types use first-order pointing (no mean loss)%s.",
            ", ".join(flat),
            " -- with no smooth type left, this run is effectively taylor: first" if everywhere else "",
        )
    return 2 if requested == 2 and any(curved) else 1


def _provider_label(provider, index: int) -> str:
    """A short name for a beam type in log messages."""
    name = getattr(provider, "name", "") or getattr(getattr(provider, "beam", None), "name", "")
    return f"{index} ({name or type(provider).__name__})"


def _log_pointing(pointing, order, providers, type_is_altaz, ell, emm, freqs, chi_grid, step, added_gib) -> None:
    """INFO summary of the pointing model, with the predicted mean loss in laplacian mode."""
    from simms.skymodel.beams import UnityBeamProvider, _pointing_derivatives

    sigma = pointing.sigma_eff
    mode = "laplacian" if order == 2 else "first"
    log.info(
        "Pointing errors: sigma_eff %.2f arcsec per axis, taylor: %s, %d antennas; the derivative grids add %.3f GiB.",
        np.degrees(sigma) * 3600.0,
        mode,
        pointing.nant,
        added_gib,
    )
    if order == 2 and ell.size:
        # The source nearest the beam centre, the first smooth type's feed-0 voltage (a
        # FITS-cube type has L = 0, which would report no loss), the middle PA sample and
        # channel. Re-derived here (5 evaluations) rather than read off the grid, which a
        # full-Jones run holds in the MS correlation basis. Order 2 implies one exists.
        ti = next(i for i, p in enumerate(providers) if getattr(p, "smooth_curvature", True))
        s = int(np.argmin(ell * ell + emm * emm))
        k, f = chi_grid.size // 2, freqs.size // 2
        chi = chi_grid[k : k + 1] if type_is_altaz[ti] else np.zeros(1)
        _, _, lap = _pointing_derivatives(
            providers[ti], ell[s : s + 1], emm[s : s + 1], freqs[f : f + 1], chi, True, step, False
        )
        log.info(
            "Predicted mean voltage change 0.5*sigma_eff^2*lap(E) = %.3g (beam type %s, feed 0, source %d at "
            "%.3f deg from the pointing centre, %.4g MHz).",
            0.5 * sigma * sigma * lap[0, 0, 0].real,
            _provider_label(providers[ti], ti),
            s,
            np.degrees(np.hypot(ell[s], emm[s])),
            freqs[f] / 1e6,
        )
    if all(isinstance(p, UnityBeamProvider) for p in providers):
        log.warning("Pointing errors are set but every antenna has a unity beam, so they change nothing.")


def _warn_taylor_error(
    pointing,
    order,
    providers,
    type_is_altaz,
    ell,
    emm,
    freqs,
    chi_grid,
    transform,
    beam_grid,
    beam_dl,
    beam_dm,
    beam_lap,
) -> None:
    """Warn when the Taylor model misses the exact offset beam at the largest offset drawn.

    Probes the middle PA sample at ``+-d_max`` along each feed axis, with ``d_max`` the
    largest per-antenna ``|static| + amplitude`` offset; beyond ~1% of the beam peak the
    offsets are too large a fraction of the beam for a Taylor expansion.
    """
    d_max = float(np.max(np.hypot(*(np.abs(pointing.static) + pointing.amplitude).T))) if pointing.nant else 0.0
    if d_max == 0.0:
        return
    k = chi_grid.size // 2
    worst = 0.0
    for ti, prov in enumerate(providers):
        chi = chi_grid[k : k + 1] if type_is_altaz[ti] else np.zeros(1)
        e0 = beam_grid[ti, k].astype(np.complex128)
        peak = float(np.abs(e0).max())
        if peak == 0.0:
            continue
        for dl, dm in ((d_max, 0.0), (-d_max, 0.0), (0.0, d_max), (0.0, -d_max)):
            if transform is None:
                exact = prov.voltage(ell, emm, freqs, chi, offset=(dl, dm))[0]
            else:
                exact = np.einsum("ij,sfjk->sfik", transform, prov.jones(ell, emm, freqs, chi, offset=(dl, dm))[0])
            model = e0 + dl * beam_dl[ti, k] + dm * beam_dm[ti, k]
            if order == 2:
                model = model + 0.25 * (dl * dl + dm * dm) * beam_lap[ti, k]
            worst = max(worst, float(np.abs(exact - model).max()) / peak)
    if worst > POINTING_TAYLOR_WARN:
        log.warning(
            "Pointing-error Taylor model is off by up to %.1f%% of the beam peak at the largest offset drawn "
            "(%.1f arcsec): the offsets are too large a fraction of the beam for a %s expansion.",
            100.0 * worst,
            np.degrees(d_max) * 3600.0,
            "second-order" if order == 2 else "first-order",
        )


def attach_smearing(prepared, smearing: Smearing | None):
    """Return a copy of ``prepared`` whose prediction is time/bandwidth smeared.

    Works for any prepared sky the DFT kernels consume -- :class:`PreparedSky` and
    the DFT-backend :class:`~simms.skymodel.fits_skies.PreparedFitsSky` alike.
    Predicting a smeared model then needs the per-row ``EXPOSURE`` alongside the
    ``UVW``; see :func:`predict_block`. ``None`` is a no-op, so callers can pass
    the option through unconditionally.
    """
    return prepared if smearing is None else replace(prepared, smearing=smearing)


def smear_kernel_args(smearing: Smearing | None, uvw: np.ndarray, exposure) -> tuple:
    """The ``(smear, bw_half, smear_uvw)`` trailing arguments of the DFT kernels.

    Raises
    ------
    ValueError
        If ``smearing`` is set but no per-row ``exposure`` was supplied.
    """
    if smearing is None:
        return False, 0.0, NO_SMEAR_UVW
    if exposure is None:
        raise ValueError(
            "time/bandwidth smearing needs the per-row integration time; pass the MS EXPOSURE column as 'exposure'."
        )
    return True, smearing.bw_half, smearing.row_uvw(uvw, exposure)


def pointing_kernel_args(prepared: PreparedSky, times, antenna1, antenna2) -> tuple:
    """The ``(ptg_order, ptg_p, ptg_q, beam_dl, beam_dm, beam_lap)`` trailing beam-kernel arguments.

    Placeholders (:data:`~simms.skymodel.kernels.NO_POINTING`) when ``prepared`` has no
    pointing model; otherwise each row's offsets for its two antennas, evaluated at the
    row's own time, and the derivative grids (the Laplacian a placeholder at order 1).
    """
    jones = prepared.beam_full_jones
    if prepared.pointing is None:
        return NO_POINTING_JONES if jones else NO_POINTING
    times = np.asarray(times, dtype=np.float64)
    order = prepared.pointing_order
    no_lap = NO_BEAM_GRAD_JONES if jones else NO_BEAM_GRAD
    return (
        order,
        prepared.pointing.offsets(times, antenna1),
        prepared.pointing.offsets(times, antenna2),
        np.ascontiguousarray(prepared.beam_dl),
        np.ascontiguousarray(prepared.beam_dm),
        np.ascontiguousarray(prepared.beam_lap) if order == 2 else no_lap,
    )


def predict_channel_block(
    prepared: PreparedSky,
    uvw: np.ndarray,
    chan_ids: np.ndarray,
    times: np.ndarray = None,
    antenna1: np.ndarray = None,
    antenna2: np.ndarray = None,
    exposure: np.ndarray = None,
    out_dtype: np.dtype = None,
) -> np.ndarray:
    """Predict one (row, channel) block, restricting the model to ``chan_ids``."""
    return predict_block(
        prepared.select_channels(chan_ids),
        uvw,
        times=times,
        antenna1=antenna1,
        antenna2=antenna2,
        exposure=exposure,
        out_dtype=out_dtype,
    )


def predict_block(
    prepared: PreparedSky,
    uvw: np.ndarray,
    times: np.ndarray = None,
    antenna1: np.ndarray = None,
    antenna2: np.ndarray = None,
    exposure: np.ndarray = None,
    noise_vis: float | None = None,
    out_dtype: np.dtype = None,
    seed=None,
) -> np.ndarray:
    """
    Predict visibilities for one block of rows.

    Parameters
    ----------
    prepared : PreparedSky
        Sky model arrays from :func:`prepare_skymodel`.
    uvw : numpy.ndarray
        UVW coordinates of shape (nrows, 3), in metres.
    times : numpy.ndarray, optional
        Time stamp per row. Required if the model contains transient sources.
    exposure : numpy.ndarray, optional
        Integration time per row (the MS ``EXPOSURE`` column, seconds). Required
        if ``prepared`` carries a :class:`~simms.skymodel.smearing.Smearing`.
    noise_vis : float, optional
        RMS noise per visibility (Jy). If provided, noise is added.
    seed : optional
        Seed for that noise; see :func:`sim_noise`. ``None`` draws fresh entropy. When
        looping over blocks, pass one shared ``numpy.random.Generator`` rather than a
        repeated integer, or every block gets the same realisation.
    out_dtype : numpy.dtype, optional
        Complex dtype to cast the result to. Sources are summed in the (higher)
        precision of ``prepared.bmat``, so a single-precision output column does
        not degrade the accumulation. Named ``out_dtype`` because ``da.blockwise``
        consumes any ``dtype`` kwarg itself and would never forward it.

    Returns
    -------
    numpy.ndarray
        Visibility array of shape (nrows, nchan, ncorr).
    """
    uvw = np.ascontiguousarray(uvw, dtype=np.float64)
    nrow = uvw.shape[0]
    ncorr = prepared.ncorr
    nspec = prepared.nspec

    if prepared.unique_times is None:
        time_index = np.zeros(nrow, dtype=np.int64)
    else:
        if times is None:
            raise ValueError("parameter 'times' must be provided for skymodels with transient sources")
        time_index = np.searchsorted(prepared.unique_times, times).astype(np.int64)

    smear_args = smear_kernel_args(prepared.smearing, uvw, exposure)
    if prepared.pointing is not None and not prepared.beam_enabled:
        raise ValueError("pointing errors act through the primary beam, but no beam is attached")

    vis = np.zeros((nrow, prepared.freqs.size, nspec), dtype=prepared.bmat.dtype)
    if prepared.beam_enabled:
        if times is None or antenna1 is None or antenna2 is None:
            raise ValueError("primary beam prediction requires 'times', 'antenna1' and 'antenna2'")
        # Map each row's timestamp to its position on the (time-uniform) PA grid and
        # interpolate between the two bracketing samples.
        tgrid = prepared.tgrid
        dt = tgrid[1] - tgrid[0]
        if dt > 0:
            gpos = np.clip((np.asarray(times, dtype=np.float64) - tgrid[0]) / dt, 0.0, tgrid.size - 1)
        else:
            gpos = np.zeros(nrow, dtype=np.float64)  # degenerate (zero-span) grid
        pa_lo = np.clip(np.floor(gpos).astype(np.int64), 0, tgrid.size - 2)
        pa_wt = np.clip(gpos - pa_lo, 0.0, 1.0)
        a1 = np.ascontiguousarray(antenna1)
        a2 = np.ascontiguousarray(antenna2)
        common = (
            uvw,
            prepared.freqs,
            prepared.uniform_freqs,
            prepared.lmn,
            prepared.gauss_shape,
            prepared.is_gauss,
            prepared.bmat,
            prepared.lightcurve,
            time_index,
            vis,
            a1,
            a2,
            prepared.ant_type,
            prepared.beam_grid,
            pa_lo,
            pa_wt,
        )
        ptg_args = pointing_kernel_args(prepared, times, a1, a2)
        if prepared.beam_full_jones:
            predict_vis_jones(*common, *smear_args, *ptg_args)
        else:
            predict_vis_beam(*common, prepared.corr_feed_p, prepared.corr_feed_q, *smear_args, *ptg_args)
    else:
        predict_vis(
            uvw,
            prepared.freqs,
            prepared.uniform_freqs,
            prepared.lmn,
            prepared.gauss_shape,
            prepared.is_gauss,
            prepared.bmat,
            prepared.lightcurve,
            time_index,
            vis,
            *smear_args,
        )

    if nspec != ncorr:
        # Unpolarised: XX == YY == Stokes I, cross-hands vanish.
        vis = stack_unpolarised_vis(vis[..., 0], ncorr)

    if noise_vis:
        vis = add_noise(vis, noise_vis, seed=seed)

    if out_dtype is not None:
        vis = vis.astype(out_dtype, copy=False)
    return vis


def compute_vis(
    skymodel: ASCIISkymodel,
    uvw: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray = None,
    ncorr: int = 2,
    polarisation: bool = False,
    linear_basis: bool = True,
    ra0: float | None = None,
    dec0: float | None = None,
    noise_vis: float | None = None,
    unique_times: np.ndarray = None,
    dtype: np.dtype = np.complex128,
    seed=None,
):
    """
    Compute model visibilities for an ASCII sky model.

    Convenience wrapper that prepares the sky model and predicts a single block.
    Callers looping over row blocks should call :func:`prepare_skymodel` once and
    :func:`predict_block` per block instead: transient lightcurves are referenced
    to the first of `unique_times`, which must therefore span the whole
    observation rather than a single block.

    Parameters
    ----------
    skymodel : ASCIISkymodel
        Parsed sky model object.
    uvw : numpy.ndarray
        UVW coordinates of shape (nrows, 3), in metres.
    freqs : numpy.ndarray
        Channel centre frequencies (Hz).
    times : numpy.ndarray, optional
        Time stamps per row if transient sources are present.
    ncorr : int, optional
        Number of correlations (2 or 4). Default 2.
    polarisation : bool, optional
        If True, include cross-hands when available. Default False.
    linear_basis : bool, optional
        Use linear (True) or circular (False) basis. Default True.
    ra0 : float, optional
        Phase centre right ascension (radians).
    dec0 : float, optional
        Phase centre declination (radians).
    noise_vis : float, optional
        RMS noise per visibility (Jy). If provided, noise is added.
    seed : optional
        Seed for that noise; see :func:`sim_noise`. ``None`` draws fresh entropy.
    unique_times : numpy.ndarray, optional
        Sorted unique time stamps of the whole observation. Defaults to the
        unique values of `times`, which is only correct when `uvw` covers every
        row of the observation.
    dtype : numpy.dtype, optional
        Complex dtype of the output. Default complex128.

    Returns
    -------
    numpy.ndarray
        Visibility array of shape (nrows, nchan, ncorr).
    """
    if unique_times is None and times is not None and skymodel.has_transient:
        unique_times = np.unique(times)
    prepared = prepare_skymodel(
        skymodel,
        freqs,
        ra0,
        dec0,
        ncorr=ncorr,
        polarisation=polarisation,
        linear_basis=linear_basis,
        unique_times=unique_times,
        dtype=dtype,
    )
    return predict_block(prepared, uvw, times=times, noise_vis=noise_vis, seed=seed)
