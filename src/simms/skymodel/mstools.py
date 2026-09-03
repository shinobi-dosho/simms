from __future__ import annotations

from dataclasses import dataclass, replace

import dask.array as da
import numpy as np
from daskms import xds_from_ms, xds_from_table

from simms.constants import FWHM_TO_GAUSS_SCALE
from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.skymodel.kernels import (
    NO_SMEAR_UVW,
    is_uniform_grid,
    predict_vis,
    predict_vis_beam,
    predict_vis_jones,
)
from simms.skymodel.smearing import Smearing
from simms.utilities import radec2lm

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
        return replace(
            self,
            freqs=freqs,
            bmat=self.bmat[:, :, chan_ids],
            uniform_freqs=is_uniform_grid(freqs),
            beam_grid=beam_grid,
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
    """
    from simms.skymodel.beams import (
        BEAM_GRID_MAX_GIB_DEFAULT,
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
    if full_jones:
        beam_grid = build_beam_grid_jones(
            providers, type_is_altaz, ell, emm, prepared.freqs, chi_grid, basis_transform, max_gib=max_gib
        )
        corr_feed_p = corr_feed_q = None
    else:
        beam_grid = build_beam_grid(providers, type_is_altaz, ell, emm, prepared.freqs, chi_grid, max_gib=max_gib)
        corr_feed_p, corr_feed_q = corr_feed_maps(ncorr)
    return replace(
        prepared,
        beam_enabled=True,
        beam_full_jones=full_jones,
        ant_type=np.ascontiguousarray(ant_type, dtype=np.int64),
        beam_grid=beam_grid,
        tgrid=tgrid,
        corr_feed_p=corr_feed_p,
        corr_feed_q=corr_feed_q,
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
        if prepared.beam_full_jones:
            predict_vis_jones(*common, *smear_args)
        else:
            predict_vis_beam(*common, prepared.corr_feed_p, prepared.corr_feed_q, *smear_args)
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
