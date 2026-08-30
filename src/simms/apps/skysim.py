from __future__ import annotations

import glob
import logging
import os.path
from types import SimpleNamespace
from typing import Literal

import dask
import dask.array as da
import numpy as np
import shinobi
from dask import config as dask_config
from daskms import xds_from_ms, xds_from_table, xds_to_table
from pydantic import BaseModel, Field
from tqdm.dask import TqdmCallback

from simms import BIN, SCHEMADIR, set_logger
from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.skymodel.beams import load_beam_config, resolve_antenna_beams
from simms.skymodel.corruptions import apply_corruptions, load_corruption_spec, needs_feed_basis, validate_spec
from simms.skymodel.fits_skies import component_sky_from_fits_dft, predict_fits_channel_block, prepare_fits_sky
from simms.skymodel.mstools import (
    attach_beam,
    auto_row_chunks,
    noise_visibilities,
    predict_channel_block,
    prepare_skymodel,
    to_full_corr,
    vis_noise_from_sefd_and_ms,
)
from simms.skymodel.wsclean_skies import prepare_wsclean_sky

log = logging.getLogger(BIN.skysim)


def _beam_row_args(msds, beam_ctx):
    """The ``(ANTENNA1, idx, ANTENNA2, idx)`` blockwise args, or ``None`` slots when off."""
    if beam_ctx:
        return (msds.ANTENNA1.data, ("row",), msds.ANTENNA2.data, ("row",))
    return (None, None, None, None)


def _array_lonlat(positions):
    """Geodetic (lon, lat) in radians from the mean of ITRF/ECEF antenna positions."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    x, y, z = np.asarray(positions, dtype=np.float64).mean(axis=0)
    loc = EarthLocation.from_geocentric(x * u.m, y * u.m, z * u.m)
    return loc.lon.to_value(u.rad), loc.lat.to_value(u.rad)


def _corr_basis(codes, what="Primary beam"):
    """'linear' or 'circular' from POLARIZATION.CORR_TYPE codes; raise on anything else.

    ``what`` names the caller in the error, since more than one path depends on
    the correlations being in a standard order.
    """
    codes = list(codes)
    if codes in ([9, 12], [9, 10, 11, 12]):  # XX(9) XY(10) YX(11) YY(12)
        return "linear"
    if codes in ([5, 8], [5, 6, 7, 8]):  # RR(5) RL(6) LR(7) LL(8)
        return "circular"
    raise RuntimeError(
        f"{what} needs standard linear (XX..YY) or circular (RR..LL) correlations; got CORR_TYPE codes {codes}."
    )


class _BeamContext:
    """Everything needed to attach a primary beam to a prepared sky."""

    def __init__(self, opts, ms, msds, ra0, dec0, ncorr):
        ant_ds = xds_from_table(f"{ms}::ANTENNA")[0]
        pol_ds = xds_from_table(f"{ms}::POLARIZATION")[0]
        # A Cattery/DDFacet heterogeneous-beam json types antennas by raw ANTENNA.NAME (its
        # own convention, exact/regex/cmd::default); simms' own beam-config YAML instead
        # groups by the TELESCOPE_NAME-column label. Detect which by the config file's extension.
        is_cattery_json = str(opts.primary_beam).lower().endswith(".json")
        if is_cattery_json:
            typing_col = "NAME"
        else:
            typing_col = opts.telescope_name_column
            if typing_col not in ant_ds:
                # The per-antenna telescope name is required and never inferred (a single,
                # authoritative source). Fail clearly instead of guessing from e.g. dish size.
                raise RuntimeError(
                    f"The ANTENNA table has no '{typing_col}' column (the per-antenna telescope "
                    f"name that selects a primary beam). Create the MS with telsim, or point "
                    f"--telescope-name-column at the column that holds it."
                )
        type_keys, mount, pos, t0, t1, interval, corr_type = dask.compute(
            ant_ds[typing_col].data,
            ant_ds.MOUNT.data,
            ant_ds.POSITION.data,
            msds.TIME.data.min(),
            msds.TIME.data.max(),
            msds.INTERVAL.data[0],
            pol_ds.CORR_TYPE.data[0],
        )
        self.basis = _corr_basis(list(np.asarray(corr_type).ravel()))
        self.full_jones = opts.beam_jones == "full"
        if is_cattery_json:
            from simms.skymodel.beams import load_cattery_beam_json, resolve_cattery_antenna_beams

            cattery_cfg = load_cattery_beam_json(opts.primary_beam)
            self.ant_type, self.providers, self.type_is_altaz = resolve_cattery_antenna_beams(
                type_keys, mount, cattery_cfg, pol_basis=self.basis, l_axis=opts.beam_l_axis, m_axis=opts.beam_m_axis
            )
        else:
            beam_config = load_beam_config(opts.primary_beam)
            self.ant_type, self.providers, self.type_is_altaz = resolve_antenna_beams(
                type_keys, mount, beam_config, opts.beam_band
            )
        self.lon, self.lat = _array_lonlat(pos)
        # The beam is centred on the antenna pointing centre (POINTING.DIRECTION), not the phase
        # centre; source l/m are prepared for the phase centre, so keep it too for reprojection.
        from simms.skymodel.beams import read_pointing_centre

        self.phase_ra0, self.phase_dec0 = ra0, dec0
        self.ra0, self.dec0 = read_pointing_centre(ms, ra0, dec0, opts.field_id)
        self.ncorr = ncorr
        self.t_start = float(t0)
        self.duration = float(t1 - t0) + float(interval)
        self.pa_step = opts.beam_pa_step
        self.beam_grid_max_gib = opts.beam_grid_max_gib

    @property
    def brightness_linear_basis(self):
        """Brightness is built in the linear feed basis when beams are on.

        Diagonal beams require linear correlations anyway; full Jones needs the
        feed-basis coherency (the basis transform to the MS frame is folded into the
        beam Jones).
        """
        return True

    def attach_image(self, prepared, freqs):
        """Apply an approximate PA-averaged power beam to a FITS-image sky model (any basis)."""
        from simms.skymodel.fits_skies import attach_image_beam

        if len(self.providers) > 1:
            log.warning(
                "Primary beam on the FITS-image path uses a single representative antenna "
                "type, but %d types are present; heterogeneity is ignored.",
                len(self.providers),
            )
        mid_freq = 0.5 * (float(freqs[0]) + float(freqs[-1]))
        return attach_image_beam(
            prepared,
            self.providers[0],
            bool(self.type_is_altaz[0]),
            self.ra0,
            self.dec0,
            self.lon,
            self.lat,
            self.t_start,
            self.duration,
            self.pa_step,
            mid_freq,
            phase_ra0=self.phase_ra0,
            phase_dec0=self.phase_dec0,
        )

    def attach_aterm(self, prepared, freq_tol):
        """Attach exact per-antenna a-terms to a gridder-backend FITS sky (see simms.skymodel.aterms)."""
        from simms.skymodel.aterms import attach_fits_aterm
        from simms.skymodel.beams import corr_basis_transform

        if self.full_jones:
            if self.ncorr != 4:
                raise RuntimeError("--beam-jones full requires 4 correlations (XX,XY,YX,YY or RR,RL,LR,LL).")
            basis_transform = corr_basis_transform(self.basis == "circular")
        else:
            if self.basis != "linear":
                # Callers fall back to the scalar power beam before reaching here.
                raise RuntimeError("Diagonal per-feed a-terms require linear correlations.")
            basis_transform = None
        return attach_fits_aterm(
            prepared,
            ant_type=self.ant_type,
            providers=self.providers,
            type_is_altaz=self.type_is_altaz,
            ra0=self.ra0,
            dec0=self.dec0,
            lon=self.lon,
            lat=self.lat,
            t_start=self.t_start,
            duration=self.duration,
            pa_step=self.pa_step,
            freq_tol=freq_tol,
            full_jones=self.full_jones,
            basis_transform=basis_transform,
            phase_ra0=self.phase_ra0,
            phase_dec0=self.phase_dec0,
            max_gib=self.beam_grid_max_gib,
        )

    def attach(self, prepared):
        """Force full-correlation brightness and attach the beam grid (diagonal or full Jones)."""
        from simms.skymodel.beams import corr_basis_transform

        if self.full_jones:
            if self.ncorr != 4:
                raise RuntimeError("--beam-jones full requires 4 correlations (XX,XY,YX,YY or RR,RL,LR,LL).")
            basis_transform = corr_basis_transform(self.basis == "circular")
        elif self.basis != "linear":
            raise RuntimeError(
                "The diagonal primary beam requires linear correlations; use --beam-jones full "
                "for a circular-correlation MS."
            )
        else:
            basis_transform = None
        prepared = to_full_corr(prepared)
        return attach_beam(
            prepared,
            self.ant_type,
            self.providers,
            self.type_is_altaz,
            self.ra0,
            self.dec0,
            self.lon,
            self.lat,
            self.t_start,
            self.duration,
            self.pa_step,
            self.ncorr,
            full_jones=self.full_jones,
            basis_transform=basis_transform,
            phase_ra0=self.phase_ra0,
            phase_dec0=self.phase_dec0,
            beam_grid_max_gib=self.beam_grid_max_gib,
        )


def runit(opts):
    # Set logger here, so subsequent modeules get it via logging.getLogger(<name>)
    set_logger(BIN.skysim, opts.log_level)

    # --seed only ever seeded the thermal noise, so it maps onto --seed-noise.
    seed_noise = opts.seed_noise
    if opts.seed is not None:
        log.warning("--seed is deprecated; use --seed-noise (the realisation for a given value is unchanged).")
        if seed_noise is None:
            seed_noise = opts.seed

    ms = opts.ms
    ascii_sky = opts.ascii_sky
    fs = opts.fits_sky
    wsclean_sky = opts.wsclean_sky

    dask_config.set(scheduler="threads", num_workers=opts.nworkers)

    if sum(bool(x) for x in (ascii_sky, fs, wsclean_sky)) > 1:
        raise RuntimeError("Choose a single sky model: one of --ascii-sky, --fits-sky, or --wsclean-sky.")

    # --row-chunks is an upper bound, not a fixed size: a fixed size ties the task
    # count to the length of the MS, so a short track can produce fewer chunks than
    # there are workers and leave most of them idle. Probe the row count first (a
    # metadata-only open) and size the chunks against --nworkers.
    ms_open = dict(
        group_cols=["DATA_DESC_ID"],
        taql_where=f"FIELD_ID=={opts.field_id}",
    )
    nrows = xds_from_ms(ms, **ms_open, chunks=dict(row=opts.row_chunks))[opts.spw_id].UVW.data.shape[0]
    row_chunks = auto_row_chunks(nrows, opts.nworkers, cap=opts.row_chunks)
    if row_chunks != opts.row_chunks:
        log.info(
            f"Using {row_chunks} rows/chunk ({-(-nrows // row_chunks)} chunks over "
            f"{opts.nworkers} workers); --row-chunks {opts.row_chunks} would have given "
            f"{-(-nrows // opts.row_chunks)}."
        )

    msds = xds_from_ms(ms, **ms_open, chunks=dict(row=row_chunks))[opts.spw_id]

    vis_noise = vis_noise_from_sefd_and_ms(ms, opts.sefd, opts.spw_id, opts.field_id) if opts.sefd else 0

    spw_ds = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    field_ds = xds_from_table(f"{ms}::FIELD")[0]

    radec0, freqs, dfreq = dask.compute(
        field_ds.PHASE_DIR.data[opts.field_id],
        spw_ds.CHAN_FREQ.data[opts.spw_id],
        spw_ds.CHAN_WIDTH.data[opts.spw_id][0],
    )
    ra0, dec0 = radec0[0][0], radec0[0][1]
    ncorr = msds.DATA.data.shape[-1]
    vis_dtype = msds.DATA.data.dtype
    linear_basis = opts.pol_basis == "linear"

    # Primary beam (sky models only, not noise-only runs). When enabled, the model must
    # carry every correlation so the per-feed voltage can be applied (nspec == ncorr).
    has_sky = bool(ascii_sky or wsclean_sky or fs)
    if opts.primary_beam and not has_sky:
        log.warning("--primary-beam is set but no sky model was given; the primary beam is ignored.")
    beam_ctx = _BeamContext(opts, ms, msds, ra0, dec0, ncorr) if opts.primary_beam and has_sky else None

    # Primary-beam handling for the FITS-image path. The default ('aterm') applies exact
    # per-antenna a-terms; diagonal per-feed beams cannot represent circular correlations,
    # so that combination falls back to the basis-independent scalar power beam.
    fits_beam_mode = opts.fits_beam_mode if (beam_ctx and fs) else None
    if fits_beam_mode == "aterm" and not beam_ctx.full_jones and beam_ctx.basis != "linear":
        log.warning(
            "Diagonal per-feed a-terms need linear correlations but this MS is circular; falling "
            "back to the PA-averaged power beam (--fits-beam-mode average). Use --beam-jones full "
            "for exact a-terms on a circular-correlation MS."
        )
        fits_beam_mode = "average"
    if fits_beam_mode == "average" and opts.beam_jones == "full":
        log.warning("--beam-jones full is ignored with --fits-beam-mode average (a scalar power beam is applied).")

    # Channel chunking. A channel-index array carries the chan dimension into
    # da.blockwise; each predict task restricts the model to its own channels.
    nchan = freqs.size
    chan_chunks = opts.chan_chunks if opts.chan_chunks and opts.chan_chunks > 0 else nchan
    chan_ids = da.arange(nchan, chunks=chan_chunks)

    simvis = None

    if ascii_sky:
        if opts.source_schema and opts.ascii_species:
            raise RuntimeError("Cannot use custom map and built-in map simultaneously")
        elif opts.source_schema:
            source_schema = opts.source_schema
        elif opts.ascii_species:
            source_schema = f"{SCHEMADIR}/{opts.ascii_species}.yaml"
        else:
            source_schema = f"{SCHEMADIR}/source_schema.yaml"

        skymodel = ASCIISkymodel(ascii_sky, delimiter=opts.ascii_delimiter, source_schema_file=source_schema)

        # Transient lightcurves are defined relative to the start of the
        # observation, so the time axis has to be resolved globally rather than
        # per row block.
        unique_times = np.unique(msds.TIME.data.compute()) if skymodel.has_transient else None

        # Prepared once and shared by every block, rather than rebuilt per block.
        # Sources are summed in double precision regardless of the column dtype.
        prepared = prepare_skymodel(
            skymodel,
            freqs,
            ra0,
            dec0,
            ncorr=ncorr,
            polarisation=opts.polarisation or bool(beam_ctx),
            # Beams build the coherency in the linear feed basis (the MS-basis transform is
            # folded into the beam Jones); otherwise honour the requested pol basis.
            linear_basis=beam_ctx.brightness_linear_basis if beam_ctx else linear_basis,
            unique_times=unique_times,
        )
        if beam_ctx:
            prepared = beam_ctx.attach(prepared)

        # A blockwise index of None passes the argument through untouched, so a
        # model without transients (and no beam) must be handed a literal None, not
        # the (unblocked) TIME dask array.
        need_time = skymodel.has_transient or bool(beam_ctx)
        time_args = (msds.TIME.data, ("row",)) if need_time else (None, None)
        ant_args = _beam_row_args(msds, beam_ctx)

        simvis = da.blockwise(
            predict_channel_block,
            ("row", "chan", "corr"),
            prepared,
            None,
            msds.UVW.data,
            ("row", "uvw"),
            chan_ids,
            ("chan",),
            *time_args,
            *ant_args,
            out_dtype=vis_dtype,
            new_axes={"corr": ncorr},
            dtype=vis_dtype,
            concatenate=True,
        )

    elif wsclean_sky:
        # A WSClean component list shares the ASCII prediction path: it is flattened
        # into a PreparedSky and handed to the same kernel.
        prepared = prepare_wsclean_sky(wsclean_sky, freqs, ra0, dec0, ncorr=ncorr)
        if beam_ctx:
            prepared = beam_ctx.attach(prepared)

        time_args = (msds.TIME.data, ("row",)) if beam_ctx else (None, None)
        ant_args = _beam_row_args(msds, beam_ctx)

        simvis = da.blockwise(
            predict_channel_block,
            ("row", "chan", "corr"),
            prepared,
            None,
            msds.UVW.data,
            ("row", "uvw"),
            chan_ids,
            ("chan",),
            *time_args,
            *ant_args,
            out_dtype=vis_dtype,
            new_axes={"corr": ncorr},
            dtype=vis_dtype,
            concatenate=True,
        )

    # TODO(Sphe,Senkhosi) This is too tailored to a specific use case and input file schema.
    # We need more generic API for handling multiple FITS images as input.
    # Something like we do in skymods.skymodel_from_fits(stack_axis)
    elif fs:
        if os.path.exists(fs):
            if os.path.isdir(fs):
                fits_dir = fs
                fs = []
                try:
                    fs.append(glob.glob(f"{fits_dir}/*I.fits")[0])
                    fs.append(glob.glob(f"{fits_dir}/*Q.fits")[0])
                    fs.append(glob.glob(f"{fits_dir}/*U.fits")[0])
                    fs.append(glob.glob(f"{fits_dir}/*V.fits")[0])
                except IndexError as exc:
                    raise RuntimeError("Could not find all required FITS files in the specified directory") from exc

                if len(fs) > 4:
                    raise RuntimeError("Too many FITS files found in the specified directory")

            elif not fs.endswith(".fits"):
                raise OSError("Invalid FITS file specified")
        else:
            raise FileNotFoundError("FITS file/directory does not exist")

        # Prepared once and shared by every row block. Under a-term beams the DFT
        # backend consumes a feed-linear-basis brightness (like the ASCII path).
        prepare_kw = dict(
            nrow=msds.UVW.data.shape[0],
            polarisation=opts.polarisation,
            tol=opts.pixel_tol,
            backend=opts.predict_backend,
            spectrum=opts.fits_spectrum,
            spi_maps=opts.fits_spi,
            ref_freq=opts.fits_ref_freq,
            spectrum_order=opts.fits_spectrum_order,
            interpolation=opts.fits_sky_interp,
        )
        prepared = prepare_fits_sky(
            fs,
            ra0,
            dec0,
            freqs,
            dfreq,
            ncorr,
            linear_basis=True if fits_beam_mode == "aterm" else linear_basis,
            **prepare_kw,
        )

        # A-terms hold per-antenna-type voltage maps over the whole image, so a large
        # enough image cannot be served within --beam-grid-max-gib. Degrade to the
        # legacy PA-averaged power beam (3.0.1 behaviour) rather than failing a run
        # that the previous release completed. Checked here, not before preparing,
        # because a non-SIN image is reprojected during preparation and only then has
        # its final size.
        if fits_beam_mode == "aterm" and prepared.backend != "dft":
            from simms.skymodel.aterms import aterm_cache_min_gib

            needed = aterm_cache_min_gib(prepared.npix_l, prepared.npix_m, beam_ctx.full_jones)
            if needed > opts.beam_grid_max_gib:
                log.warning(
                    "A-term beams need a %.2f GiB voltage-map cache for this %dx%d image, above the "
                    "%.2f GiB --beam-grid-max-gib ceiling; falling back to the PA-averaged power beam "
                    "(--fits-beam-mode average). Raise --beam-grid-max-gib to keep exact per-antenna beams.",
                    needed,
                    prepared.npix_l,
                    prepared.npix_m,
                    opts.beam_grid_max_gib,
                )
                fits_beam_mode = "average"
                if not linear_basis:
                    # The model was built in the linear feed basis only for the a-term
                    # path; the scalar power beam is basis-independent, so honour
                    # --pol-basis again rather than silently returning linear correlations.
                    prepared = prepare_fits_sky(
                        fs, ra0, dec0, freqs, dfreq, ncorr, linear_basis=linear_basis, **prepare_kw
                    )

        use_component_path = False
        if fits_beam_mode == "average":
            # Approximate: one PA-averaged power beam on the apparent sky (see attach_image).
            prepared = beam_ctx.attach_image(prepared, freqs)
        elif fits_beam_mode == "aterm":
            if prepared.backend == "dft":
                # Few bright pixels: bridge to the component kernels, which apply
                # per-antenna beams exactly (same machinery as the ASCII path).
                log.info("FITS DFT backend with a primary beam: using the exact per-component beam kernels.")
                prepared = beam_ctx.attach(component_sky_from_fits_dft(prepared))
                use_component_path = True
            else:
                prepared = beam_ctx.attach_aterm(prepared, opts.aterm_freq_tol)

        epsilon = 1e-7 if opts.fft_precision == "double" else 1e-6

        if use_component_path:
            simvis = da.blockwise(
                predict_channel_block,
                ("row", "chan", "corr"),
                prepared,
                None,
                msds.UVW.data,
                ("row", "uvw"),
                chan_ids,
                ("chan",),
                msds.TIME.data,
                ("row",),
                *_beam_row_args(msds, beam_ctx),
                out_dtype=vis_dtype,
                new_axes={"corr": ncorr},
                dtype=vis_dtype,
                concatenate=True,
            )
        else:
            aterm_on = prepared.aterm is not None
            time_args = (msds.TIME.data, ("row",)) if aterm_on else (None, None)
            simvis = da.blockwise(
                predict_fits_channel_block,
                ("row", "chan", "corr"),
                prepared,
                None,
                msds.UVW.data,
                ("row", "uvw"),
                chan_ids,
                ("chan",),
                *time_args,
                *_beam_row_args(msds, beam_ctx if aterm_on else None),
                out_dtype=vis_dtype,
                epsilon=epsilon,
                do_wgridding=opts.do_wstacking,
                new_axes={"corr": ncorr},
                dtype=vis_dtype,
                concatenate=True,
            )

    # Corruptions go on the model, before the noise. Receiver noise enters the
    # signal chain after the antenna gains, so a noisy RIME is
    # V' = J_p V J_q^H + n -- corrupting the sum would gain-modulate the noise.
    if opts.corruptions:
        # Loaded and validated even when there is nothing to corrupt: a malformed
        # spec must fail here rather than let a noise-only run write a clean
        # column that looks like it was corrupted.
        spec = load_corruption_spec(opts.corruptions)
        validate_spec(spec, ncorr=ncorr)
        if needs_feed_basis(spec, ncorr):
            # diagonal/full terms map correlation index to feed index by
            # position, so they need the correlations in a standard order. A
            # scalar gain reaches every correlation alike and does not.
            corr_type = xds_from_table(f"{ms}::POLARIZATION")[0].CORR_TYPE.data[0].compute()
            _corr_basis(list(np.asarray(corr_type).ravel()), what="Non-scalar RIME corruptions")

        if simvis is None:
            if vis_noise:
                log.warning(
                    "--corruptions has no effect on a noise-only run: gains apply to the sky signal, "
                    "and thermal noise is added after the gain chain."
                )
        else:
            # Phase origins and array size come from the MS, not from this
            # field/SPW selection: skysim runs one field and one SPW at a time,
            # so selection-derived references would give the same antenna a
            # different gain in each run over the same MS.
            ant_ds = xds_from_table(f"{ms}::ANTENNA")[0]
            ms_nant = ant_ds.sizes["row"]
            all_freqs = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0].CHAN_FREQ.data
            ms_freq0 = float(all_freqs.min().compute())
            ms_t0 = float(xds_from_ms(ms, columns=["TIME"], group_cols=[])[0].TIME.data.min().compute())

            simvis = apply_corruptions(
                simvis,
                msds.TIME.data,
                msds.ANTENNA1.data,
                msds.ANTENNA2.data,
                freqs,
                spec,
                random_seed=opts.seed_gains,
                nant=ms_nant,
                time_ref=ms_t0,
                freq_ref=ms_freq0,
            )

    # Thermal noise, added once for every path. With --seed-noise the draw is
    # reproducible across runs at a given chunking; changing the chunking changes
    # the realisation, as dask keys each block's stream to its position in the grid.
    if vis_noise:
        noise = noise_visibilities(
            msds.DATA.data.shape,
            (msds.UVW.data.chunks[0], chan_chunks, ncorr),
            vis_noise,
            vis_dtype,
            seed=seed_noise,
        )
        simvis = noise if simvis is None else simvis + noise

    if simvis is None:
        raise RuntimeError("Nothing to simulate: provide a sky model (--ascii-sky/--fits-sky/--wsclean-sky) or --sefd.")

    # 'add'/'subtract' combine the simulation with an existing column. --input-column
    # names it, but "add to DATA" is what the mode means without one, so it defaults to
    # the output column -- otherwise the mode was silently a no-op that overwrote it.
    input_column = opts.input_column
    if opts.mode in ("add", "subtract") and not input_column:
        input_column = opts.column
        log.info("--mode %s with no --input-column: combining with '%s'.", opts.mode, input_column)
    elif opts.mode == "sim" and input_column:
        log.warning("--input-column '%s' is ignored in 'sim' mode, which overwrites '%s'.", input_column, opts.column)
        input_column = None

    if input_column:
        if not hasattr(msds, input_column):
            raise RuntimeError(
                f"Specified input-column '{input_column}' does not exist in the MS. "
                f"--mode {opts.mode} needs an existing column to combine with; run with "
                f"--mode sim first, or point --input-column at a column that is there."
            )
        incol = getattr(msds, input_column).data
        outvis = incol + simvis if opts.mode == "add" else incol - simvis
    else:
        outvis = simvis

    ms_dsl = [
        msds.assign(
            **{
                opts.column: (("row", "chan", "corr"), outvis),
            },
        )
    ]

    writes = xds_to_table(ms_dsl, ms, columns=[opts.column])

    with TqdmCallback(desc="Computing and writing visibilities"):
        da.compute(writes)


class SimmsOutputs(BaseModel):
    """Passthrough MS path, so telsim/skysim can be wired into a shinobi Recipe or dosho."""

    ms: str | None = None


@shinobi.pystep(name=BIN.skysim, info="Predict model visibilities from a sky model into an MS.")
def skysim(
    ms: str = Field(..., description="Measurement set."),
    ascii_sky: str | None = Field(
        None,
        description="Catalogue of sources. See the documentation for accepted units.",
        json_schema_extra={"abbreviation": "as"},
    ),
    fits_sky: str | None = Field(
        None,
        description="FITS file (or directory of Stokes cubes) containing the sky model.",
        json_schema_extra={"abbreviation": "fs"},
    ),
    wsclean_sky: str | None = Field(
        None,
        description="WSClean component list (point and Gaussian components, Stokes I).",
        json_schema_extra={"abbreviation": "ws"},
    ),
    fits_sky_interp: Literal["nearest", "linear", "cubic"] = Field(
        "linear",
        description="Interpolation method when the MS and FITS frequency grids do not match and the cube is kept.",
        json_schema_extra={"abbreviation": "fsi"},
    ),
    polarisation: bool = Field(
        True,
        description="Simulate all available Stokes parameters. If false, only Stokes I.",
        json_schema_extra={"abbreviation": "pol"},
    ),
    pol_basis: Literal["linear", "circular"] = Field("linear", description="Polarization basis for the simulation."),
    pixel_tol: float = Field(
        1e-7,
        description="Minimum brightness for a pixel to be considered in direct Fourier transform.",
        json_schema_extra={"abbreviation": "pt"},
    ),
    fits_spectrum: Literal["auto", "flat", "poly", "cube"] = Field(
        "auto",
        description="How the FITS sky model varies with frequency.",
        json_schema_extra={"abbreviation": "fsp"},
    ),
    fits_spi: list[str] | None = Field(
        None,
        description="Spectral-index (and higher-order) coefficient maps, ordered c1, c2, ... Requires --fits-ref-freq.",
    ),
    fits_ref_freq: float | None = Field(
        None,
        description="Reference frequency (Hz) of an analytic FITS spectrum. Defaults to the MS band centre.",
        json_schema_extra={"abbreviation": "frf"},
    ),
    fits_spectrum_order: int = Field(
        2, description="Order of the fitted log-polynomial spectrum. 1 is a plain spectral index."
    ),
    predict_backend: Literal["auto", "dft", "fft", "perchan"] = Field(
        "auto", description="Backend for FITS sky model prediction."
    ),
    fft_precision: Literal["single", "double"] = Field("double", description="Precision of the FFT calculation."),
    do_wstacking: bool = Field(True, description="Whether to use w-stacking for FFT-based visibility prediction."),
    ascii_delimiter: str | None = Field(
        None, description="Delimiter used in the ascii-sky.", json_schema_extra={"abbreviation": "ad"}
    ),
    column: str = Field("DATA", description="Data column for simulation.", json_schema_extra={"abbreviation": "col"}),
    nworkers: int = Field(4, description="Number of workers (one per CPU)."),
    row_chunks: int = Field(
        10000,
        description="Maximum number of rows per chunk. Controls the row-wise task/memory "
        "granularity; the effective size is reduced when needed so every worker gets "
        "several chunks.",
        json_schema_extra={"abbreviation": "rcs"},
    ),
    chan_chunks: int | None = Field(
        None,
        description="Number of channels per chunk. Defaults to all channels in one chunk.",
        json_schema_extra={"abbreviation": "ccs"},
    ),
    primary_beam: str | None = Field(
        None,
        description="Beam model config: a simms beam-config YAML mapping each ANTENNA telescope "
        "name to a beam model, or a Cattery/DDFacet heterogeneous-beam json (--Beam-FITSFile json "
        "form, keyed by ANTENNA.NAME) if the path ends in .json. For Cattery/DDFacet beams a "
        "circular-correlation MS may be used when --beam-jones full is selected.",
        json_schema_extra={"abbreviation": "pb"},
    ),
    beam_band: Literal["UHF", "L"] = Field(
        "L", description="Default band for JimBeam entries that omit an explicit model/CSV."
    ),
    beam_pa_step: float = Field(
        1.0, description="Spacing (degrees) of the parallactic-angle grid the beam is sampled on."
    ),
    beam_grid_max_gib: float = Field(
        4.0, description="Hard ceiling (GiB) on the sampled beam grid held in memory for the whole run."
    ),
    beam_jones: Literal["diagonal", "full"] = Field(
        "diagonal", description="Primary-beam application: per-feed voltage or full 2x2 E-Jones."
    ),
    fits_beam_mode: Literal["aterm", "average"] = Field(
        "aterm",
        description="Primary-beam handling for the FITS-image path: 'aterm' applies exact per-antenna "
        "a-terms in the image domain (time- and frequency-interpolated, heterogeneity-aware); 'average' "
        "multiplies the sky by a single PA-averaged power beam (legacy approximation).",
        json_schema_extra={"abbreviation": "fbm"},
    ),
    aterm_freq_tol: float = Field(
        1e-3,
        description="Largest allowed error (in voltage-beam units, beam peak ~1) of the a-term's "
        "linear-in-frequency interpolation between knot channels. Smaller means more frequency knots; "
        "0 or negative samples the beam at every channel.",
        json_schema_extra={"abbreviation": "aft"},
    ),
    telescope_name_column: str = Field(
        "TELESCOPE_NAME",
        description="ANTENNA-table column holding the per-antenna telescope/type label that maps to a beam model.",
        json_schema_extra={"abbreviation": "tnc"},
    ),
    beam_l_axis: Literal["-X", "X"] = Field(
        "-X",
        description="Sign convention for a Cattery/DDFacet .json primary-beam config's L axis; "
        "ignored for YAML beam configs, which specify their own axis convention. "
        "Matches DDFacet's --Beam-FITSLAxis.",
        json_schema_extra={"abbreviation": "bla"},
    ),
    beam_m_axis: Literal["Y", "-Y"] = Field(
        "Y",
        description="Sign convention for a Cattery/DDFacet .json primary-beam config's M axis; "
        "ignored for YAML beam configs, which specify their own axis convention. "
        "Matches DDFacet's --Beam-FITSMAxis.",
        json_schema_extra={"abbreviation": "bma"},
    ),
    field_id: int = Field(0, description="Field ID.", json_schema_extra={"abbreviation": "fi"}),
    spw_id: int = Field(0, description="Spectral Window ID."),
    sefd: float | None = Field(None, description="Add noise using this SEFD value."),
    seed_noise: int | None = Field(
        None,
        description="Random seed for the thermal noise. Omit for a non-reproducible run. The "
        "realisation also depends on the row chunking, which --nworkers feeds into, so "
        "reproducing a previous run needs the same --seed-noise, --row-chunks and --nworkers. "
        "This is the renamed --seed; the same value gives the same noise.",
    ),
    seed_gains: int | None = Field(
        None,
        description="Random seed for the corruption terms' random phases/matrices. Omit for a "
        "deterministic per-label draw; the thermal noise stream is unaffected either way.",
    ),
    seed: int | None = Field(
        None,
        description="Deprecated alias for --seed-noise; --seed-noise wins if both are given.",
    ),
    corruptions: str | None = Field(
        None,
        description="YAML file describing RIME Jones corruptions to apply to the predicted visibilities.",
    ),
    ascii_species: Literal["bdsf_gaul", "aegean", "wsclean"] | None = Field(
        None, description="Non-simms sky model type.", json_schema_extra={"abbreviation": "asp"}
    ),
    input_column: str | None = Field(
        None,
        description="Column that 'add'/'subtract' combine the simulation with. Defaults to --column.",
        json_schema_extra={"abbreviation": "ic"},
    ),
    mode: Literal["sim", "add", "subtract"] = Field(
        "sim",
        description="How the simulation reaches --column: 'sim' overwrites it, 'add' adds the simulation to "
        "--input-column, 'subtract' subtracts it from --input-column. --input-column defaults to --column.",
    ),
    source_schema: str | None = Field(
        None,
        description="Custom source schema (YAML) mapping columns in a custom sky model to the columns simms expects.",
    ),
    log_level: str = Field("INFO", description="Logging verbosity."),
) -> SimmsOutputs:
    opts = SimpleNamespace(**locals())
    runit(opts)
    return SimmsOutputs(ms=ms)
