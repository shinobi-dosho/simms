"""Operations for the ``simms primary-beam`` command.

Four modes, none of which run a visibility simulation:
- ``to_fits``      : sample an analytic cosine-taper beam onto a FITS beam cube, in either
  simms' own single-file layout or the Cattery/DDFacet 8-file ``--Beam-Model FITS`` schema.
- ``tag_ms``       : write the per-antenna telescope-name column onto an existing MS.
- ``apply``/``correct`` : multiply / divide a sky model (FITS image or ASCII components) by
  the frequency- and parallactic-angle-averaged Stokes-I power beam ``A(l, m)``.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np

from simms import BIN
from simms.utilities import load_yaml

log = logging.getLogger(BIN.primary_beam)


# --------------------------------------------------------------------- geometry


def _observation(ms, field_id=0, spw_id=0):
    """Read the geometry an averaged beam needs from an MS (for the given field/spw)."""
    import dask
    from daskms import xds_from_ms, xds_from_table

    from simms.skymodel.beams import array_lonlat, read_pointing_centre

    ant = xds_from_table(f"{ms}::ANTENNA")[0]
    spw = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    field = xds_from_table(f"{ms}::FIELD")[0]
    msds = xds_from_ms(ms, group_cols=["DATA_DESC_ID"], taql_where=f"FIELD_ID=={int(field_id)}")[int(spw_id)]
    pos, t0, t1, interval, chan_freq, phase_dir = dask.compute(
        ant.POSITION.data,
        msds.TIME.data.min(),
        msds.TIME.data.max(),
        msds.INTERVAL.data[0],
        spw.CHAN_FREQ.data[int(spw_id)],
        field.PHASE_DIR.data[int(field_id)],
    )
    lon, lat = array_lonlat(pos)
    # Beam centre is the antenna pointing centre, not the phase centre. POINTING carries no
    # FIELD_ID, so the selected rows' TIME span is what picks this field's pointing.
    ra0, dec0 = read_pointing_centre(
        ms, phase_dir[0][0], phase_dir[0][1], int(field_id), time_range=(float(t0), float(t1))
    )
    return {
        "t_start": float(t0),
        "duration": float(t1 - t0) + float(interval),
        "lon": lon,
        "lat": lat,
        "freqs": np.asarray(chan_freq, dtype=np.float64),
        "ra0": ra0,
        "dec0": dec0,
    }


def _averaged_beam(provider, ell, emm, ra0, dec0, obs, pa_step):
    """Freq- and PA-averaged power beam ``A(l, m)`` at the given directions (beam centre ra0/dec0)."""
    from simms.skymodel.beams import averaged_power_beam, pa_sample_grid

    _, chi_grid = pa_sample_grid(obs["t_start"], obs["duration"], ra0, dec0, obs["lon"], obs["lat"], pa_step)
    return averaged_power_beam(provider, ell, emm, obs["freqs"], chi_grid)


def _angular_separation(ra1, dec1, ra2, dec2):
    """Great-circle angle (radians) between two directions given in radians."""
    return float(
        np.arccos(
            np.clip(
                np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2),
                -1.0,
                1.0,
            )
        )
    )


# --------------------------------------------------------------------- to-fits


def to_fits(opts):
    from astropy.coordinates import Angle

    from simms.skymodel.beams import JimBeamProvider, resolve_beam, write_beam_fits, write_beam_fits_cattery

    provider = resolve_beam(opts.beam_pattern, opts.beam_band)
    if not isinstance(provider, JimBeamProvider):
        raise RuntimeError("to-fits needs an analytic cosine-taper beam (CSV or built-in), not a FITS cube.")
    beam = provider.beam

    from simms.telescope.generate_ms import parse_frequency

    pixel_rad = Angle(opts.pixel_size).to_value("rad")
    npix = int(opts.npix)
    grid = (np.arange(npix) - npix // 2) * pixel_rad  # direction cosines, centred at 0

    # Uniform frequency grid (the FITS FREQ axis is linear and the model is continuous in
    # frequency, so a uniform resample loses nothing). Defaults follow the beam's table.
    nchan = int(opts.nchan) if opts.nchan else beam.freqs_mhz.size
    start = parse_frequency(opts.start_freq, "start-freq") if opts.start_freq else beam.freqs_mhz[0] * 1e6
    if opts.chan_width:
        width = parse_frequency(opts.chan_width, "chan-width")
        freqs = start + np.arange(nchan) * width
    elif nchan > 1:
        freqs = np.linspace(start, beam.freqs_mhz[-1] * 1e6, nchan)
    else:
        freqs = np.array([start])

    if opts.fits_format == "cattery":
        prefix = opts.output or "beam"
        if prefix.lower().endswith(".fits"):
            prefix = prefix[: -len(".fits")]
        paths = write_beam_fits_cattery(
            beam, grid, grid, freqs, prefix, pol_basis=opts.pol_basis, l_axis=opts.beam_l_axis, m_axis=opts.beam_m_axis
        )
        log.info(
            "Wrote Cattery-schema beam (%d x %d pixels, %d channels, %s basis) -> %s",
            npix,
            npix,
            freqs.size,
            opts.pol_basis,
            ", ".join(paths),
        )
    else:
        if opts.beam_l_axis != "-X" or opts.beam_m_axis != "Y":
            log.warning(
                "--beam-l-axis/--beam-m-axis only apply to --fits-format cattery; ignored for %r.", opts.fits_format
            )
        output = opts.output or "beam.fits"
        write_beam_fits(beam, grid, grid, freqs, output)
        log.info("Wrote beam FITS cube %s (%d x %d pixels, %d channels)", output, npix, npix, freqs.size)


# --------------------------------------------------------------------- tag-ms


def _resolve_labels(opts, names):
    """Per-antenna telescope-name labels from --label, --label-map, or --from-layout."""
    nant = len(names)
    if opts.label:
        return [str(opts.label)] * nant
    if opts.label_map:
        mapping = load_yaml(opts.label_map)
        missing = [n for n in names if n not in mapping]
        if missing:
            raise RuntimeError(f"--label-map has no entry for antennas: {missing[:5]}")
        return [str(mapping[n]) for n in names]
    if opts.from_layout:
        from simms.telescope.array_utilities import Array

        arr = Array(opts.from_layout)
        layout = dict(zip([str(x) for x in arr.names], [str(x) for x in arr.telescope_name], strict=True))
        missing = [n for n in names if n not in layout]
        if missing:
            raise RuntimeError(
                f"--from-layout {opts.from_layout!r} has no entry for MS antennas: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return [layout[n] for n in names]
    raise RuntimeError("tag-ms needs one of --label, --label-map or --from-layout.")


def tag_ms(opts):
    import dask
    import dask.array as da
    from daskms import xds_from_table, xds_to_table

    ms, col = opts.ms, opts.telescope_name_column
    ant = xds_from_table(f"{ms}::ANTENNA")[0]
    names = [str(x) for x in np.asarray(ant.NAME.data.compute()).astype(str)]
    labels = _resolve_labels(opts, names)
    if col in ant:
        log.warning("ANTENNA already has a %r column; overwriting it.", col)
    # casacore STRING columns are numpy object dtype, one chunk (see generate_ms).
    values = np.array(labels, dtype=object)
    tagged = ant.assign(**{col: (("row",), da.from_array(values, chunks=len(names)))})
    writes = xds_to_table([tagged], f"{ms}::ANTENNA", columns=[col], descriptor="mssubtable('ANTENNA')")
    dask.compute(writes)
    log.info("Tagged %d antennas in %s::ANTENNA[%s] -> %s", len(names), ms, col, sorted(set(labels)))


# --------------------------------------------------------------- apply / correct


def apply_correct_image(opts, invert):
    """Multiply (apply) or divide (correct) a FITS image by the averaged power beam."""
    from astropy.io import fits
    from astropy.wcs import WCS

    from simms.skymodel.fits_skies import pixel_lm

    with fits.open(opts.fits_sky) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header
        out_dtype = hdul[0].data.dtype

    cel = WCS(header).celestial
    # The primary beam sits on the antenna pointing centre (POINTING.DIRECTION) -- not the
    # correlator phase centre, and not necessarily the image's reference pixel. Centre the beam
    # (pixel l/m and the PA track) there; the image WCS only maps pixels to world coordinates.
    obs = _observation(opts.ms, opts.field_id, opts.spw_id)
    ra0, dec0 = obs["ra0"], obs["dec0"]
    img_ra0 = np.radians(cel.wcs.crval[cel.wcs.lng])
    img_dec0 = np.radians(cel.wcs.crval[cel.wcs.lat])
    sep = _angular_separation(img_ra0, img_dec0, ra0, dec0)
    if sep > np.radians(1.0 / 3600.0):  # > 1 arcsec: image reference and antenna pointing disagree
        log.warning(
            "Image reference pixel (%.6f, %.6f deg) differs from the antenna pointing centre "
            "(%.6f, %.6f deg) by %.1f arcsec; centring the beam on the pointing centre.",
            np.degrees(img_ra0),
            np.degrees(img_dec0),
            np.degrees(ra0),
            np.degrees(dec0),
            np.degrees(sep) * 3600.0,
        )

    # Standard axis order: FITS axis 1 = RA (numpy last), axis 2 = DEC (numpy second-last).
    npix_dec, npix_ra = data.shape[-2], data.shape[-1]
    i_ra, j_dec = np.meshgrid(np.arange(npix_ra), np.arange(npix_dec))  # (npix_dec, npix_ra)

    ell, emm = pixel_lm(cel, ra0, dec0, i_ra.ravel(), j_dec.ravel())
    A = _averaged_beam(provider_from(opts), ell, emm, ra0, dec0, obs, opts.beam_pa_step)
    A = A.reshape(npix_dec, npix_ra)

    if invert:
        safe = np.where(opts.pb_cutoff > A, np.nan, A)  # blank where the beam is negligible
        result = data / safe
    else:
        result = data * A

    output = opts.output or ("corrected.fits" if invert else "apparent.fits")
    fits.PrimaryHDU(data=result.astype(out_dtype, copy=False), header=header).writeto(output, overwrite=True)
    log.info("%s primary beam -> %s", "Corrected" if invert else "Applied", output)


def apply_correct_ascii(opts, invert):
    """Scale ASCII component fluxes by the averaged power beam (apply) or its inverse (correct)."""
    from simms.skymodel.ascii_skies import ASCIISkymodel, ASCIISource
    from simms.utilities import radec2lm

    obs = _observation(opts.ms, opts.field_id, opts.spw_id)
    # ASCIISkymodel falls back to the built-in source schema when source_schema is unset
    sky = ASCIISkymodel(opts.ascii_sky, delimiter=opts.ascii_delimiter, source_schema_file=opts.source_schema)
    lm = np.array([radec2lm(obs["ra0"], obs["dec0"], s.ra, s.dec) for s in sky.sources])
    ell, emm = (lm[:, 0], lm[:, 1]) if len(lm) else (np.array([]), np.array([]))
    A = _averaged_beam(provider_from(opts), ell, emm, obs["ra0"], obs["dec0"], obs, opts.beam_pa_step)

    # ASCIISkymodel is read-only, so we edit the flux fields in the original text (preserving
    # formatting, comments and unknown columns) rather than reserialising. Each parsed source
    # carries its line index (source.lineno) -- the single source of truth for which line it
    # came from -- so we never re-implement the comment/blank-line skipping here.
    with open(opts.ascii_sky) as fh:
        lines = fh.read().splitlines()
    cols = lines[0].replace("#format:", "").strip().split(sky.delimiter)
    # The header holds column aliases when a custom schema renames them; map each
    # column back to its schema field before looking for the stokes columns.
    alias_to_field = ASCIISource(sky.schema).alias_to_field_mapper()
    fields_by_col = [alias_to_field.get(col, col) for col in cols]
    stokes_idx = [i for i, f in enumerate(fields_by_col) if f in ("stokes_i", "stokes_q", "stokes_u", "stokes_v")]

    dropped = set()
    for src, source in enumerate(sky.sources):
        a = A[src]
        if invert and a < opts.pb_cutoff:  # source outside the beam -> drop it
            dropped.add(source.lineno)
            continue
        factor = (1.0 / a) if invert else a
        fields = lines[source.lineno].split(sky.delimiter)
        for idx in stokes_idx:
            if idx < len(fields) and fields[idx].lower() not in ("null", "none", ""):
                # A non-numeric Stokes field is left exactly as written.
                with contextlib.suppress(ValueError):
                    fields[idx] = f"{float(fields[idx]) * factor:.8g}"
        lines[source.lineno] = (sky.delimiter or " ").join(fields)

    out_lines = [ln for i, ln in enumerate(lines) if i not in dropped]
    output = opts.output or ("corrected.txt" if invert else "apparent.txt")
    with open(output, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    log.info(
        "%s primary beam to %d sources -> %s",
        "Corrected" if invert else "Applied",
        len(sky.sources) - len(dropped),
        output,
    )


def provider_from(opts):
    from simms.skymodel.beams import resolve_beam

    return resolve_beam(opts.beam_pattern, opts.beam_band)
