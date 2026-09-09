"""Operations for the ``simms primary-beam`` command.

Four modes, none of which run a visibility simulation:

* ``to_fits``: sample an analytic cosine-taper beam onto a FITS beam cube, in either
  simms' own single-file layout or the Cattery/DDFacet 8-file ``--Beam-Model FITS`` schema.
* ``tag_ms``: write the per-antenna telescope-name column onto an existing MS.
* ``apply``/``correct``: multiply / divide a sky model (FITS image or ASCII components) by
  the parallactic-angle-averaged Stokes-I power beam ``A(l, m, nu)``. The beam narrows across
  the band, so this is not a scale factor: a cube gets a beam per plane, and ASCII components
  have it folded into their log-polynomial spectrum (:func:`fit_log_beam`). Only a model that
  cannot carry a spectrum -- a 2D image, a single-channel MS, a source schema without the
  continuum fields -- falls back to one frequency-averaged number. For a joint mosaic, the
  effective response is the weighted RMS ``sqrt(sum(w A**2) / sum(w))`` used by a standard
  flat-noise joint mosaic; pointings come from repeated MS inputs or explicit centres sharing
  one reference MS's observation metadata.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np

from simms import BIN
from simms.utilities import load_yaml

log = logging.getLogger(BIN.primary_beam)


# --------------------------------------------------------------------- geometry


def _observation(ms, field_id=0, spw_id=0, pointing_centre=None):
    """Read the geometry an averaged beam needs from an MS (for the given field/spw)."""
    import dask
    from daskms import xds_from_ms, xds_from_table

    from simms.skymodel.beams import array_lonlat, is_altaz_mount, read_pointing_centre, warn_unknown_mounts

    ant = xds_from_table(f"{ms}::ANTENNA")[0]
    spw = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    msds = xds_from_ms(ms, group_cols=["DATA_DESC_ID"], taql_where=f"FIELD_ID=={int(field_id)}")[int(spw_id)]
    if "MOUNT" not in ant:
        # Whether the beam rotates with parallactic angle is metadata, not something to
        # guess from: assuming alt-az smears a fixed beam, assuming fixed freezes a
        # rotating one, and both are silent. MOUNT is a required MSv2 ANTENNA column.
        raise RuntimeError(
            f"The ANTENNA table of {ms!r} has no MOUNT column, so whether the primary beam "
            f"rotates with parallactic angle cannot be determined. Add the column (MSv2 "
            f"requires it) with the mount of each antenna, e.g. 'ALT-AZ'."
        )
    pos, mount, t0, t1, interval, chan_freq = dask.compute(
        ant.POSITION.data,
        ant.MOUNT.data,
        msds.TIME.data.min(),
        msds.TIME.data.max(),
        msds.INTERVAL.data[0],
        spw.CHAN_FREQ.data[int(spw_id)],
    )
    lon, lat = array_lonlat(pos)
    # One representative beam is applied to the whole array here, so one mount decides
    # whether it rotates. Take the first antenna's, as resolve_antenna_beams does per type,
    # and say so when the array is actually mixed rather than deciding silently.
    mounts = [str(m) for m in np.asarray(mount).astype(str)]
    warn_unknown_mounts(mounts)
    is_altaz = is_altaz_mount(mounts[0]) if mounts else True
    if len({is_altaz_mount(m) for m in mounts}) > 1:
        log.warning(
            "ANTENNA.MOUNT mixes rotating and non-rotating mounts; treating the array as "
            "%s after the first antenna (%r).",
            "ALT-AZ" if is_altaz else "non-rotating",
            mounts[0],
        )
    if pointing_centre is None:
        field = xds_from_table(f"{ms}::FIELD")[0]
        (phase_dir,) = dask.compute(field.PHASE_DIR.data[int(field_id)])
        # Beam centre is the antenna pointing centre, not the phase centre. POINTING carries
        # no FIELD_ID, so the selected rows' TIME span is what picks this field's pointing.
        ra0, dec0 = read_pointing_centre(
            ms, phase_dir[0][0], phase_dir[0][1], int(field_id), time_range=(float(t0), float(t1))
        )
    else:
        # An explicit centre is authoritative. In particular, do not require a usable
        # POINTING table merely to read unrelated time/location/frequency metadata.
        ra0, dec0 = pointing_centre
    return {
        "t_start": float(t0),
        "duration": float(t1 - t0) + float(interval),
        "lon": lon,
        "lat": lat,
        "freqs": np.asarray(chan_freq, dtype=np.float64),
        "ra0": ra0,
        "dec0": dec0,
        "is_altaz": is_altaz,
    }


def _ms_paths(value):
    """Normalize a scalar or repeatable ``--ms`` value to a list."""
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


def _parse_pointing_centre(value):
    """Parse ``frame,ra,dec`` (or ``ra,dec``) and return J2000 radians."""
    from astropy.coordinates import Angle
    from casacore.measures import measures

    from simms.exceptions import InvalidInputError

    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) == 2:
        frame, ra, dec = "J2000", *parts
    elif len(parts) == 3:
        frame, ra, dec = parts
    else:
        raise InvalidInputError(
            f"--pointing-centre takes 'frame,ra,dec' (or 'ra,dec'), e.g. 'J2000,1h0m0s,-31deg'; got {value!r}."
        )
    try:
        dec_deg = float(Angle(dec).deg)
    except Exception:
        dec_deg = None
    if dec_deg is not None and abs(dec_deg) > 90:
        raise InvalidInputError(f"--pointing-centre declination must be within +/-90 degrees; got {dec!r}.")
    try:
        dm = measures()
        direction = dm.measure(dm.direction(frame, ra, dec), "J2000")
    except Exception as exc:
        raise InvalidInputError(f"Could not parse --pointing-centre {value!r}: {exc}") from None
    return float(direction["m0"]["value"]), float(direction["m1"]["value"])


def _observations(opts):
    """Observation metadata for every mosaic pointing requested by apply/correct."""
    from simms.exceptions import InvalidInputError

    paths = _ms_paths(opts.ms)
    centres = _ms_paths(opts.pointing_centre)
    if centres and len(paths) != 1:
        raise InvalidInputError(
            "--pointing-centre requires exactly one --ms whose time, location, mount and "
            "frequencies are reused for every explicit centre."
        )

    if centres:
        parsed = [_parse_pointing_centre(value) for value in centres]
        template = _observation(paths[0], opts.field_id, opts.spw_id, pointing_centre=parsed[0])
        observations = [template | {"ra0": ra0, "dec0": dec0} for ra0, dec0 in parsed]
    else:
        observations = [_observation(path, opts.field_id, opts.spw_id) for path in paths]

    supplied_weights = opts.mosaic_weight
    if supplied_weights is None:
        weights = [1.0] * len(observations)
    elif np.isscalar(supplied_weights):
        weights = [supplied_weights]
    else:
        weights = list(supplied_weights)
    if len(weights) != len(observations):
        raise InvalidInputError(
            f"Got {len(weights)} --mosaic-weight value(s) for {len(observations)} mosaic pointing(s); "
            "provide exactly one per --ms or --pointing-centre."
        )
    try:
        weights = [float(weight) for weight in weights]
    except (TypeError, ValueError):
        raise InvalidInputError("Every --mosaic-weight must be a number greater than zero.") from None
    if not all(np.isfinite(weight) and weight > 0 for weight in weights):
        raise InvalidInputError("Every --mosaic-weight must be finite and greater than zero.")
    # Only ratios matter. Scaling by the largest value prevents otherwise valid weights
    # such as [1e308, 1e308] from overflowing their sum or weighted beam accumulator.
    scale = max(weights)
    weights = [weight / scale for weight in weights]
    for obs, weight in zip(observations, weights, strict=True):
        obs["mosaic_weight"] = float(weight)

    if len(observations) > 1:
        reference = observations[0]["freqs"]
        for path, obs in zip(paths[1:], observations[1:], strict=False):
            if obs["freqs"].shape != reference.shape or not np.allclose(obs["freqs"], reference, rtol=1e-10, atol=1e-3):
                label = path if path else "an explicit pointing"
                raise InvalidInputError(
                    f"Mosaic pointing {label!r} has a different selected frequency grid. "
                    "All --ms values must select matching channels; use one reference --ms "
                    "with repeated --pointing-centre values when only the centres differ."
                )
        log.info(
            "Using the weighted PB-squared response of %d pointings for the joint mosaic%s.",
            len(observations),
            " (equal weights)" if supplied_weights is None else "",
        )
    return observations


def _beam_over_frequency(provider, ell, emm, ra0, dec0, obs, pa_step, freqs=None, moment=1):
    """PA-averaged power-beam moment, shape ``(npts, nfreq)`` (beam centre ra0/dec0).

    ``freqs`` defaults to the MS channel centres; the FITS-cube path passes the *cube's* own
    frequencies instead, since that is where its planes live.
    """
    from simms.skymodel.beams import image_power_beam, pa_sample_grid

    _, chi_grid = pa_sample_grid(obs["t_start"], obs["duration"], ra0, dec0, obs["lon"], obs["lat"], pa_step)
    freqs = obs["freqs"] if freqs is None else freqs
    return image_power_beam(provider, obs["is_altaz"], ell, emm, freqs, chi_grid, moment=moment)


def _averaged_beam(provider, ell, emm, ra0, dec0, obs, pa_step):
    """Freq- and PA-averaged power beam ``A(l, m)`` at the given directions (beam centre ra0/dec0)."""
    return _beam_over_frequency(provider, ell, emm, ra0, dec0, obs, pa_step).mean(axis=1)


def _mosaic_beam(provider, observations, lm_for_observation, pa_step, freqs=None):
    """Weighted-RMS power response over pointings, shaped ``(npts, nfreq)``.

    A standard linear mosaic has normal matrix ``sum(w_i <A_i(t)**2>_t)``. Dividing by
    ``sum(w_i)`` gives an effective response that is invariant to duplicating every
    pointing and reduces exactly to ``A`` for one pointing. Accumulate in place so a
    large FITS grid does not retain one full beam array per pointing.
    """
    sum_weighted_beam2 = None
    sum_weight = 0.0
    for obs in observations:
        ell, emm = lm_for_observation(obs)
        beam2 = _beam_over_frequency(
            provider,
            ell,
            emm,
            obs["ra0"],
            obs["dec0"],
            obs,
            pa_step,
            freqs=freqs,
            moment=2,
        )
        weight = obs["mosaic_weight"]
        if sum_weighted_beam2 is None:
            sum_weighted_beam2 = weight * beam2
        else:
            sum_weighted_beam2 += weight * beam2
        sum_weight += weight
    return np.sqrt(sum_weighted_beam2 / sum_weight)


def _effective_beam(provider, observations, lm_for_observation, pa_step, freqs=None):
    """Single-pointing apparent PB, or multi-pointing joint normal-matrix PB.

    The pre-mosaic CLI corrects an ordinary apparent image with ``<A>``. Preserve
    that contract for one pointing; ``sqrt(sum(w <A**2>) / sum(w))`` is specifically
    the PB-aware joint-mosaic response requested when multiple pointings are supplied.
    """
    if len(observations) == 1:
        obs = observations[0]
        ell, emm = lm_for_observation(obs)
        return _beam_over_frequency(provider, ell, emm, obs["ra0"], obs["dec0"], obs, pa_step, freqs=freqs)
    return _mosaic_beam(provider, observations, lm_for_observation, pa_step, freqs=freqs)


# The ASCII schema carries cont_coeff_1..3, so a refit can spend at most three coefficients.
MAX_SPECTRUM_ORDER = 3


def fit_log_beam(beam, freqs, ref_freq, order=MAX_SPECTRUM_ORDER):
    """Fit ``ln A(nu) = a0 + a1 x + ... + a_order x**order``, ``x = ln(nu / ref_freq)``.

    Returns ``(coeffs, max_fractional_residual)`` with ``coeffs`` shaped
    ``(order + 1, npts)``: ``a0`` is the beam at the reference frequency (in the log), and
    ``a1..a_order`` are exactly the log-polynomial coefficients a source spectrum already
    uses. That is what makes applying a beam to a spectral model a matter of *adding*
    coefficients rather than approximating: simms' continuum spectrum is
    ``S(nu) = S_ref (nu/nu_ref) ** (c1 + c2 x + ...)``, i.e. ``ln S = ln S_ref + sum c_k x**k``
    (:func:`simms.skymodel.source_factory.contspec`), and multiplying two such spectra about
    the same reference adds their coefficients term by term.

    The order is capped by the number of frequencies: a single-channel MS carries no spectral
    information at all, and then only ``a0`` is fitted -- a plain scale factor.
    """
    from simms.skymodel.fits_spectrum import _design_matrix

    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    beam = np.atleast_2d(np.asarray(beam, dtype=np.float64))  # (npts, nfreq)
    order = int(min(order, freqs.size - 1))

    # A zero (or negative, from a noisy FITS beam) sample has no logarithm. Clamping to a
    # tiny positive floor keeps the fit finite; such a point is far outside the beam and is
    # either dropped by --pb-cutoff or already negligible.
    ln_beam = np.log(np.maximum(beam, np.finfo(np.float64).tiny)).T  # (nfreq, npts)
    design = _design_matrix(freqs, ref_freq, order)
    coeffs, *_ = np.linalg.lstsq(design, ln_beam, rcond=None)

    residual = np.abs(design @ coeffs - ln_beam).max(initial=0.0)
    return coeffs, float(np.expm1(residual))


def _fits_spectral_axis(header, ndim):
    """``(numpy_axis, frequencies_hz)`` for a cube's spectral axis, or ``(None, None)``.

    ``None`` for a plain 2D image (which has no frequency to evaluate a beam at), for a
    degenerate one-plane axis, and for a velocity axis carrying no rest frequency to convert
    through -- the last of which is reported, since it silently costs accuracy.
    """
    from astropy import units
    from astropy.wcs import WCS

    wcs = WCS(header)
    fits_axis = wcs.wcs.spec
    if fits_axis < 0:
        return None, None
    # WCS axis order is the reverse of numpy's.
    axis = wcs.naxis - 1 - fits_axis
    nchan = int(header[f"NAXIS{fits_axis + 1}"])
    if nchan < 2 or axis >= ndim - 2:
        return None, None
    try:
        world = wcs.spectral.pixel_to_world(np.arange(nchan))
        freqs = np.atleast_1d(world.to_value(units.Hz, equivalencies=units.spectral()))
    except Exception as exc:
        log.warning(
            "Could not convert the spectral axis to frequency (%s); falling back to one "
            "band-averaged beam for every plane. A velocity axis needs a rest frequency (RESTFRQ).",
            exc,
        )
        return None, None
    return axis, freqs.astype(np.float64)


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
    """Write the beam as FITS and return ``(handle, files)``.

    ``files`` lists every file written. ``handle`` is what a dependent step chains onto:
    the single cube for ``--fits-format simms``, and the bare *prefix* for ``cattery``,
    which writes eight files -- the prefix is what both :meth:`FitsBeamProvider.from_cattery`
    and DDFacet's ``--Beam-FITSFile`` accept, so it round-trips into a later
    ``--beam-pattern``, while an individual one of the eight would not.
    """
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
        return prefix, list(paths)
    else:
        if opts.beam_l_axis != "-X" or opts.beam_m_axis != "Y":
            log.warning(
                "--beam-l-axis/--beam-m-axis only apply to --fits-format cattery; ignored for %r.", opts.fits_format
            )
        output = opts.output or "beam.fits"
        write_beam_fits(beam, grid, grid, freqs, output)
        log.info("Wrote beam FITS cube %s (%d x %d pixels, %d channels)", output, npix, npix, freqs.size)
        return output, [output]


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

    from simms.exceptions import InvalidInputError

    paths = _ms_paths(opts.ms)
    if len(paths) != 1:
        raise InvalidInputError("tag-ms requires exactly one --ms.")
    ms, col = paths[0], opts.telescope_name_column
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
    """Multiply (apply) or divide (correct) a FITS image by the averaged power beam.

    Returns the path actually written, which is the defaulted name when ``--output`` was omitted.
    """
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
    observations = _observations(opts)
    img_ra0 = np.radians(cel.wcs.crval[cel.wcs.lng])
    img_dec0 = np.radians(cel.wcs.crval[cel.wcs.lat])
    if len(observations) == 1:
        ra0, dec0 = observations[0]["ra0"], observations[0]["dec0"]
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

    def pixel_directions(obs):
        return pixel_lm(cel, obs["ra0"], obs["dec0"], i_ra.ravel(), j_dec.ravel())

    provider = provider_from(opts)

    # A cube's planes each sit at their own frequency, and the beam narrows across the band,
    # so one averaged map applied to every plane would impose the band-average attenuation on
    # channels where the true beam is far wider or narrower. Give each plane its own beam.
    spectral_axis, cube_freqs = _fits_spectral_axis(header, data.ndim)
    if spectral_axis is None:
        # With no spectral axis, approximate a continuum normal matrix with equal MS-channel
        # weights. The square root belongs outside that frequency average, just as it does
        # outside the PA average above.
        A_nu = _effective_beam(provider, observations, pixel_directions, opts.beam_pa_step)
        A = A_nu.mean(axis=1) if len(observations) == 1 else np.sqrt(np.mean(np.square(A_nu), axis=1))
        A = A.reshape(npix_dec, npix_ra)
    else:
        log.info(
            "Evaluating the beam per plane over the cube's %d channels (%.3f-%.3f GHz).",
            cube_freqs.size,
            cube_freqs.min() / 1e9,
            cube_freqs.max() / 1e9,
        )
        A = _effective_beam(provider, observations, pixel_directions, opts.beam_pa_step, freqs=cube_freqs)
        # (npix_dec * npix_ra, nchan) -> the cube's own axis order, singleton elsewhere. The
        # spectral axis always precedes both celestial axes, so no element reordering is needed.
        shape = [1] * data.ndim
        shape[-2], shape[-1], shape[spectral_axis] = npix_dec, npix_ra, cube_freqs.size
        A = A.reshape(npix_dec, npix_ra, cube_freqs.size).transpose(2, 0, 1).reshape(shape)

    if invert:
        safe = np.where(opts.pb_cutoff > A, np.nan, A)  # blank where the beam is negligible
        result = data / safe
    else:
        result = data * A

    output = opts.output or ("corrected.fits" if invert else "apparent.fits")
    fits.PrimaryHDU(data=result.astype(out_dtype, copy=False), header=header).writeto(output, overwrite=True)
    log.info("%s primary beam -> %s", "Corrected" if invert else "Applied", output)
    return output


def _schema_supports_continuum(sky, order):
    """Whether the model's schema can express a log-polynomial spectrum of this order.

    A custom ``--source-schema`` need not declare the continuum fields at all (the built-in
    one always does). Folding the beam into a spectrum those fields cannot hold would write a
    model the same schema can no longer read, so that case keeps the scalar behaviour.
    """
    params = getattr(sky.schema, "parameters", None)
    fields = ["cont_reffreq", *(f"cont_coeff_{k}" for k in range(1, order + 1))]
    return all(hasattr(params, field) for field in fields)


def _ascii_columns(lines, sky, order=None):
    """Column field names for the model, extended with any continuum columns a refit needs.

    Returns ``(header_line, fields_by_col, added)``. A model written without a spectrum has
    no column to hold the one the beam gives it, so the missing columns are appended rather
    than the spectral change being dropped. Where the file's schema renames a field, its own
    alias is reused so the result still parses under the same ``--source-schema``.
    """
    from simms.skymodel.ascii_skies import ASCIISource

    cols = lines[0].replace("#format:", "").strip().split(sky.delimiter)
    alias_to_field = ASCIISource(sky.schema).alias_to_field_mapper()
    fields_by_col = [alias_to_field.get(col, col) for col in cols]
    field_to_alias = {field: alias for alias, field in alias_to_field.items()}

    # order None: read the columns as they are, for the paths that write no spectrum.
    needed = [] if order is None else ["cont_reffreq", *(f"cont_coeff_{k}" for k in range(1, order + 1))]
    added = [field for field in needed if field not in fields_by_col]
    for field in added:
        cols.append(field_to_alias.get(field, field))
        fields_by_col.append(field)
    header = "#format: " + (sky.delimiter or " ").join(cols)
    return header, fields_by_col, added


def _set_field(fields, index, value):
    """Write ``value`` at ``index``, padding short lines so the column lands where it belongs."""
    fields.extend("0" for _ in range(index + 1 - len(fields)))
    fields[index] = value


def apply_correct_ascii(opts, invert):
    """Fold the beam into ASCII component spectra (apply), or divide it out (correct).

    The beam is not a scale factor: it narrows across the band, so a source off-axis is
    attenuated far more at the top of the band than the bottom, and the *spectrum* of the
    apparent source differs from the intrinsic one. At 0.5 degrees off-axis in MeerKAT
    L-band the beam alone contributes about -1.1 to the spectral index -- larger than a
    typical synchrotron index, so scaling the flux by a band-averaged number and leaving the
    index alone (which is what this used to do) misplaces the in-band flux by tens of
    percent.

    Both simms' continuum spectrum and the fitted beam are log-polynomials about the same
    reference frequency, so the fold is exact in that basis: the reference flux picks up
    ``exp(a0)`` and each ``cont_coeff_k`` picks up ``a_k`` (:func:`fit_log_beam`). ``correct``
    is the same with the beam's coefficients negated.

    Returns the path actually written, which is the defaulted name when ``--output`` was omitted.
    """
    from simms.skymodel.ascii_skies import ASCIISkymodel
    from simms.utilities import radec2lm

    observations = _observations(opts)
    # ASCIISkymodel falls back to the built-in source schema when source_schema is unset
    sky = ASCIISkymodel(opts.ascii_sky, delimiter=opts.ascii_delimiter, source_schema_file=opts.source_schema)

    def source_directions(obs):
        lm = np.array([radec2lm(obs["ra0"], obs["dec0"], s.ra, s.dec) for s in sky.sources])
        return (lm[:, 0], lm[:, 1]) if len(lm) else (np.array([]), np.array([]))

    freqs = observations[0]["freqs"]
    beam = _effective_beam(provider_from(opts), observations, source_directions, opts.beam_pa_step)

    # Sources are refit about their own reference frequency where they declare one, so the
    # flux column keeps meaning what it did; the rest share the band centre.
    band_centre = float(np.exp(np.mean(np.log(freqs))))
    order = int(min(MAX_SPECTRUM_ORDER, freqs.size - 1))

    # ASCIISkymodel is read-only, so we edit the fields in the original text (preserving
    # formatting, comments and unknown columns) rather than reserialising. Each parsed source
    # carries its line index (source.lineno) -- the single source of truth for which line it
    # came from -- so we never re-implement the comment/blank-line skipping here.
    with open(opts.ascii_sky) as fh:
        lines = fh.read().splitlines()
    refit = order >= 1 and _schema_supports_continuum(sky, order)
    if refit:
        header, fields_by_col, added = _ascii_columns(lines, sky, order)
        lines[0] = header
    else:
        _, fields_by_col, added = _ascii_columns(lines, sky)
        if order >= 1:
            log.warning(
                "The source schema in use declares no continuum fields, so the beam's frequency "
                "dependence cannot be written into the model; scaling the flux by the band-averaged "
                "beam instead. Predict with skysim --primary-beam to apply the beam per channel."
            )
    stokes_idx = [i for i, f in enumerate(fields_by_col) if f in ("stokes_i", "stokes_q", "stokes_u", "stokes_v")]
    reffreq_idx = fields_by_col.index("cont_reffreq") if refit else None
    coeff_idx = {k: fields_by_col.index(f"cont_coeff_{k}") for k in range(1, order + 1)} if refit else {}

    dropped = set()
    worst_residual = 0.0
    for src, source in enumerate(sky.sources):
        a_nu = beam[src]
        if invert and a_nu.min() < opts.pb_cutoff:
            # Dropped on the *weakest* channel: correcting divides by the beam, so a source
            # the beam nulls anywhere in the band cannot be recovered over the whole band.
            dropped.add(source.lineno)
            continue
        if refit:
            ref_freq = source.value_or_default("cont_reffreq") or band_centre
            coeffs, residual = fit_log_beam(a_nu, freqs, ref_freq, order)
            worst_residual = max(worst_residual, residual)
            coeffs = -coeffs[:, 0] if invert else coeffs[:, 0]
            scale = float(np.exp(coeffs[0]))
        else:
            a = float(a_nu.mean())
            scale = (1.0 / a) if invert else a

        fields = lines[source.lineno].split(sky.delimiter)
        for idx in stokes_idx:
            if idx < len(fields) and fields[idx].lower() not in ("null", "none", ""):
                # A non-numeric Stokes field is left exactly as written.
                with contextlib.suppress(ValueError):
                    fields[idx] = f"{float(fields[idx]) * scale:.8g}"
        if refit:
            old_coeffs = source.continuum_coefficients()
            _set_field(fields, reffreq_idx, f"{ref_freq:.8g}")
            for k, idx in coeff_idx.items():
                old = old_coeffs[k - 1] if k <= len(old_coeffs) else 0.0
                _set_field(fields, idx, f"{old + coeffs[k]:.8g}")
        lines[source.lineno] = (sky.delimiter or " ").join(fields)

    out_lines = [ln for i, ln in enumerate(lines) if i not in dropped]
    output = opts.output or ("corrected.txt" if invert else "apparent.txt")
    with open(output, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    if added:
        log.info("Added %s to the model; the beam gives every source a spectrum.", ", ".join(added))
    if order < MAX_SPECTRUM_ORDER:
        log.warning(
            "The MS has %d channel(s), so the beam's frequency dependence was fitted to order %d. "
            "A single channel carries none at all, leaving a plain scale factor.",
            freqs.size,
            order,
        )
    if worst_residual > 0.01:
        log.warning(
            "The order-%d log-polynomial reproduces the beam to only %.1f%% over the band for the "
            "worst source (typically one near a null). Predicting with skysim --primary-beam "
            "applies the beam per channel and needs no fit.",
            order,
            100 * worst_residual,
        )
    log.info(
        "%s primary beam to %d sources -> %s",
        "Corrected" if invert else "Applied",
        len(sky.sources) - len(dropped),
        output,
    )
    return output


def provider_from(opts):
    from simms.skymodel.beams import resolve_beam

    return resolve_beam(opts.beam_pattern, opts.beam_band)
