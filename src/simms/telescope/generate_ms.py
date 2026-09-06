import logging
import os
import shutil

import dask
import dask.array as da
import daskms
import numpy as np
from casacore.measures import measures
from daskms import xds_to_table
from scipy import interpolate
from tqdm.dask import TqdmCallback

from simms import BIN, PCKGDIR
from simms.constants import PI
from simms.exceptions import InvalidInputError
from simms.telescope import array_utilities as autils
from simms.utilities import get_noise, load_yaml

CORR_TYPES = load_yaml(f"{PCKGDIR}/telescope/ms_corr_types.yaml")["CORR_TYPES"]
# Direction reference frames casacore accepts, used only to tell 'frame,ra' (a dropped
# declination) apart from a frameless 'ra,dec'.
DIRECTION_FRAMES = {"J2000", "B1950", "ICRS", "GALACTIC", "SUPERGAL", "ECLIPTIC", "HADEC", "AZEL", "ITRF"}
dm = measures()


# Numpy does the correct type setting for strings
def nda(items):
    """Convert a list of items to a Dask array with appropriate dtype."""
    np_array = np.array(items)
    return da.asarray(np_array)


log = logging.getLogger(BIN.telsim)


def remove_ms(ms: str):
    if os.path.exists(ms):
        shutil.rmtree(ms, ignore_errors=True)
        log.debug(f"MS file {ms} exists. It will be overriden.")
    else:
        log.debug(f"MS file {ms} is being created.")


def _parse_dec_degrees(value: str):
    """The declination in degrees, or ``None`` when astropy cannot read the literal.

    casacore silently *wraps* an out-of-range declination -- ``-999d0m0s`` becomes +81d --
    so a typo produces a valid-looking MS pointed somewhere else entirely. astropy parses
    the sexagesimal spellings simms documents and reports the value as written, which is
    what makes the range check possible. casacore-only spellings it cannot read are left
    for casacore, hence the ``None``.
    """
    from astropy.coordinates import Angle

    try:
        return float(Angle(value).deg)
    except Exception:
        return None


def validate_ms_inputs(
    *,
    pointing_direction,
    dtime,
    ntimes,
    nchan,
    correlations,
    row_chunks,
    freq_range=None,
    low_source_limit=None,
    high_source_limit=None,
):
    """Reject option values that cannot produce a valid MS.

    Runs first thing in ``create_ms``, before any work and well before ``remove_ms``, because
    every one of these used to be accepted: ``--ntime 0``, ``--dtime -8`` and ``--nchan 0``
    each wrote an MS whose EXPOSURE or NUM_CHAN made it unusable -- and a ``--dtime -8`` MS
    then made a later ``skysim --sefd`` run fill DATA with NaNs, both steps exiting 0.
    """
    if nchan is not None and nchan < 1:
        raise InvalidInputError(f"--nchan must be at least 1, got {nchan}.")
    if ntimes < 1:
        raise InvalidInputError(f"--ntime must be at least 1, got {ntimes}.")
    if dtime <= 0:
        # A non-positive EXPOSURE/INTERVAL is written straight into the MS, where it turns
        # the thermal-noise 1/sqrt(2 dnu dt) into a NaN and the smearing kernels to nonsense.
        raise InvalidInputError(f"--dtime must be greater than 0 seconds, got {dtime}.")
    if row_chunks < 1:
        raise InvalidInputError(f"--row-chunks must be at least 1, got {row_chunks}.")

    if freq_range is not None:
        if len(freq_range) != 3:
            raise InvalidInputError(
                "--freq-range takes exactly start-freq,end-freq,nchan; "
                f"got {len(freq_range)} value(s): {','.join(str(v) for v in freq_range)}."
            )
        try:
            range_nchan = int(freq_range[2])
        except (TypeError, ValueError):
            raise InvalidInputError(f"--freq-range nchan must be an integer, got {freq_range[2]!r}.") from None
        if range_nchan < 2:
            # The channel width is (end - start) / (nchan - 1).
            raise InvalidInputError(f"--freq-range nchan must be at least 2, got {range_nchan}.")

    if not correlations or not all(str(corr).strip() for corr in correlations):
        raise InvalidInputError("--correlations must name at least one correlation, e.g. 'XX,YY'.")
    unknown = [str(corr) for corr in correlations if str(corr).strip() not in CORR_TYPES]
    if unknown:
        raise InvalidInputError(
            f"--correlations has unknown correlation(s) {','.join(unknown)}. "
            f"Valid values: {', '.join(sorted(CORR_TYPES))}."
        )
    if len(correlations) not in (1, 2, 4):
        raise InvalidInputError(f"--correlations must give 1, 2 or 4 correlations, got {len(correlations)}.")

    if len(pointing_direction) not in (2, 3):
        raise InvalidInputError(
            "--direction takes 'frame,ra,dec' (or 'ra,dec'), e.g. 'J2000,0h24m20s,-30d12m33s'; "
            f"got {len(pointing_direction)} comma-separated value(s)."
        )
    if len(pointing_direction) == 2 and pointing_direction[0].strip().upper() in DIRECTION_FRAMES:
        # Two values are read as 'ra,dec' with the frame defaulted, so a dropped declination
        # after an explicit frame ('J2000,1h0m0s') otherwise reaches casacore as an RA of
        # "J2000" and fails somewhere unrecognisable.
        raise InvalidInputError(
            f"--direction gives the reference frame {pointing_direction[0]!r} but only one "
            f"coordinate; it needs 'frame,ra,dec', e.g. 'J2000,0h24m20s,-30d12m33s'."
        )
    dec_deg = _parse_dec_degrees(pointing_direction[-1])
    if dec_deg is not None and abs(dec_deg) > 90:
        raise InvalidInputError(
            f"--direction declination must be within +/-90 degrees, got {pointing_direction[-1]!r} "
            f"({dec_deg:g} deg). casacore would silently wrap this to a different part of the sky."
        )

    if (
        low_source_limit is not None
        and high_source_limit is not None
        and float(low_source_limit) >= float(high_source_limit)
    ):
        raise InvalidInputError(
            f"--low-source-limit ({low_source_limit}) must be below --high-source-limit "
            f"({high_source_limit}); as given, every row is flagged."
        )


def validate_frequencies(start_freq, dfreq, nchan):
    """Reject parsed frequencies that cannot describe a spectral window.

    Runs on the values *after* unit parsing, so it catches both ``--start-freq -1420MHz``
    and a ``--freq-range`` that resolves to the same thing. A negative ``dfreq`` is left
    alone: a reversed sideband is a real spectral window, and the smearing kernels are
    written against a signed channel width.
    """
    if start_freq <= 0:
        raise InvalidInputError(f"--start-freq must be a positive frequency, got {start_freq:g} Hz.")
    if nchan > 1 and dfreq == 0:
        raise InvalidInputError(
            f"--chan-width must be non-zero for a {nchan}-channel MS; every channel would sit at {start_freq:g} Hz."
        )


def create_ms(
    ms: str,
    telescope_name: str,
    pointing_direction: str,
    dtime: int,
    ntimes: int,
    start_freq: str | float,
    dfreq: str | float,
    nchan: int,
    correlations: str,
    row_chunks: int,
    sefd: float,
    column: str,
    smooth: str | None = None,
    fit_order: int | None = None,
    start_time: str | list[str] | None = None,
    start_ha: float | None = None,
    freq_range: str | None = None,
    sfile: str | None = None,
    tsys_over_eta: float | None = None,
    subarray_list: list[str] | None = None,
    subarray_range: list[int] | None = None,
    subarray_file: str | None = None,
    low_source_limit: float | str | None = None,
    high_source_limit: float | str | None = None,
    telescope_name_column: str = "TELESCOPE_NAME",
):
    """Generate a CASA Measurement Set (MS) for a simulated observation.

    Parameters
    ----------
    ms : str
        Path to output Measurement Set directory (will be overwritten if it exists).
    telescope_name : Union[str, File]
        Telescope or array definition (name or configuration file).
    pointing_direction : str
        Target sky direction (e.g. 'J2000 12:34:56.7 -45.23.10').
    dtime : int
        Integration time per visibility (seconds).
    ntimes : int
        Number of time steps.
    start_freq : Union[str, float]
        Starting frequency (string with units or raw Hz).
    dfreq : Union[str, float]
        Channel frequency increment (string with units or raw Hz).
    nchan : int
        Number of spectral channels.
    correlations : str
        Correlation types (e.g. 'XX', 'YY', 'RR', 'LL').
    row_chunks : int
        Dask chunk size along the row (time-baseline) dimension.
    sefd : float
        System Equivalent Flux Density (Jy). If provided, noise is generated.
    column : str
        Name of data column to populate (e.g. 'DATA' or 'CORRECTED_DATA').
    smooth : str, optional
        SEFD frequency smoothing method ('polyn' or 'spline').
    fit_order : int, optional
        Polynomial degree or spline smoothing parameter.
    start_time : Union[str, List[str]], optional
        Observation start time (string or list for multi-part definitions).
    start_ha : float, optional
        Starting hour angle (radians).
    freq_range : str, optional
        Alternative frequency specification: [start, end, nchan].
    sfile : File, optional
        Sensitivity file to override SEFD values.
    tsys_over_eta : float, optional
        Tsys/eta value used when deriving SEFD if not explicitly given.
    subarray_list : List[str], optional
        List of antenna names to include.
    subarray_range : List[int], optional
        Index range of antennas to include.
    subarray_file : File, optional
        File specifying subarray selection.
    low_source_limit : Union[float, str], optional
        Elevation (deg) below which rows are flagged.
    high_source_limit : Union[float, str], optional
        Elevation (deg) above which rows are flagged.

    Returns
    -------
    None
        Writes the Measurement Set and its subtables to disk.
    """
    # Validate before remove_ms: a rejected run must not delete the MS already sitting at
    # the target path (a --telescope typo used to do exactly that).
    validate_ms_inputs(
        pointing_direction=pointing_direction,
        dtime=dtime,
        ntimes=ntimes,
        nchan=nchan,
        correlations=correlations,
        row_chunks=row_chunks,
        freq_range=freq_range,
        low_source_limit=low_source_limit,
        high_source_limit=high_source_limit,
    )

    telescope_array = autils.Array(
        telescope_name,
        sefd=sefd,
        tsys_over_eta=tsys_over_eta,
        sensitivity_file=sfile,
        subarray_list=subarray_list,
        subarray_range=subarray_range,
        subarray_file=subarray_file,
    )

    size = telescope_array.size
    mount = telescope_array.mount
    antnames = telescope_array.names
    antlocation = telescope_array.antlocations
    ant_telescope_names = telescope_array.telescope_name

    if freq_range:
        start_freq = parse_frequency(freq_range[0], "freq-range start-freq")
        end_freq = parse_frequency(freq_range[1], "freq-range end-freq")
        nchan = int(freq_range[2])
        dfreq = (end_freq - start_freq) / (nchan - 1)
    else:
        start_freq = parse_frequency(start_freq, "start-freq")
        dfreq = parse_frequency(dfreq, "dfreq")
        end_freq = start_freq + dfreq * (nchan - 1)

    validate_frequencies(start_freq, dfreq, nchan)

    freqs = np.linspace(start_freq, end_freq, nchan)

    uvcoverage_data = telescope_array.uvgen(
        pointing_direction,
        dtime,
        ntimes,
        "0Hz",
        "0Hz",
        nchan,
        start_time,
        start_ha,
    )

    num_rows = uvcoverage_data.times.shape[0]
    num_row_chunks = row_chunks

    num_chans = len(freqs)
    num_corr = len(correlations)
    corr_types = np.array([[CORR_TYPES[x] for x in correlations]])
    # TODO(sphe) use casacore to determine this
    corr_products = np.array([([0, 0], [1, 1])]) if num_corr == 2 else np.array([([0, 0], [0, 1], [1, 0], [1, 1])])

    num_ants = antlocation.shape[0]
    ant1 = uvcoverage_data.antenna1 * ntimes
    ant2 = uvcoverage_data.antenna2 * ntimes

    if len(pointing_direction) == 3:
        ra_dec = dm.direction(rf=pointing_direction[0], v0=pointing_direction[1], v1=pointing_direction[2])
    else:
        log.warning("Option --direction  not given reference frame. Assuming J2000.")
        ra_dec = dm.direction(rf="J2000", v0=pointing_direction[0], v1=pointing_direction[1])

    ra = ra_dec["m0"]["value"]
    dec = ra_dec["m1"]["value"]

    phase_dir = np.array([[[ra, dec]]])

    data = da.zeros((num_rows, num_chans, num_corr), chunks=(num_row_chunks, num_chans, num_corr))
    times = da.from_array(uvcoverage_data.times, chunks=num_row_chunks)
    time_range = np.array([[uvcoverage_data.times[0], uvcoverage_data.times[-1]]])
    uvw = da.from_array(uvcoverage_data.uvw, chunks=(num_row_chunks, 3))
    antenna1 = da.from_array(ant1, chunks=num_row_chunks)
    antenna2 = da.from_array(ant2, chunks=num_row_chunks)
    interval = da.full(num_rows, dtime, chunks=num_row_chunks)
    sigma = da.full((num_rows, num_corr), 1.0, chunks=(num_row_chunks, num_corr))
    sigma_spec = da.full((num_rows, num_chans, num_corr), 1.0, chunks=(num_row_chunks, num_chans, num_corr))
    flag = da.zeros(
        (num_rows, num_chans, num_corr),
        dtype=bool,
        chunks=(num_row_chunks, num_chans, num_corr),
    )

    freqs = freqs.reshape(1, freqs.shape[0])
    channel_widths = da.full(freqs.shape, dfreq)
    total_bandwidth = nchan * dfreq

    # Only now, with the layout resolved, the subarray selected, the frequencies parsed and
    # the uv coverage computed, is the existing MS at this path removed. Doing it on entry
    # meant a typo in --telescope (or any other late failure) destroyed the MS already there
    # and then exited with an error, leaving nothing behind.
    remove_ms(ms)

    ds = {
        "DATA": (("row", "chan", "corr"), data),
        "UVW": (("row", "uvw_dim"), uvw),
        "TIME": (("row"), times),
        "TIME_CENTROID": (("row"), times),
        "INTERVAL": (("row"), interval),
        "EXPOSURE": (("row"), interval),
        "ANTENNA1": (("row"), antenna1),
        "ANTENNA2": (("row"), antenna2),
        "SIGMA": (("row", "corr"), sigma),
        "WEIGHT": (("row", "corr"), sigma),
        "SIGMA_SPECTRUM": (("row", "chan", "corr"), sigma_spec),
        "WEIGHT_SPECTRUM": (("row", "chan", "corr"), sigma_spec),
        "FLAG": (("row", "chan", "corr"), flag),
        "FLAG_CATEGORY": (("row", "flagcat", "chan", "corr"), flag[:, None, :, :]),
    }

    dd_id = 0
    ddid = da.rechunk(da.full(num_rows, dd_id), chunks=num_row_chunks)

    ds["DATA_DESC_ID"] = ("row",), ddid

    field_id = 0
    ds["FIELD_ID"] = ("row"), da.full_like(ddid, fill_value=field_id)

    scan_number = 1
    ds["SCAN_NUMBER"] = ("row"), da.full_like(ddid, fill_value=scan_number)

    state_id = 0
    ds["STATE_ID"] = ("row"), da.full_like(ddid, fill_value=state_id)

    array_id = 0
    ds["ARRAY_ID"] = ("row"), da.full_like(ddid, fill_value=array_id)

    obs_id = 0
    ds["OBSERVATION_ID"] = ("row"), da.full_like(ddid, fill_value=obs_id)

    proc_id = 0
    ds["PROCESSOR_ID"] = ("row"), da.full_like(ddid, fill_value=proc_id)

    nbaselines = num_ants * (num_ants - 1) // 2

    sefd = telescope_array.sefd
    noise_freqs = telescope_array.noise_freqs

    if sefd:
        dummy_data = da.random.standard_normal(
            (num_rows, num_chans, num_corr), chunks=(num_row_chunks, num_chans, num_corr)
        ) + 1j * da.random.standard_normal(
            (num_rows, num_chans, num_corr), chunks=(num_row_chunks, num_chans, num_corr)
        )

        # Lifted from https://github.com/shinobi-dosho/msutils/blob/v1.2.0/MSUtils/ClassESW.py
        if noise_freqs:

            def freqs_hz(freqs):
                freqs_hz = []
                for i in np.array(freqs).flatten():
                    freqs_hz.append(parse_frequency(i, "frequencies for SEFD"))
                return freqs_hz

            x = freqs_hz(noise_freqs)
            y = sefd

            ms_freqs = freqs_hz(freqs)

            if smooth == "polyn":
                fit_parms = np.polyfit(x, y, fit_order)
                fit_func = np.poly1d(fit_parms)
            elif smooth == "spline":
                sorted_x, sorted_y = zip(*sorted(zip(x, y, strict=True)), strict=True)
                fit_parms = interpolate.splrep(sorted_x, sorted_y, s=fit_order)

                def fit_func(freqs):
                    return interpolate.splev(freqs, fit_parms, der=0)

            sefd_chan = fit_func(ms_freqs)
            noise_chan = sefd_chan / np.sqrt(2 * dfreq * dtime)
            if len(noise_chan.shape) > 1:
                noise_chan = noise_chan[0]
            noisy_data = da.array(dummy_data * noise_chan[None, :, None], like=data).rechunk(chunks=data.chunks)

        elif not noise_freqs:
            noise = get_noise(sefd, dtime, dfreq)

            if isinstance(noise, float | int):
                noisy_data = da.array(dummy_data * noise, like=data).rechunk(chunks=data.chunks)

            else:
                dummy_data_resh = dummy_data.reshape((ntimes, nbaselines, num_chans, num_corr))
                noisy_dummy_data = dummy_data_resh * np.array(noise)[None, :, None, None]
                noisy_result = noisy_dummy_data.reshape((ntimes * nbaselines, num_chans, num_corr))
                noisy_data = da.array(noisy_result, like=data).rechunk(chunks=data.chunks)

        ds[column] = (("row", "chan", "corr"), noisy_data)

    src_elevs = uvcoverage_data.source_elevations
    expanded_src_elevations = np.repeat(src_elevs, nbaselines)

    flag_row = np.zeros(num_rows, dtype=bool)

    if low_source_limit:
        flag_row |= expanded_src_elevations < low_source_limit

    if high_source_limit:
        flag_row |= expanded_src_elevations > high_source_limit

    ds["FLAG_ROW"] = (("row",), da.from_array(flag_row, chunks=num_row_chunks))

    main_table = daskms.Dataset(ds, coords={"ROWID": ("row", da.arange(num_rows))})

    write_main = xds_to_table(main_table, ms, columns="ALL", descriptor="ms(False)")
    with TqdmCallback(desc=f"Writing the Main Table to {ms}"):
        dask.compute(write_main)

    pol_response = da.array([[[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]])

    receptor_angles = np.tile([0, 0.5 * PI], (num_ants, 1))

    feed_ds = {
        "ANTENNA_ID": (("row"), da.arange(num_ants)),
        "BEAM_ID": (("row"), da.from_array(np.full(num_ants, -1))),
        "BEAM_OFFSET": (
            ("row", "receptors", "radec"),
            da.from_array(np.zeros((num_ants, 2, 2), dtype=float)),
        ),
        "INTERVAL": (("row"), da.from_array(np.full(num_ants, 1.0e30))),
        "NUM_RECEPTORS": (("row"), da.from_array(np.full(num_ants, 2))),
        "SPECTRAL_WINDOW_ID": (("row"), da.from_array(np.full(num_ants, -1))),
        # TODO: @Allycan Account for circular polarisation
        "POLARIZATION_TYPE": (
            ("row", "receptors"),
            da.from_array(np.full((num_ants, 2), (["X", "Y"]))),
        ),
        "RECEPTOR_ANGLE": (
            ("row", "receptors"),
            da.from_array(receptor_angles),
        ),
        "POL_RESPONSE": (
            ("row", "receptors", "receptors-2"),
            da.from_array(np.full((num_ants, 2, 2), pol_response)),
        ),
    }

    feed_table = daskms.Dataset(feed_ds)
    ms_desc = "ms_subtable(True)"
    write_feed = xds_to_table(feed_table, f"{ms}::FEED", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the FEED table to {ms}"):
        dask.compute(write_feed)

    spw_ds = {
        "CHAN_FREQ": (("row", "chan"), da.from_array(freqs)),
        "CHAN_WIDTH": (("row", "chan"), channel_widths),
        "EFFECTIVE_BW": (("row", "chan"), channel_widths),
        "RESOLUTION": (("row", "chan"), channel_widths),
        "REF_FREQ": (("row"), da.from_array([start_freq])),
        "MEAS_RES_FREQ": (("row"), da.from_array([start_freq])),
        "TOTAL_BANDWIDTH": (("row"), da.from_array([total_bandwidth])),
        "NUM_CHAN": (("row"), da.from_array([nchan])),
        "NAME": (("row"), da.from_array(["00"])),
        "NET_SIDEBAND": (("row"), da.array([1])),
        "FREQ_GROUP_NAME": (("row"), da.from_array(["GROUP 1"])),
        # Frequency measure reference: 5 == TOPO (topocentric). Without this casacore
        # defaults it to 0 (REST), which leaves the spectral frame undefined and makes CASA
        # imaging fail ("No MeasFrame specified for conversion of Frequency").
        "MEAS_FREQ_REF": (("row"), da.array([5])),
    }

    spw_table = daskms.Dataset(spw_ds)

    write_spw = xds_to_table(spw_table, f"{ms}::SPECTRAL_WINDOW")
    with TqdmCallback(desc=f"Writing the SPECTRAL_WINDOW table to {ms}"):
        dask.compute(write_spw)

    dish_diameter = [size] * num_ants if isinstance(size, int | float) else np.array(size)

    ant_mount = [mount] * num_ants if isinstance(mount, str) else np.array(mount)

    names = np.array(antnames)
    teltype = ["GROUND_BASED"] * num_ants
    if isinstance(ant_telescope_names, str):
        telescope_names = [ant_telescope_names] * num_ants
    else:
        telescope_names = list(ant_telescope_names)
    # casacore STRING columns map to numpy object dtype (not fixed-width unicode).
    telescope_names = np.array(telescope_names, dtype=object)
    ant_ds = {
        "DISH_DIAMETER": (("row"), da.from_array(dish_diameter)),
        "MOUNT": (("row"), da.from_array(ant_mount)),
        "POSITION": (("row", "xyz"), da.from_array(antlocation)),
        "NAME": (("row"), da.from_array(names)),
        "STATION": (("row"), da.from_array(names)),
        "TYPE": (("row"), da.from_array(teltype)),
        # Per-antenna telescope/type label used by skysim to select a primary beam
        # (heterogeneous arrays). Non-standard ANTENNA column; the name is configurable
        # (default TELESCOPE_NAME) because it is still a provisional spec decision.
        telescope_name_column: (("row"), da.from_array(telescope_names, chunks=num_ants)),
    }

    ant_table = daskms.Dataset(ant_ds)

    # ANTENNA carries the non-standard TELESCOPE_NAME column, so it needs a valid
    # subtable descriptor builder (the shared ms_desc string is only tolerated for
    # all-standard subtables, where the column descriptors are never built).
    write_ant = xds_to_table(ant_table, f"{ms}::ANTENNA", descriptor="mssubtable('ANTENNA')")
    with TqdmCallback(desc=f"Writing the ANTENNA table to {ms}"):
        dask.compute(write_ant)

    field_name = "TARGET"
    ftimes = [0.0]

    fld_ds = {
        "NAME": (("row"), nda([field_name])),
        "PHASE_DIR": (("row", "field-poly", "field-dir"), da.from_array(phase_dir)),
        "DELAY_DIR": (("row", "field-poly", "field-dir"), da.from_array(phase_dir)),
        "REFERENCE_DIR": (("row", "field-poly", "field-dir"), da.from_array(phase_dir)),
        "TIME": (("row"), da.from_array(ftimes)),
        "SOURCE_ID": (("row"), da.from_array([field_id])),
    }

    fld_table = daskms.Dataset(fld_ds)

    write_fld = xds_to_table(fld_table, f"{ms}::FIELD", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the FIELD table to {ms}"):
        dask.compute(write_fld)

    obs_ds = {
        "TIME_RANGE": (("row", "obs-exts"), da.from_array(time_range)),
        "OBSERVER": (("row"), da.from_array(np.array(["simms simulator"]))),
        "PROJECT": (("row"), da.from_array(np.array(["simms simulation"]))),
        "TELESCOPE_NAME": (("row"), da.from_array(np.array([telescope_name]))),
    }

    obs_table = daskms.Dataset(obs_ds)

    write_obs = xds_to_table(obs_table, f"{ms}::OBSERVATION", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the OBSERVATION table to {ms}"):
        dask.compute(write_obs)

    pol_ds = {
        "NUM_CORR": (("row"), da.from_array(np.array([num_corr]))),
        "CORR_PRODUCT": (("row", "corr", "corrprod_idx"), da.from_array(corr_products)),
        "CORR_TYPE": (("row", "corr"), da.from_array(corr_types)),
    }

    pol_table = daskms.Dataset(pol_ds)

    write_pol = xds_to_table(pol_table, f"{ms}::POLARIZATION", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the POLARIZATION table to {ms}"):
        dask.compute(write_pol)

    # POINTING is keyed by (ANTENNA_ID, TIME) -- one row per antenna per timeslot, not one
    # row per main-table row. Writing num_rows rows would repeat each pointing nbl/nant times
    # over and drop every antenna but the first, which is what makes the beam centre in
    # POINTING.DIRECTION readable for antenna 0 alone.
    num_pntng_rows = num_ants * ntimes
    pntng_chunks = min(row_chunks, num_pntng_rows)
    # Time-major, matching the main table: all antennas at t0, then all antennas at t1, ...
    pntng_ant_id = np.tile(np.arange(num_ants), ntimes)
    pntng_times = np.repeat(uvcoverage_data.time_entries, num_ants)

    phase_dir_small = da.from_array(phase_dir, chunks=-1)
    phase_arr = da.broadcast_to(phase_dir_small, (num_pntng_rows, 1, 2), chunks=(pntng_chunks, 1, 2))

    pntng_ds = {
        "ANTENNA_ID": (("row",), da.from_array(pntng_ant_id, chunks=pntng_chunks)),
        "TIME": (("row",), da.from_array(pntng_times, chunks=pntng_chunks)),
        "INTERVAL": (("row",), da.full(num_pntng_rows, dtime, chunks=pntng_chunks)),
        "TRACKING": (("row",), da.full(num_pntng_rows, True, chunks=pntng_chunks, dtype=bool)),
        "DIRECTION": (("row", "point-poly", "radec"), phase_arr),
        # TARGET is a required MSv2 column; with no pointing offsets it equals DIRECTION.
        "TARGET": (("row", "point-poly", "radec"), phase_arr),
    }

    pntng_table = daskms.Dataset(pntng_ds)

    write_pntng = xds_to_table(pntng_table, f"{ms}::POINTING", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the POINTING table to {ms}"):
        dask.compute(
            write_pntng,
        )

    # add PROCESSOR table
    processor_table = daskms.Dataset(
        {
            "TYPE": (("row",), nda(["CORRELATOR"])),
            "SUB_TYPE": (("row",), nda(["UNSET"])),
            "TYPE_ID": (("row",), nda([proc_id])),
            "MODE_ID": (("row",), nda([proc_id])),
            "FLAG_ROW": (("row",), nda([False])),
        }
    )

    with TqdmCallback(desc=f"Writing the PROCESSOR table to {ms}"):
        dask.compute(
            xds_to_table(processor_table, f"{ms}::PROCESSOR"),
        )

    # add DATA_DESC table
    datadesc_table = daskms.Dataset(
        {
            "SPECTRAL_WINDOW_ID": (("row",), da.array([dd_id])),
            "POLARIZATION_ID": (("row",), da.array([dd_id])),
            "LAG_ID": (("row",), da.array([0])),
            "FLAG_ROW": (("row",), da.array([False])),
        }
    )

    with TqdmCallback(desc=f"Writing the DATA_DESCRIPTION table to {ms}"):
        dask.compute(
            xds_to_table(datadesc_table, f"{ms}::DATA_DESCRIPTION"),
        )

    # add state table
    state_table = daskms.Dataset(
        {
            "SIG": (("row",), da.array([True])),
            "REF": (("row",), da.array([False])),
            "CAL": (("row",), da.array([0.0])),
            "LOAD": (("row",), da.array([0.0])),
            "SUB_SCAN": (("row",), da.array([state_id])),
            "OBS_MODE": (("row",), nda(["OBSERVE_TARGET.ON_SOURCE"])),
            "FLAG_ROW": (("row",), da.array([False], dtype=bool)),
        }
    )

    with TqdmCallback(desc=f"Writing the STATE table to {ms}"):
        dask.compute(
            xds_to_table(state_table, f"{ms}::STATE", descriptor=ms_desc),
        )

    log.info(f"{ms} successfully generated.")


def parse_frequency(value, option: str):
    """Parses a frequency value which may be a string with units or a float without units.

    Parameters
    -----------
    value: Union[str, float, int]
        The frequency value to parse.
    """
    dm = measures()

    if isinstance(value, str):
        if "Hz" in value:
            freq = dm.frequency(v0=value)["m0"]["value"]
        else:
            log.warning(f"Option --{option} given without units, assuming canonical units 'Hz'.")
            freq = dm.frequency(v0=f"{value}Hz")["m0"]["value"]
    else:
        freq = dm.frequency(v0=f"{value}Hz")["m0"]["value"]

    return freq
