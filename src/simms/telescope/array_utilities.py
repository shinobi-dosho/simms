import logging

import astropy.units as u
import numpy as np
import yaml
from astropy.constants import k_B
from astropy.time import Time
from casacore.measures import measures

from simms import BIN, constants
from simms.telescope.layouts import custom_telescopes, resolve_layout
from simms.utilities import ObjDict, load_yaml

log = logging.getLogger(BIN.telsim)


class Array:
    """
    The Array class has functions for converting from one coordinate system to another.
    """

    def __init__(
        self,
        layout: str,
        degrees: bool = True,
        sefd: int | float | list[int | float] | None = None,
        tsys_over_eta: int | float | list[int | float] | None = None,
        sensitivity_file: str | None = None,
        subarray_list: list[str] | None = None,
        subarray_range: list[int] | None = None,
        subarray_file: str | None = None,
    ):
        """
        Initialize the Array.

        Parameters
        ----
        layout: Union[str, File]
            Observatory name or path to a layout configuration (YAML). If "vla", defaults to "vla-c".
        degrees: bool, optional
            Whether geodetic long/lat are provided in degrees. Default is True.
        sefd: int | float | List[int | float], optional
            System Equivalent Flux Density per antenna (Jy). If a scalar, applied to all antennas.
            Overrides tsys_over_eta if provided. Default is None.
        tsys_over_eta: int | float | List[int | float], optional
            Tsys/eta per antenna (K). Used to compute SEFD if sefd is not provided. Default is None.
        sensitivity_file: File, optional
            Path to a YAML file containing "sefd" and/or "tsys_over_eta" (and optional "freq"/"freqs")
            to override values from the layout. Default is None.
        subarray_list: List[str], optional
            Explicit list of antenna/station names to include in a custom subarray. Default is None.
        subarray_range: List[int], optional
            Range [start, stop] of antenna indices to include in a custom subarray. Default is None.
        subarray_file: File, optional
            Path to a file describing a custom subarray selection. Default is None.
        """

        self.degrees = degrees
        if layout.lower() == "vla":
            log.warning("VLA configuration not specified. Will default to the C configuration (vla-c).")
            layout = "vla-c"

        self.layout, layout_file = resolve_layout(layout)
        # A user-supplied layout file need not declare the array centre; derive it and cache it
        # back into the file. Built-in entries resolve to layout_file=None and are left alone.
        if layout_file is not None and "centre" not in self.layout:
            centre = calculate_array_centre(self.layout["antlocations"])
            write_centre_to_array_config(layout_file, centre)
            self.layout["centre"] = centre

        if subarray_list:
            self.layout = custom_telescopes(layout=layout, subarray_list=subarray_list)
        elif subarray_range:
            self.layout = custom_telescopes(layout=layout, subarray_range=subarray_range)
        elif subarray_file:
            self.layout = custom_telescopes(layout=layout, subarray_file=subarray_file)

        self.sefd = sefd
        self.tsys_over_eta = tsys_over_eta
        self.sensitivity_file = sensitivity_file
        self.noise_freqs = None

        self._set_arrayinfo()
        if self.layout["coord_sys"] == "geodetic":
            self.set_itrf()

    def _set_arrayinfo(self):
        """
        - Extract the array information from the schema
        - Set values to SI units
        """

        centre = self.layout["centre"]
        xyz = np.array(self.layout["antlocations"])
        nant, ndim = xyz.shape
        self.nant = nant

        self.alt0 = 0 if len(centre) == 2 else centre[2]

        if self.layout["coord_sys"] == "geodetic":
            if ndim == 2:
                xyz = np.hstack(xyz, np.zeros(nant))
            if self.degrees:
                xyz[:, 0] = np.deg2rad(xyz[:, 0])
                xyz[:, 1] = np.deg2rad(xyz[:, 1])

                self.lon0 = np.deg2rad(centre[0])
                self.lat0 = np.deg2rad(centre[1])
            else:
                self.lon0 = centre[0]
                self.lat0 = centre[1]

            self.geodtic_antennas = xyz

        elif self.layout["coord_sys"] == "itrf":
            if ndim == 2:
                raise RuntimeError("Input antenna ITRF coordinates have 2 coordinates. Three (X,Y,Z) are required.")
            self.lon0, self.lat0, self.alt0 = self.global2geodetic(*centre)
            self.x0, self.y0, self.z0 = centre
            self.antlocations = xyz
        else:
            raise ValueError("Unknown coordinate system. Please use Geodetic (WGS84) or ITRF (XYZ)")

        self.mount = self.layout["mount"]
        self.names = self.layout["antnames"]
        self.coordsys = self.layout["coord_sys"]
        self.size = self.layout["size"]

        # Per-antenna telescope/type label for the ANTENNA table (heterogeneous
        # arrays), defaulting to the array name. Normalised to a per-antenna list.
        tel_name = self.layout.get("telescope_name", "") or self.layout.get("name", "")
        if isinstance(tel_name, str):
            self.telescope_name = [tel_name] * nant
        else:
            self.telescope_name = list(tel_name)

        if self.sefd is None:
            sefd = self.layout.get("sefd", None)
            self.sefd = sefd
            if self.sensitivity_file:
                sensitivity_data = load_yaml(self.sensitivity_file)
                if "sefd" in sensitivity_data:
                    self.sefd = sensitivity_data["sefd"]

                if "freq" in sensitivity_data:
                    self.noise_freqs = sensitivity_data["freq"]
                if "freqs" in sensitivity_data:
                    self.noise_freqs = sensitivity_data["freqs"]

            if isinstance(sefd, float | int):
                self.sefd = [sefd]
            elif (not isinstance(sefd, str)) and isinstance(sefd, list):
                self.sefd = sefd

        if self.tsys_over_eta and self.sefd is None:
            self.sefd = 2 * k_B.value * self.tsys_over_eta / (np.pi * np.array(self.size) ** 2 / 4)

        elif self.tsys_over_eta is None and self.sefd is None:
            tsys_over_eta = self.layout.get("tsys_over_eta", None)
            self.tsys_over_eta = tsys_over_eta
            if self.sensitivity_file and "tsys_over_eta" in sensitivity_data and "sefd" not in sensitivity_data:
                self.tsys_over_eta = sensitivity_data["tsys_over_eta"]
                self.sefd = list(2 * k_B.value * self.tsys_over_eta / (np.pi * np.array(self.size) ** 2 / 4))

            if isinstance(sefd, float | int):
                self.sefd = [sefd]
            elif (not isinstance(sefd, str)) and isinstance(sefd, list):
                self.sefd = sefd

    def set_itrf(self):
        if not hasattr(self, "antlocations"):
            self.antlocations, xyz0 = self.geodetic2global()
            self.x0, self.y0, self.z0 = xyz0

    def geodetic2global(self):
        """
        Convert the antenna positions from the geodetic (WGS84) frame to the global (ITRF/ECEF) frame

        Returns
        ----
        An array of the antennas XYZ positions and an array of the array center in XYZ
        """

        longitude = self.geodtic_antennas[:, 0]
        latitude = self.geodtic_antennas[:, 1]
        altitude = self.geodtic_antennas[:, 2]

        ref_longitude = self.lon0
        ref_latitude = self.lat0
        ref_altitude = self.alt0

        nnp = constants.earth_emaj / np.sqrt(1 - constants.esq * np.sin(latitude) ** 2)
        nn0 = constants.earth_emaj / np.sqrt(1 - constants.esq * np.sin(ref_latitude**2))

        # calculating the global coordinates of the antennas.
        x = (nnp + altitude) * np.cos(latitude) * np.cos(longitude)
        y = (nnp + altitude) * np.cos(latitude) * np.sin(longitude)
        z = ((1 - constants.esq) * nnp + altitude) * np.sin(latitude)

        # calculating the global coordinates of the array center.
        x0 = (nn0 + ref_altitude) * np.cos(ref_latitude) * np.cos(ref_longitude)
        y0 = (nn0 + ref_altitude) * np.cos(ref_latitude) * np.sin(ref_longitude)
        z0 = ((1 - constants.esq) * nn0 + ref_altitude) * np.sin(ref_latitude)

        xyz = np.column_stack((x, y, z))
        xyz0 = np.array([x0, y0, z0])

        return xyz, xyz0

    def global2geodetic(self, X, Y, Z):
        """
        Convert the antenna positions from the global (ITRF/ECEF) frame to the geodetic (WGS84) frame.

        Parameters
        ----
        (X,Y,Z): float
                : global coordinates of the antennas.

        Returns
        ----
        (phi,lam,h): float
                : geodetic coordinates of the antennas.
        """

        f = 1 / 298.257223563
        b = constants.earth_emaj * (1 - f)
        ep2 = (constants.earth_emaj**2 - b**2) / b**2

        p = (X**2 + Y**2) ** 0.5
        theta = np.arctan2(Z * constants.earth_emaj, p * b)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        phi = np.arctan2(Z + ep2 * b * sin_theta**3, p - constants.esq * constants.earth_emaj * cos_theta**3)
        lam = np.arctan2(Y, X)
        v = constants.earth_emaj / (1 - constants.esq * np.sin(phi) ** 2) ** 0.5
        h = p / np.cos(phi) - v

        geodetic = np.array((lam, phi, h))

        return geodetic

    def uvgen(
        self, pointing_direction, dtime, ntimes, start_freq, dfreq, nchan, start_time=None, start_ha=None
    ) -> ObjDict:
        """
        Generate uvw coordimates

        Parameters
        ---

        pointing_direction: List[str]
                    : pointing direction.
        dtime: int
                    : integration time.
        ntimes: int
                    : number of times the sky should be snapped.
        start_freq: Union[str,float]
                    : starting freq of the observation.
        dfreq: Union[str,float]
                    : frequency interval.
        nchan: int
                    : number of channels.

        start_time: Union[str, List[str]]
                    : start time of the observation in UTC (YYYY-MM-DDTHH:MM:SS). Default is the current machine time.
        start_ha: float
                    : local hour angle of the first integration centre, in hours

        Returns
        ---
        An array of the uvw time dependent positions, the time array and the frequency array
        """
        dm = measures()

        latitude = self.lat0

        if len(pointing_direction) == 3:
            ra_dec = dm.direction(rf=pointing_direction[0], v0=pointing_direction[1], v1=pointing_direction[2])
        else:
            ra_dec = dm.direction(rf="J2000", v0=pointing_direction[0], v1=pointing_direction[1])
        ra0 = ra_dec["m0"]["value"]
        dec0 = ra_dec["m1"]["value"]

        start_day = Time(start_time, format="isot", scale="utc") if start_time else Time.now()
        if start_ha is not None:
            if not np.isfinite(start_ha):
                raise ValueError("start_ha must be a finite local hour angle in hours")
            # Choose the occurrence nearest the supplied UTC epoch (or now).
            # Iterate in sidereal time so the geometry and MS timestamps agree.
            target = start_ha * u.hourangle
            for _ in range(3):
                local_ha = start_day.sidereal_time("mean", longitude=self.lon0 * u.rad) - ra0 * u.rad
                error = (local_ha - target).wrap_at(180 * u.deg)
                start_day -= error.to_value(u.hourangle) / 1.00273790935 * u.hour

        sample_times = start_day + np.arange(ntimes) * dtime * u.s
        time_entries = sample_times.to_value("mjd") * 86400
        # Absolute ITRF baselines need Greenwich HA; elevations need local HA.
        h0 = sample_times.sidereal_time("mean", longitude=0 * u.deg).to_value("rad") - ra0
        source_elevations = self.get_source_elevation(latitude, dec0, h0 + self.lon0)

        # Transformation matrix
        dec = np.ones(ntimes) * dec0
        transform_matrix = np.array(
            [
                [np.sin(h0), np.cos(h0), np.zeros(ntimes)],
                [-np.sin(dec) * np.cos(h0), np.sin(dec) * np.sin(h0), np.cos(dec)],
                [-np.cos(dec) * np.cos(h0), np.cos(dec) * np.sin(h0), np.sin(dec)],
            ]
        )

        antenna1_list = []
        antenna2_list = []

        # Compute baselines and track antenna coordinates (vectorized)
        i_idx, j_idx = np.triu_indices(self.nant, k=1)
        bl_array = self.antlocations[j_idx] - self.antlocations[i_idx]
        antenna1_list = i_idx.tolist()
        antenna2_list = j_idx.tolist()

        u_coord = (
            np.outer(transform_matrix[0, 0], bl_array[:, 0])
            + np.outer(transform_matrix[0, 1], bl_array[:, 1])
            + np.outer(transform_matrix[0, 2], bl_array[:, 2])
        )
        v_coord = (
            np.outer(transform_matrix[1, 0], bl_array[:, 0])
            + np.outer(transform_matrix[1, 1], bl_array[:, 1])
            + np.outer(transform_matrix[1, 2], bl_array[:, 2])
        )
        w_coord = (
            np.outer(transform_matrix[2, 0], bl_array[:, 0])
            + np.outer(transform_matrix[2, 1], bl_array[:, 1])
            + np.outer(transform_matrix[2, 2], bl_array[:, 2])
        )

        u_coord, v_coord, w_coord = [x.flatten() for x in (u_coord, v_coord, w_coord)]
        uvw = np.column_stack((u_coord, v_coord, w_coord))

        time_table = np.repeat(time_entries[:, np.newaxis], len(antenna1_list), axis=1).flatten()

        start_freq = dm.frequency(v0=start_freq)["m0"]["value"]
        dfreq = dm.frequency(v0=dfreq)["m0"]["value"]
        end_freq = start_freq + dfreq * (nchan - 1)

        freqs = np.linspace(start_freq, end_freq, nchan)

        uvcoverage = ObjDict(
            {
                "antenna1": antenna1_list,
                "antenna2": antenna2_list,
                "uvw": uvw,
                "freqs": freqs,
                "times": time_table,
                # The ntimes distinct timestamps, before the per-baseline repeat above.
                # Subtables indexed by (antenna, time) -- POINTING -- need these, not the
                # main-table row times.
                "time_entries": time_entries,
                "source_elevations": source_elevations,
                "ra0": ra0,
                "dec0": dec0,
            }
        )

        return uvcoverage

    def get_source_elevation(self, latitude: float, declination: float, hour_angles: list[float]) -> list[float]:
        """
        Track the source during the observation and get its elevation.

        Parameters
        ----
        latitude: float
                : latitude of the observer in radians.
        declination: float
                : declination of the source in radians.
        hour_angles: array[float]
                : hour angles of the source in radians.

        Returns
        ----
        source_elevations: array[float]
                : the elevations of the source in degrees.
        """

        sin_elevation = np.sin(declination) * np.sin(latitude) + np.cos(declination) * np.cos(latitude) * np.cos(
            hour_angles
        )

        elevation = np.degrees(np.arcsin(sin_elevation))

        return elevation


def calculate_array_centre(ant_locations: list[list[float]]) -> list[float]:
    """
    Calculate the geometric centre of an array given the antenna locations.

    Parameters
    ----------
    ant_locations : List[List[float]]
        A list of antenna locations, where each location is a list of [X, Y, Z] coordinates.

    Returns
    -------
    List[float]
        The [X, Y, Z] coordinates of the geometric centre of the array.
    """
    # convert to NumPy array and compute mean along axis 0 (across antennas)
    ant_array = np.array(ant_locations)
    centre = np.mean(ant_array, axis=0)

    return centre.tolist()


def write_centre_to_array_config(array_config_path: str, centre: list[float]) -> None:
    """
    Write the calculated centre to the array configuration YAML file.

    Parameters
    ----------
    array_config_path : File
        Path to the array configuration YAML file.
    centre : List[float]
        The [X, Y, Z] coordinates of the geometric centre to write.
    """
    with open(array_config_path) as f:
        array_info = yaml.safe_load(f)

    array_info["centre"] = centre

    with open(array_config_path, "w") as f:
        yaml.dump(array_info, f)
