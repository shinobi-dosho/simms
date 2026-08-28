import os.path
from dataclasses import dataclass
from typing import Any

import astropy.units as aunits
import numpy as np
from astropy.coordinates import (
    Angle,
    Latitude,
    Longitude,
    SpectralCoord,
)

from simms import SCHEMADIR
from simms.exceptions import (
    ASCIISkymodelError,
    ASCIISourceError,
    SkymodelSchemaError,
)
from simms.skymodel.source_factory import (
    StokesData,
    continuum_source,
    contspec,
    exoplanet_transient_logistic,
    exoplanet_transient_source,
    gauss_1d,
    gaussian_source,
    line_source,
    point_source,
    polarised_source,
)
from simms.utilities import ObjDict, load_yaml, quantity_to_value

DEFAULT_SOURCE_SCHEMA = os.path.join(SCHEMADIR, "source_schema.yaml")

# A catalogue is one table, so a file holding more than one source type has to carry
# every type's columns; the rows that are not of a given type mark theirs as unset with
# one of these tokens (or an empty field, for a delimited file). An unset column behaves
# exactly as if it were absent from the header for that row -- in particular it does not
# make the source claim a source type whose values are all None.
NULL_TOKENS = frozenset({"", "null", "none", "nan"})


def is_unset(value: str | float | int | None) -> bool:
    """True if an ASCII field marks the parameter as not set for this source."""
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in NULL_TOKENS


PTYPE_MAPPER = {
    # parameter type: (converter, default_units, null_value)
    "longitude": (Longitude, "rad", None),
    "latitude": (Latitude, "rad", None),
    "angle": (Angle, "rad", 0),
    "frequency": (SpectralCoord, "Hz", None),
    "number": (aunits.Quantity, None, None),
    "flux": (aunits.Quantity, "Jy", 0),
    "time": (aunits.Quantity, "s", None),
    "string": (str, None, ""),
}


@dataclass
class SkymodelParameter:
    info: str | None = None
    units: str | None = None
    alias: str | None = None
    ptype: str | None = "number"
    frame: str | None = None
    required: bool | None = False
    join: list[str] | None = None
    value: Any = None

    def set_value(self, value: str | float | int):
        """
        Set and convert the parameter value to the appropriate type and units.

        Parameters
        ----------
        value : str or float or int
            Value to set. Can be a plain number or a string with units.

        Returns
        -------
        None
            The converted value is stored in ``self.value``.

        Raises
        ------
        KeyError
            If the parameter type is unknown.
        ValueError
            If the value cannot be parsed or converted to the target units.
        """
        ptype_coord, target_units, null_value = PTYPE_MAPPER[self.ptype]
        self.value = quantity_to_value(ptype_coord, value, self.units, target_units=target_units, null_value=null_value)


@dataclass
class ASCIISourceSchema:
    info: str
    parameters: dict[str, SkymodelParameter]

    def __post_init__(self):
        # SkymodelParameter is a dataclass whose every field has a default, so constructing
        # it from a partial schema entry fills the rest in -- which is all the OmegaConf
        # structured-merge this replaced was doing, minus a deep copy of the whole config.
        # An unknown key now raises TypeError here rather than being merged in silently.
        # These are shared scratch converters: set_source_param copies the converted value
        # onto the source.
        self.parameters = ObjDict({key: SkymodelParameter(**val) for key, val in self.parameters.items()})


@dataclass
class ASCIISource:
    schema: ASCIISourceSchema

    def __post_init__(self):
        self.parameters = self.schema.parameters
        self.existing_fields = []
        self.raw_values = {}
        self.is_finalised = False

    def set_source_param(self, field: str, value: str | float | int):
        """
        Set a parameter value for the source.

        Parameters
        ----------
        field : str
            Name of the source parameter (e.g., 'ra', 'dec', 'flux', 'name').
        value : str or float or int
            Value to assign. Converted to the appropriate type and units.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If ``field`` is not a valid parameter in the schema.
        ValueError
            If the value cannot be converted to the required type/units.

        Notes
        -----
        This method updates the instance in place. A value that :func:`is_unset`
        recognises (an empty field, or ``null``/``none``/``nan``) leaves the
        parameter unset, as if the column were absent from the header.
        """
        if is_unset(value):
            # Leave the parameter off the source entirely: registering it would put the
            # field in existing_fields, and a null-valued line_* or transient_* column
            # would then type the source as a line/transient source whose parameters are
            # all None. See NULL_TOKENS.
            return
        param = getattr(self.parameters, field)
        param.set_value(value)
        setattr(self, field, param.value)
        # Joined fields (e.g. dec from dec_d/dec_m/dec_s) are re-parsed from the
        # original strings in finalise(), because the sign is lost once "-00" is
        # converted to 0.0.
        self.raw_values[field] = value
        self.existing_fields.append(field)

    def alias_to_field_mapper(self):
        mapper = {}
        for key, val in self.schema.parameters.items():
            mapper[getattr(val, "alias", None) or key] = key
        return mapper

    def field_to_alias_mapper(self):
        mapper = {}
        for key, val in self.schema.parameters.items():
            mapper[key] = getattr(val, "alias", None) or key
        return mapper

    def finalise(self):
        """
        Validate set fields and derive source type flags.

        Notes
        -----
        - Ensures the source satisfies the minimal point-source requirements.
        - Sets boolean flags: ``is_point``, ``is_polarised``, ``is_line``,
          ``is_continuum``, ``is_transient``, and ``is_exoplanet_transient``.
        - Computes and sets fields that are defined as a join of other fields.

        Raises
        ------
        ASCIISourceError
            If the source does not satisfy minimal point-source requirements.
        """
        fields = self.existing_fields
        # If the source is not a valid point source, it's not a valid source
        point_source.is_valid(fields, raise_exception=True)

        self.is_point = not gaussian_source.is_valid(fields)

        self.is_polarised = polarised_source.is_valid(fields)

        # none_or_all: a source with some but not all line fields is an error,
        # not silently a continuum source
        self.is_line = line_source.is_valid(fields, none_or_all=True)
        if self.is_line:
            # gauss_1d divides by the width, so width <= 0 gives NaNs, and
            # 1 + redshift <= 0 gives a zero/negative observed line centre
            width = self.value_or_default("line_width")
            if width <= 0:
                raise ASCIISourceError(f"Spectral line source line_width must be positive, got {width} Hz")
            redshift = getattr(self, "line_redshift", 0.0)
            if redshift <= -1:
                raise ASCIISourceError(f"Spectral line source line_redshift must be > -1, got {redshift}")

        self.is_continuum = continuum_source.is_valid(fields)

        self.is_transient = self.is_exoplanet_transient = exoplanet_transient_source.is_valid(fields, none_or_all=True)

        for field, param in self.parameters.items():
            if getattr(param, "join", []) and field not in fields:
                join_us = param.join
                if not any(key in fields for key in join_us):
                    continue
                # Join the components as one sexagesimal string so the sign of the
                # leading component applies to the minutes/seconds too: summing the
                # already-converted values gives -65 + 45' = -64.25 deg instead of
                # -65.75 deg, and a "-00" degree field loses its sign entirely.
                sexagesimal = ":".join(str(self.raw_values.get(key, 0)) for key in join_us)
                # The leading component's units set the sexagesimal scale
                # (hourangle for h:m:s of RA, deg for d:m:s of declination).
                scale_units = getattr(self.parameters, join_us[0]).units or param.units
                ptype_coord, target_units, _ = PTYPE_MAPPER[param.ptype]
                try:
                    joined = ptype_coord(sexagesimal, unit=scale_units).to(target_units).value
                except ValueError as exc:
                    raise ASCIISourceError(
                        f"Cannot parse field '{field}' from its components {join_us} (joined as '{sexagesimal}'): {exc}"
                    ) from exc
                setattr(self, field, joined)

    def value_or_default(self, field: str) -> int | float | str:
        """
        Return the value of a field or its default.

        Parameters
        ----------
        field : str
            Source field name (e.g., 'ra', 'dec', 'stokes_i').

        Returns
        -------
        int or float or str
            The value set from the ASCII file, or the schema default if unset.
        """
        val = getattr(self, field, None)
        if val is None:
            param = getattr(self.parameters, field)
            param.set_value(None)
            val = param.value
        return val

    def get_brightness_matrix(
        self,
        chan_freqs: np.ndarray,
        ncorr: int,
        unique_times: np.ndarray = None,
        time_index_mapper: np.ndarray = None,
        full_stokes: bool = True,
        linear_basis: bool = True,
    ) -> np.ndarray:
        """
        Generate the brightness matrix across frequency (and time if transient).

        Parameters
        ----------
        chan_freqs : numpy.ndarray
            1D array of channel frequencies in Hz.
        ncorr : int
            Number of correlations (e.g., 4 for full Stokes).
        unique_times : numpy.ndarray, optional
            1D array of unique time samples (seconds) for transient sources.
        time_index_mapper : numpy.ndarray, optional
            Mapping indices from unique time samples to the output time axis.
        full_stokes : bool, default True
            If True, compute all Stokes parameters; otherwise Stokes I only.
        linear_basis : bool, default True
            If True, use linear polarization basis (XX, XY, YX, YY). If False, use circular basis (RR, RL, LR, LL).

        Returns
        -------
        numpy.ndarray
            Brightness matrix with shape ``(ncorr, ntime, nfreq)`` for transient
            sources or ``(ncorr, nfreq)`` for non-transient sources.

        Raises
        ------
        ValueError
            If required spectral or transient fields are missing or invalid.

        Notes
        -----
        - Spectrum is modeled as a Gaussian line or a polynomial continuum,
          depending on which fields are present.
        - For transient sources, a logistic lightcurve is applied. The lightcurve
          only scales the spectrum, so it is applied as a separable factor here.
        """
        self.stokes = StokesData([getattr(self, f"stokes_{x}", 0) for x in "iquv"], linear_basis=linear_basis)
        if self.is_line:
            specfunc = gauss_1d
            # The line centre is the observed peak frequency when given;
            # otherwise it is derived from the rest frequency and redshift.
            x0 = getattr(self, "line_peak", None)
            if x0 is None:
                x0 = self.line_restfreq / (1.0 + getattr(self, "line_redshift", 0.0))
            kwargs = {
                "x0": x0,
                "width": self.value_or_default("line_width"),
            }
        elif self.is_continuum:
            specfunc = contspec
            kwargs = {
                "coeff": self.continuum_coefficients(),
                "nu_ref": self.value_or_default("cont_reffreq"),
            }

        self.stokes.set_spectrum(chan_freqs, specfunc, full_pol=full_stokes, **kwargs)
        bmatrix = self.stokes.get_brightness_matrix(ncorr)

        if self.is_transient and unique_times is not None:
            light_curve = self.get_lightcurve(unique_times)
            bmatrix = bmatrix[:, np.newaxis, :] * light_curve[np.newaxis, :, np.newaxis]
            if time_index_mapper is not None:
                bmatrix = bmatrix[:, time_index_mapper, ...]

        return bmatrix

    def continuum_coefficients(self) -> list[float]:
        """
        Continuum polynomial coefficients, highest unset orders dropped.

        Returns
        -------
        list of float
            ``[cont_coeff_1, cont_coeff_2, cont_coeff_3]`` truncated after the
            last coefficient the source actually sets. An empty list means the
            source has no continuum spectrum.
        """
        coeffs = [self.value_or_default(f"cont_coeff_{order}") for order in (1, 2, 3)]
        while coeffs and coeffs[-1] is None:
            coeffs.pop()
        return [0.0 if coeff is None else coeff for coeff in coeffs]

    def get_lightcurve(self, unique_times: np.ndarray) -> np.ndarray:
        """
        Evaluate the transient lightcurve on a time axis.

        Parameters
        ----------
        unique_times : numpy.ndarray
            Sorted unique time stamps spanning the whole observation. Transient
            parameters are defined relative to ``unique_times[0]``, so this must
            not be a per-block subset of the observation's times.

        Returns
        -------
        numpy.ndarray
            Fractional intensity of shape ``(unique_times.size,)``.
        """
        return exoplanet_transient_logistic(
            times=unique_times - unique_times[0],
            transient_start=self.value_or_default("transient_start"),
            transient_period=self.value_or_default("transient_period"),
            transient_ingress=self.value_or_default("transient_ingress"),
            transient_absorb=self.value_or_default("transient_absorb"),
        )


@dataclass
class ASCIISkymodel:
    """
    A sky model built from an ASCII column file.

    Parameters
    ----------
    skymodel_file : str
        Path to the sky model file.
    delimiter : str, optional
        Column delimiter used in the file. If ``None``, split on whitespace.
    source_schema_file : str, optional
        Path to the YAML source schema. Defaults to ``DEFAULT_SOURCE_SCHEMA``.
    sources : list of ASCIISource, optional
        Populated after parsing.

    Raises
    ------
    ASCIISkymodelError
        If the file cannot be read or validated.
    """

    skymodel_file: str
    delimiter: str = None
    source_schema_file: str = None
    sources: list[ASCIISource] = None

    def __post_init__(self):
        self.source_schema_file = self.source_schema_file or DEFAULT_SOURCE_SCHEMA
        schema = load_yaml(self.source_schema_file)
        self.schema = ASCIISourceSchema(**schema)
        sources = []

        # make a dummy source for some book keeping
        dummy_source = ASCIISource(self.schema)
        alias_to_field = dummy_source.alias_to_field_mapper()

        with open(self.skymodel_file) as stdr:
            line = stdr.readline().strip()

            if not line.startswith("#format:"):
                raise ASCIISkymodelError("ASCII sky model needs to have a header starting with the string #format:")

            header = line.strip().replace("#format:", "").strip().split(self.delimiter)

            unknown = [key for key in header if key not in alias_to_field]
            if unknown:
                raise ASCIISkymodelError(
                    f"ASCII sky model header has unknown fields {unknown}."
                    f" Known fields/aliases are: {sorted(alias_to_field.keys())}"
                )

            try:
                point_source.is_valid(fields=[alias_to_field[key] for key in header], raise_exception=True)
            except ASCIISourceError as exc:
                raise ASCIISkymodelError(f"ASCII Sky model file header is missig required fields: {exc}") from exc

            for counter, line in enumerate(stdr.readlines()):
                # skip lines that are commented
                if line.startswith("#"):
                    continue
                # skip empty lines before splitting: "".split(",") is [""], not [],
                # so a blank line in a delimited file would fail the column count
                if not line.strip():
                    continue
                rowdata = line.strip().split(self.delimiter)
                if len(rowdata) != len(header):
                    # plus 2 because header line is not counted (+1), and python starts counting at 0 (+1)
                    raise ASCIISkymodelError(
                        f"The number of columns in row {counter + 2} rows does not match the number of"
                        " the number of columns in the header"
                    )

                source = ASCIISource(self.schema)
                try:
                    for param, value in zip(header, rowdata, strict=True):
                        source.set_source_param(alias_to_field[param], value)
                    # source has been fully set, finalise and add it to rest of the sources
                    source.finalise()
                except (ASCIISourceError, SkymodelSchemaError, ValueError, aunits.UnitsError) as exc:
                    # Point at the offending row: the parse and validation errors name a
                    # field but not which source it came from, which is useless in a
                    # catalogue of thousands. Re-raise the same type so callers that
                    # discriminate on ASCIISourceError still see it.
                    raise type(exc)(f"{self.skymodel_file}, line {counter + 2}: {exc}") from exc
                # Record the source's line index in the file (header is line 0, so the first
                # data line -- counter 0 -- is line 1). Lets consumers map a parsed source back
                # to its original text line without re-deriving the comment/blank-line skips.
                source.lineno = counter + 1
                sources.append(source)

        self.sources = sources

    def _has_source_type(self, source_type: str):
        has_it = False
        for source in self.sources:
            if getattr(source, f"is_{source_type}", False):
                has_it = True
                break
        return has_it

    @property
    def has_transient(self):
        return self._has_source_type("transient")
