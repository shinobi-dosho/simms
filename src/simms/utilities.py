from collections.abc import Callable
from itertools import combinations
from types import NoneType

import numpy as np
import yaml
from astropy import units
from dask import config as dask_config
from numba import njit

from simms.exceptions import InvalidInputError, SkymodelSchemaError


def set_dask_workers(nworkers):
    """Point dask's threaded scheduler at ``nworkers``, rejecting a non-positive count.

    dask turns ``num_workers=0`` into an opaque ``ValueError`` from deep inside the pool
    setup, and a negative count into a bare ``ValueError`` from ``ThreadPoolExecutor``;
    neither names the option the user actually typed.
    """
    if nworkers < 1:
        raise InvalidInputError(f"--nworkers must be at least 1, got {nworkers}.")
    dask_config.set(scheduler="threads", num_workers=nworkers)


def load_yaml(path) -> dict:
    """Load a YAML file as plain Python containers.

    All simms ever needed from ``omegaconf`` was this: read a YAML file. None of the shipped
    configs use interpolation, struct mode or merging, and the loaded mappings are only ever
    indexed, so a plain ``dict`` is the whole requirement.
    """
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


class ObjDict:
    def __init__(self, items):
        """
        Converts a dictionary into an object.

        """
        # First give this objects all the attributes of the input dicttionary
        for item in dir(dict):
            if not item.startswith("__"):
                setattr(self, item, getattr(items, item, None))
        # Now set the dictionary values as attributes
        self.__dict__.update(items)


def is_numeric(string):
    """
    Checks if a string can be converted to a float.
    """
    try:
        float(string)
        return True
    except (ValueError, TypeError):
        return False


@njit
def radec2lm(ra0: float, dec0: float, ra: float | np.ndarray, dec: float | np.ndarray):
    """
    Convert (RA, Dec) to direction cosine coordinates (l,m)

    Args:
        ra0 (float|np.ndarray): phase centre RA in radians.
        dec0 (float): phase centre Dec in radians.
        ra (float or np.ndarray): RA in radians.
        dec (float or np.ndarray): Dec in radians.
    Returns:
        a tuple of l and m
    """
    dra = ra - ra0
    l_coord = np.cos(dec) * np.sin(dra)
    m_coord = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)

    return l_coord, m_coord


def get_noise(sefds: list | float, dtime: int, dfreq: float):
    """
    This function computes the noise given an SEFD/s.
    """

    if isinstance(sefds, int | float):
        noise = sefds / np.sqrt(2 * dfreq * dtime)
        return noise

    sefd_pairs = list(combinations(sefds, 2))
    noises = []
    for sefd1, sefd2 in sefd_pairs:
        prod = sefd1 * sefd2
        den = 2 * dfreq * dtime
        noise = np.sqrt(prod / den)
        noises.append(noise)

    return noises


def is_range_in_range(inner_range, outer_range):
    """
    Checks if a given range (inner_range) is fully contained within
    another range (outer_range).

    Assumes ranges are inclusive and represented as (start, end) tuples.

    Args:
        inner_range (tuple): A tuple (start, end) representing the inner range.
        outer_range (tuple): A tuple (start, end) representing the outer range.

    Returns:
        bool: True if inner_range is fully within outer_range, False otherwise.
    """

    # Unpack the tuples
    inner_start, inner_end = inner_range
    outer_start, outer_end = outer_range

    # In case the ranges are given "backwards" (e.g., (10, 5)),
    # we normalize them to be (min, max).
    if inner_start > inner_end:
        inner_start, inner_end = inner_end, inner_start
    if outer_start > outer_end:
        outer_start, outer_end = outer_end, outer_start

    # The check:
    # 1. The inner range's start must be at or after the outer range's start.
    # 2. The inner range's end must be at or before the outer range's end.
    is_start_within = outer_start <= inner_start
    is_end_within = inner_end <= outer_end

    return is_start_within and is_end_within


def quantity_to_value(
    coord: Callable | NoneType,
    value: str | int | float,
    val_units: str | None = None,
    target_units: str | None = None,
    null_value=None,
) -> int | float:
    """
    Converts a value (string or numeric) with units to a float or int in the target units.

    Args:
        coord (Callable|NoneType): Function to convert the given
        value (str|numeric): The value to convert.
        val_units (str): The units of the value.
        target_units (str): The units to convert to.
    Raises:
        SkymodelSchemaError: If the units are unknown or invalid.

    Returns:
        float|int: The converted value in the target units.
    """
    if value in [None, "null"]:
        return null_value

    if isinstance(value, float | int):
        if val_units:
            try:
                quant_value = value * getattr(units, val_units)
            except AttributeError as exc:
                raise SkymodelSchemaError(f"Unknown parameter units '{val_units}'") from exc
        else:
            quant_value = units.Quantity(value)
    elif is_numeric(value):
        quant_value = coord(f"{value} {val_units}") if val_units else coord(value)
    else:
        quant_value = coord(value)

    if isinstance(quant_value, str):
        return quant_value

    if target_units:
        return quant_value.to(target_units).to_value()
    else:
        return quant_value.to_value()
