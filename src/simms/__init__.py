import logging
import os
from importlib import metadata

from astropy.utils import iers

__version__ = metadata.version(__package__)

# Allow IERS to proceed with slightly stale data when offline,
# rather than crashing. Coordinates may be less accurate.
try:
    iers_table = iers.IERS_Auto.open()
    if iers_table.time_now.mjd - iers_table.meta["predictive_mjd"] > iers.conf.auto_max_age:
        logging.getLogger(__name__).info(
            "IERS data is stale and cannot be refreshed (offline?). "
            "Proceeding with available data; coordinates may be slightly inaccurate."
        )
        iers.conf.auto_max_age = None
except Exception:
    iers.conf.auto_max_age = None

PCKGDIR = __path__[0]
SCHEMADIR = os.path.join(PCKGDIR, "schemas")


def set_logger(name, level="INFO"):
    if isinstance(level, str):
        # An unrecognised name used to fall through to 10 (DEBUG), so a typo turned the
        # logging all the way *up*. INFO is the documented default and the safer landing.
        level = getattr(logging, level.upper(), logging.INFO)

        # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)-8s| %(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger


class BinClass:
    def __init__(self):
        self.skysim = "skysim"
        self.telsim = "telsim"
        self.primary_beam = "primary-beam"


BIN = BinClass()
