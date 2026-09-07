import logging
import os
from importlib import metadata

from astropy.time import Time
from astropy.utils import iers

__version__ = metadata.version(__package__)

# Allow IERS to proceed with slightly stale data when offline,
# rather than crashing. Coordinates may be less accurate.
try:
    iers_table = iers.IERS_Auto.open()
    if Time.now().mjd - iers_table.meta["predictive_mjd"] > iers.conf.auto_max_age:
        logging.getLogger(__name__).info(
            "IERS data is stale and cannot be refreshed (offline?). "
            "Proceeding with available data; coordinates may be slightly inaccurate."
        )
        iers.conf.auto_max_age = None
except Exception:
    iers.conf.auto_max_age = None

PCKGDIR = os.path.dirname(__file__)
SCHEMADIR = os.path.join(PCKGDIR, "schemas")


class BinClass:
    def __init__(self):
        self.skysim = "skysim"
        self.telsim = "telsim"
        self.primary_beam = "primary-beam"


BIN = BinClass()
