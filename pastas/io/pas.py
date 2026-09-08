"""Functions to load and save Pastas models to and from JSON files.

This module provides the `load` and `dump` functions for reading and writing
Pastas models using JSON format. It also includes helper functions for
encoding and decoding Pandas-specific types.
"""

import datetime
import json
from collections import OrderedDict
from io import StringIO as stringIO
from logging import getLogger

from numpy import integer

try:
    from shapely.geometry.base import BaseGeometry

    SHAPELY = True
except ModuleNotFoundError:
    SHAPELY = False
    BaseGeometry = None

from pandas import (
    DataFrame,
    Series,
    Timedelta,
    Timestamp,
    isna,
    read_json,
    to_timedelta,
)

logger = getLogger(__name__)


def load(fname: str) -> dict:
    """Load a Pastas model from a JSON file.

    Parameters
    ----------
    fname : str
        Filename of the JSON file to load.

    Returns
    -------
    dict
        Dictionary containing the model data.
    """
    with open(fname, "r") as file:
        data = json.load(file, object_hook=pastas_hook)
    return data


def pastas_hook(obj: dict):
    """Decode Pastas-specific types from JSON.

    This function is used as the object_hook parameter in json.load() to convert
    JSON data back to Python/Pandas types used by Pastas.

    Parameters
    ----------
    obj : dict
        Dictionary from the JSON file.

    Returns
    -------
    dict
        Dictionary with Pastas types decoded.
    """
    for key, value in obj.items():
        if key in ["tmin", "tmax", "date_modified", "date_created"]:
            val = Timestamp(value)
            if isna(val):
                val = None
            obj[key] = val
        elif key == "series":
            try:
                obj[key] = read_json(
                    stringIO(value), typ="series", orient="split"
                ).astype(float)
            except Exception as e:  # noqa: BLE001
                logger.debug(e)
                obj[key] = value
            if isinstance(obj[key], Series):
                obj[key].index = obj[key].index.tz_localize(None)
        elif key in ["time_offset", "warmup"]:
            if isinstance(value, (int, float)):
                obj[key] = Timedelta(value, "D")
            else:
                obj[key] = Timedelta(value)
        elif key in ["parameters", "pcov"]:
            if value is not None:
                # Necessary to maintain order when using the JSON format!
                value = json.loads(value, object_pairs_hook=OrderedDict)
                param = DataFrame(data=value, columns=value.keys()).T
                obj[key] = param.infer_objects()
            else:
                obj[key] = value
        else:
            try:
                obj[key] = json.loads(value, object_hook=pastas_hook)
            except Exception as e:  # noqa: BLE001
                logger.debug(e)
                obj[key] = value
    return obj


def dump(fname: str, data: dict) -> None:
    """Save a Pastas model to a JSON file.

    Parameters
    ----------
    fname : str
        Filename to save the JSON data to.
    data : dict
        Dictionary containing the model data to save.
    """
    with open(fname, "w") as file:
        json.dump(data, file, indent=4, cls=PastasEncoder)
    logger.info("%s file successfully exported", fname)


class PastasEncoder(json.JSONEncoder):
    """Enhanced encoder to deal with the pandas formats used throughout Pastas.

    Notes
    -----
    Currently supported formats are: DataFrame, Series, Timedelta, Timestamps.

    see: https://docs.python.org/3/library/json.html
    """

    def default(self, o):
        """Encode special types to JSON.

        Parameters
        ----------
        o : object
            Object to encode.

        Returns
        -------
        object
            JSON-serializable representation of the object.
        """
        if isinstance(o, (Timestamp, datetime.datetime, datetime.date)):
            return o.isoformat()
        elif isinstance(o, Series):
            return o.to_json(date_format="iso", orient="split")
        elif isinstance(o, DataFrame):
            # Necessary to maintain order when using the JSON format!
            # Do not use o.to_json() because of float precision
            return json.dumps(o.to_dict(orient="index"), indent=0)
        elif isinstance(o, (Timedelta, datetime.timedelta)):
            if isinstance(o, datetime.timedelta):
                o = to_timedelta(o)
            return o.to_timedelta64().__str__()
        elif SHAPELY and isinstance(o, BaseGeometry):
            return o.wkt
        elif isna(o):
            return None
        elif isinstance(o, integer):
            return int(o)
        else:
            return super().default(o)
