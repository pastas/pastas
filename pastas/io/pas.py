"""This file contains the import method for pas-files.

Import a .pas file (basically a json format)

R.A. Collenteur - August 2017

"""

import json
from collections import OrderedDict

from pandas import Series, Timedelta, DataFrame, read_json, Timestamp, \
    to_numeric, isna

from pastas import TimeSeries


def load(fname):
    data = json.load(open(fname), object_hook=pastas_hook)
    return data


def pastas_hook(obj):
    for key, value in obj.items():
        if key in ["tmin", "tmax", "date_modified", "date_created"]:
            val = Timestamp(value)
            if isna(val):
                val = None
            obj[key] = val
        elif key == "series":
            try:
                obj[key] = read_json(value, typ='series', orient="split")
            except:
                try:
                    obj[key] = TimeSeries(**value)
                except:
                    obj[key] = value
            if isinstance(obj[key], Series):
                obj[key].index = obj[key].index.tz_localize(None)
        elif key in ["time_offset", "warmup"]:
            if isinstance(value, int) or isinstance(value, float):
                obj[key] = Timedelta(value, 'd')
            else:
                obj[key] = Timedelta(value)
        elif key in ["parameters", "pcov"]:
            # Necessary to maintain order when using the JSON format!
            value = json.loads(value, object_pairs_hook=OrderedDict)
            param = DataFrame(data=value, columns=value.keys()).T
            obj[key] = param.apply(to_numeric, errors="ignore")
        else:
            try:
                obj[key] = json.loads(value, object_hook=pastas_hook)
            except:
                obj[key] = value
    return obj


def dump(fname, data, verbose=True):
    json.dump(data, open(fname, 'w'), indent=4, cls=PastasEncoder)
    if verbose:
        return print("%s file successfully exported" % fname)


class PastasEncoder(json.JSONEncoder):
    """Enhanced encoder to deal with the pandas formats used
    throughout Pastas.

    Notes
    -----
    Currently supported formats are: DataFrame, Series,
    Timedelta, TimeStamps.

    see: https://docs.python.org/3/library/json.html

    """

    def default(self, o):
        if isinstance(o, Timestamp):
            return o.isoformat()
        elif isinstance(o, Series):
            return o.to_json(date_format="iso", orient="split")
        elif isinstance(o, DataFrame):
            # Necessary to maintain order when using the JSON format!
            return o.to_json(orient="index")
        elif isinstance(o, Timedelta):
            return o.to_timedelta64().__str__()
        elif isna(o):
            return None
        else:
            return super(PastasEncoder, self).default(o)
