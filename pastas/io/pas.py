"""This file contains the import method for pas-files.

Import a .pas file (basically a json format)

R.A. Collenteur - August 2017

"""

import json
from collections import OrderedDict

from pandas import NaT, Series, Timedelta, DataFrame, read_json, Timestamp, \
    to_numeric

from pastas import TimeSeries


def load(fname):
    data = json.load(open(fname), object_hook=pastas_hook)
    return data


def pastas_hook(obj):
    for key, value in obj.items():
        if key in ["tmin", "tmax", "date_modified", "date_created"]:
            val = Timestamp(value)
            if val is NaT:
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
        elif key == "time_offset":
            obj[key] = Timedelta(value)
        elif key == "parameters":
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


def dump(fname, data):
    json.dump(data, open(fname, 'w'), indent=4, cls=PastasEncoder)
    return print("%s file succesfully exported" % fname)


class PastasEncoder(json.JSONEncoder):
    """Enhanced encoder to deal with the pandas formats used
    throughout PASTAS.

    Notes
    -----
    Currently supported formats are: DataFrame, Series,
    Timedelta, TimeStamps.

    see: https://docs.python.org/3/library/json.html

    """

    def default(self, obj):

        if isinstance(obj, Timestamp):
            return obj.isoformat()
        elif isinstance(obj, Series):
            return obj.to_json(date_format="iso", orient="split")
        elif isinstance(obj, DataFrame):
            # Necessary to maintain order when using the JSON format!
            return obj.to_json(orient="index")
        elif isinstance(obj, Timedelta):
            return obj.to_timedelta64().__str__()
        elif isinstance(obj, NaT):
            return None
        else:
            return super(PastasEncoder, self).default(obj)
