"""This file contains the import method for pas-files.

Import a .pas file (basically a json format)

R.A. Collenteur - August 2017

"""

import json
from collections import OrderedDict

import pandas as pd
import pastas as ps

def load(fname):
    data = json.load(open(fname), object_hook=pastas_hook)
    return data


def pastas_hook(obj):
    for key, value in obj.items():
        if key in ["tmin", "tmax", "date_modified", "date_created"]:
            obj[key] = pd.Timestamp(value)
        elif key == "stress":
            try:
                obj[key] = list()
                for ts in value:
                    obj[key].append(pd.read_json(ts, typ='series',
                                                 orient="split"))
            except:
                obj[key] = value
        elif key in ["series"]:
            try:
                obj[key] = pd.read_json(value, typ='series', orient="split")
            except:
                try:
                     obj[key] = ps.TimeSeries(**value)
                except:
                    obj[key] = value
        elif key in ["time_offset"]:
            obj[key] = pd.Timedelta(value)
        elif key in ["parameters"]:
            # Necessary to maintain order when using the JSON format!
            value = json.loads(value, object_pairs_hook=OrderedDict)
            obj[key] = pd.DataFrame(data=value, columns=value.keys()).T
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
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_json(date_format="iso", orient="split")
        elif isinstance(obj, pd.DataFrame):
            # Necessary to maintain order when using the JSON format!
            return obj.to_json(orient="index")
        elif isinstance(obj, pd.Timedelta):
            return obj.to_timedelta64().__str__()
        else:
            return super(PastasEncoder, self).default(obj)
