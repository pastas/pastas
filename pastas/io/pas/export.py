"""
Export model
"""

import json
import pandas as pd

def export(fname, data):

    json.dump(data, open(fname, 'w'), indent=4, cls=PastasEncoder)

    return print("%s file succesfully exported" % fname)


class PastasEncoder(json.JSONEncoder):
    """Enhanced encoder to deal with the pandas TimeStamp format used
    throughout PASTAS.

    """
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_json(date_format='iso')
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Timedelta):
            return obj.isoformat()
        else:
            return super(PastasEncoder, self).default(obj)



