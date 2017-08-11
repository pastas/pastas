"""
Export model
"""

import json

def export(fname, data):

    json.dump(data, open(fname, 'w'), indent=4)

    return print("%s file succesfully exported" % fname)