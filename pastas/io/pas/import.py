"""This file contains the import method for pas-files.

Import a .pas file (basically a json format)

R.A. Collenteur - August 2017

"""

import json

def get_data(fname):

    data = json.load(open(fname))

    return data