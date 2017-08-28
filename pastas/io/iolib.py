"""
Import model
"""

import json
import os
import importlib
import pandas as pd
import pastas as ps

def import_model(fname):

    # Dynamic import of the export module
    ext = os.path.splitext(fname)[1]
    ext = ext.replace('.', '')
    ext = '.io.' + ext + '.import'
    import_mod = importlib.import_module(ext, "pastas")

    # Get dicts for all data sources
    data = import_mod.import_data(fname)


    # Create model
    oseries = pd.read_json(data["oseries"]["stress"], typ='series')

    if "constant" in data.keys():
        constant = data["constant"]
    else:
        constant = False

    metadata = data["metadata"]

    ml = ps.Model(oseries, constant=constant, metadata=metadata)

    # Add noisemodel if present
    if "noisemodel" in data.keys():
        n = getattr(ps.tseries, data["noisemodel"]["type"])()
        ml.add_noisemodel(n)

    # Add tseriesdict
    for name, ts in data["tseriesdict"].items():
        ts = getattr(ps.tseries, ts["type"])
        #ts = ts(name=name)
        #ml.add_tseries(ts)

    # Add settings
    for setting, value in data["settings"].items():
        ml.__setattr__(setting, value)

    # Add parameters
    ml.parameters = pd.DataFrame(data=data["parameters"])

    return ml
