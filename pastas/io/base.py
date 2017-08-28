"""
Import model
"""

import importlib
import os

import pastas as ps


def load(fname, **kwargs):
    """Method to load models from file supported by the pastas library.

    Parameters
    ----------
    fname: str
        string with the name of the file to be imported including the file
        extension.
    kwargs: extension specific

    Returns
    -------
    message: str
        Returns a message if model import was successful.

    """
    # Dynamic import of the export module
    ext = os.path.splitext(fname)[1]
    load_mod = importlib.import_module("pastas.io" + ext)

    # Get dicts for all data sources
    data = load_mod.load(fname, **kwargs)

    # Create model
    oseries = data["oseries"]["series"]

    if "constant" in data.keys():
        constant = data["constant"]
    else:
        constant = False

    if "settings" in data.keys():
        settings = data["settings"]
    else:
        settings = dict()

    if "metadata" in data.keys():
        metadata = data["metadata"]
    else:
        metadata = dict(name="Model")  # Make sure there is a name

    if "name" in data.keys():
        name = data["name"]
    else:
        name = metadata["name"]

    ml = ps.Model(oseries, name=name, constant=constant, metadata=metadata,
                  settings=settings)

    # Add noisemodel if present
    if "noisemodel" in data.keys():
        n = getattr(ps.tseries, data["noisemodel"]["type"])()
        ml.add_noisemodel(n)

    # Add tseriesdict
    for name, ts in data["tseriesdict"].items():
        ts = getattr(ps.tseries, ts["type"])
        # ts = ts(name=name)
        # ml.add_tseries(ts)

    # Add parameters
    ml.parameters = data["parameters"]

    return ml


def dump(fname, data, **kwargs):
    """Method to save a pastas-model to a file. The specific dump-module is
    automatically chosen based on the provided file extension.

    Parameters
    ----------
    fname: str
        string with the name of the file, including a supported
        file-extension. Currently supported extension are: .pas.
    data: dict
        dictionary with the information to store.
    kwargs: extension specific keyword arguments can be provided using kwargs.

    Returns
    -------
    message:
        Message if the file-saving was successful.

    """
    ext = os.path.splitext(fname)[1]
    dump_mod = importlib.import_module("pastas.io" + ext)
    return dump_mod.dump(fname, data, **kwargs)
