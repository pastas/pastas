"""
Import model
"""

import importlib
import os

import pandas as pd
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

    # Determine whether it is a Pastas Project or a Pastas Model
    if "models" in data.keys():
        ml = load_project(data)
        kind = "Project"
    else:
        ml = load_model(data)
        kind = "Model"

    print("Pastas %s from file %s succesfully loaded. The Pastas-version "
          "this file was created with was %s. Your current version of Pastas "
          "is: %s" % (kind, fname, data["file_info"]["pastas_version"],
                      ps.__version__))

    return ml


def load_project(data):
    """Method to load a Pastas project.

    Parameters
    ----------
    data: dict
        Dictionary containing all information to construct the project.

    Returns
    -------
    mls: Pastas.Project class
        Pastas Project class object

    """

    mls = ps.Project(name=data["name"])

    mls.metadata = data["metadata"]
    mls.file_info = data["file_info"]

    mls.stressmodel = pd.DataFrame(data["stressmodel"],
                               columns=data["stressmodel"].keys()).T

    mls.oseries = pd.DataFrame(data["oseries"],
                               columns=data["oseries"].keys()).T

    for name, ml in data["models"].items():
        ml["oseries"]["series"] = mls.oseries.loc[ml["oseries"]["series"],
                                                  "series"]
        if ml["stressmodels"]:
            for ts in ml["stressmodels"].values():
                for i, stressmodel in enumerate(ts["stress"]):
                    ts["stress"][i] = mls.stressmodel.loc[stressmodel, "series"]

        ml = load_model(ml)
        mls.models[name] = ml

    return mls


def load_model(data):
    # Create model
    oseries = ps.TimeSeries(**data["oseries"])

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
    if "file_info" in data.keys():
        ml.file_info.update(data["file_info"])

    # Add stressmodels
    for name, ts in data["stressmodels"].items():
        stressmodel = getattr(ps.stressmodel, ts["type"])
        ts.pop("type")
        ts["rfunc"] = getattr(ps.rfunc, ts["rfunc"])
        stressmodel = stressmodel(**ts)
        ml.add_stressmodel(stressmodel)

    # Add noisemodel if present
    if "noisemodel" in data.keys():
        n = getattr(ps.stressmodel, data["noisemodel"]["type"])()
        ml.add_noisemodel(n)

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
