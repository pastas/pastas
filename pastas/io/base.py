"""
Import model
"""

from importlib import import_module
from os import path
import gc

from pandas import DataFrame, to_numeric

import pastas as ps


def load(fname, **kwargs):
    """Method to load models from file supported by the pastas library.

    Parameters
    ----------
    fname: str
        string with the name of the file to be imported including the file
        extension.
    kwargs: extension specific

    """
    if not path.exists(fname):
        raise (FileNotFoundError('File not found: {}'.format(fname)))

    # Dynamic import of the export module
    ext = path.splitext(fname)[1]
    load_mod = import_module("pastas.io" + ext)

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

    oseries = DataFrame(data["oseries"], columns=data["oseries"].keys()).T
    mls.oseries = mls.oseries.append(oseries, sort=False)

    stresses = DataFrame(data=data["stresses"],
                         columns=data["stresses"].keys()).T
    mls.stresses = mls.stresses.append(stresses, sort=False)

    for ml_name, ml in data["models"].items():
        name = str(ml["oseries"]["name"])
        ml_name = str(ml_name)
        ml["oseries"]["series"] = mls.oseries.loc[name, "series"]
        if ml["stressmodels"]:
            for ts in ml["stressmodels"].values():
                for stress in ts["stress"]:
                    if 'series' not in stress:
                        # look up the stress-series in mls.stresses
                        stress_name = stress["name"]
                        if stress_name not in mls.stresses.index:
                            raise (ValueError(
                                '{} not found in stresses'.format(
                                    stress_name)))
                        stress["series"] = mls.stresses.loc[
                            stress_name, "series"]
        try:
            ml = load_model(ml)
            mls.models[ml_name] = ml
        except:
            try:
                mls.del_model(ml_name)
            except:
                pass
            print("model", ml_name, "could not be added")
    return mls


def load_model(data):
    # Create model
    oseries = ps.TimeSeries(**data["oseries"])

    if "constant" in data.keys():
        constant = data["constant"]
    else:
        constant = False

    if "metadata" in data.keys():
        metadata = data["metadata"]
    else:
        metadata = None

    if "name" in data.keys():
        name = data["name"]
    else:
        name = None

    if "noisemodel" in data.keys():
        noise = True
    else:
        noise = False

    ml = ps.Model(oseries, constant=constant, noisemodel=noise, name=name,
                  metadata=metadata)

    if "settings" in data.keys():
        ml.settings.update(data["settings"])
    if "file_info" in data.keys():
        ml.file_info.update(data["file_info"])

    # Add stressmodels
    for name, ts in data["stressmodels"].items():
        stressmodel = getattr(ps.stressmodels, ts["stressmodel"])
        ts.pop("stressmodel")
        if "rfunc" in ts.keys():
            ts["rfunc"] = getattr(ps.rfunc, ts["rfunc"])
        if "stress" in ts.keys():
            for i, stress in enumerate(ts["stress"]):
                ts["stress"][i] = ps.TimeSeries(**stress)
        stressmodel = stressmodel(**ts)
        ml.add_stressmodel(stressmodel)

    # Add transform
    if "transform" in data.keys():
        transform = getattr(ps.transform, data["transform"]["transform"])
        data["transform"].pop("transform")
        transform = transform(**data["transform"])
        ml.add_transform(transform)

    # Add noisemodel if present
    if "noisemodel" in data.keys():
        n = getattr(ps.noisemodels, data["noisemodel"]["type"])()
        ml.add_noisemodel(n)

    # Add parameters, use update to maintain correct order
    ml.parameters = ml.get_init_parameters(noise=ml.settings["noise"])
    ml.parameters.update(data["parameters"])
    ml.parameters = ml.parameters.apply(to_numeric, errors="ignore")

    # When initial values changed
    for param, value in ml.parameters.loc[:, "initial"].iteritems():
        ml.set_initial(name=param, value=value)

    gc.collect()

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
    ext = path.splitext(fname)[1]
    dump_mod = import_module("pastas.io" + ext)
    return dump_mod.dump(fname, data, **kwargs)
