"""
Import model
"""

from gc import collect
from importlib import import_module
from logging import getLogger
from os import path

from pandas import DataFrame, to_numeric

import pastas as ps

logger = getLogger(__name__)


def load(fname, **kwargs):
    """
    Method to load a Pastas Model from file.

    Parameters
    ----------
    fname: str
        string with the name of the file to be imported including the file
        extension.
    kwargs:
        extension specific keyword arguments

    Returns
    -------
    ml: pastas.model.Model
        Pastas Model instance.

    Examples
    --------
    >>> import pastas as ps
    >>> ml = ps.io.load("model.pas")

    """
    if not path.exists(fname):
        msg = "File not found: {}".format(fname)
        logger.error(msg)

    # Dynamic import of the export module
    ext = path.splitext(fname)[1]
    load_mod = import_module("pastas.io" + ext)

    # Get dicts for all data sources
    data = load_mod.load(fname, **kwargs)

    # Determine whether it is a Pastas Project or a Pastas Model
    if "models" in data.keys():
        msg = "Deprecation Warning: the possibility to load a Pastas project" \
              "with this method is deprecated. Please use ps.io.load_project."
        logger.error(msg)
        raise DeprecationWarning(msg)

    ml = _load_model(data)

    logger.info("Pastas Model from file {} successfully loaded. This file "
                "was created with Pastas {}. Your current version of Pastas "
                "is: {}".format(fname, data["file_info"]["pastas_version"],
                                ps.__version__))
    return ml


def load_project(fname, **kwargs):
    """
    Method to load a Pastas project.

    Parameters
    ----------
    fname: str
        string with the name of the file to be imported including the file
        extension.
    kwargs:
        extension specific keyword arguments.

    Returns
    -------
    mls: pastas.project.Project
        Pastas Project class object

    Examples
    --------
    >>> import pastas as ps
    >>> mls = ps.io.load_project("project.pas")

    Warnings
    --------
    All classes and methods dealing with Pastas projects will be moved to a
    separate Python package in the near future (mid-2020).

    """
    # Dynamic import of the export module
    ext = path.splitext(fname)[1]
    load_mod = import_module("pastas.io" + ext)

    # Get dicts for all data sources
    data = load_mod.load(fname, **kwargs)

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
            ml = _load_model(ml)
            mls.models[ml_name] = ml
        except:
            try:
                mls.del_model(ml_name)
            except:
                pass
            print("model", ml_name, "could not be added")

    logger.info("Pastas project from file {} successfully loaded. This file "
                "was created with Pastas {}. Your current version of Pastas "
                "is: {}".format(fname, data["file_info"]["pastas_version"],
                                ps.__version__))

    return mls


def _load_model(data):
    """Internal method to create a model from a dictionary."""
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
        if "recharge" in ts.keys():
            ts["recharge"] = getattr(ps.recharge, ts["recharge"])()
        if "stress" in ts.keys():
            for i, stress in enumerate(ts["stress"]):
                ts["stress"][i] = ps.TimeSeries(**stress)
        if "prec" in ts.keys():
            ts["prec"] = ps.TimeSeries(**ts["prec"])
        if "evap" in ts.keys():
            ts["evap"] = ps.TimeSeries(**ts["evap"])
        if "temp" in ts.keys() and ts["temp"] is not None:
            ts["temp"] = ps.TimeSeries(**ts["temp"])
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

    # Add fit object to the model
    if "fit" in data.keys():
        fit = getattr(ps.solver, data["fit"]["name"])
        data["fit"].pop("name")
        ml.fit = fit(ml=ml, **data["fit"])

    # Add parameters, use update to maintain correct order
    ml.parameters = ml.get_init_parameters(noise=ml.settings["noise"])
    ml.parameters.update(data["parameters"])
    ml.parameters = ml.parameters.apply(to_numeric, errors="ignore")

    # When initial values changed
    for param, value in ml.parameters.loc[:, "initial"].iteritems():
        ml.set_parameter(name=param, initial=value)

    collect()

    return ml


def dump(fname, data, **kwargs):
    """
    Method to save a pastas-model to a file.

    Parameters
    ----------
    fname: str
        string with the name of the file, including a supported
        file-extension. Currently supported extension are: .pas.
    data: dict
        dictionary with the information to store.
    kwargs:
        extension specific keyword arguments can be provided using kwargs.

    Returns
    -------
    message:
        Message if the file-saving was successful.

    Notes
    -----
    The specific dump-module is automatically chosen based on the provided
    file extension.

    """
    ext = path.splitext(fname)[1]
    dump_mod = import_module("pastas.io" + ext)
    return dump_mod.dump(fname, data, **kwargs)
