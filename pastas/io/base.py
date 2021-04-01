"""
Import model
"""

from importlib import import_module
from logging import getLogger
from os import path

from pandas import to_numeric

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
        logger.error("File not found: %s", fname)

    # Dynamic import of the export module
    load_mod = import_module(f"pastas.io{path.splitext(fname)[1]}")

    # Get dicts for all data sources
    data = load_mod.load(fname, **kwargs)

    ml = _load_model(data)

    logger.info("Pastas Model from file %s successfully loaded. This file "
                "was created with Pastas %s. Your current version of Pastas "
                "is: %s", fname, data["file_info"]["pastas_version"],
                ps.__version__)
    return ml


def load_project(fname, **kwargs):
    """
    Method to load a Pastas project. (Deprecated)
    """
    msg = "Deprecation Warning: the possibility to load a Pastas project" \
          " with this method is deprecated. Please use the Pastastore " \
          "(https://github.com/pastas/pastastore). "
    logger.error(msg)


def _load_model(data):
    """Internal method to create a model from a dictionary."""
    # Create model
    _remove_keyword(data["oseries"])
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
                _remove_keyword(stress)
                ts["stress"][i] = ps.TimeSeries(**stress)
        if "prec" in ts.keys():
            _remove_keyword(ts["prec"])
            ts["prec"] = ps.TimeSeries(**ts["prec"])
        if "evap" in ts.keys():
            _remove_keyword(ts["evap"])
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


def _remove_keyword(data):
    if "to_daily_unit" in data["settings"].keys():
        logger.warning("The key 'to_daily_unit' is removed. This "
                       "file will not work from Pastas 0.17.0. Make "
                       "sure to save your model again to a .pas-file.")
        data["settings"].pop("to_daily_unit")
