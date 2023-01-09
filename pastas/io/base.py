"""Import model."""

from importlib import import_module
from logging import getLogger
from os import path
from packaging import version
from numpy import log

import pastas as ps
from pandas import to_numeric

logger = getLogger(__name__)


def load(fname, **kwargs):
    """Method to load a Pastas Model from file.

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
        # Deal with old StressModel2 files for version 0.22.0. Remove in 0.23.0.
        if ts["stressmodel"] == "StressModel2":
            logger.warning("StressModel2 is removed since Pastas 0.22.0 and "
                           "is replaced by the RechargeModel using a Linear "
                           "recharge model. Make sure to save this file "
                           "again using Pastas version 0.22.0 as this file "
                           "cannot be loaded in newer Pastas versions. This "
                           "will automatically update your model to the newer "
                           "RechargeModel stress model.")
            ts["stressmodel"] = "RechargeModel"
            ts["recharge"] = "Linear"
            ts["prec"] = ts["stress"][0]
            ts["evap"] = ts["stress"][1]
            ts.pop("stress")
            ts.pop("up")

        # Deal with old parameter value b in HantushWellModel: b_new = np.log(b_old)
        if ((ts["stressmodel"] == "WellModel") and
            (version.parse(data["file_info"]["pastas_version"]) <
             version.parse("0.22.0"))):
            logger.warning("The value of parameter 'b' in HantushWellModel"
                           "was modified in 0.22.0: b_new = log(b_old). The value of "
                           "'b' is automatically updated on load.")
            wnam = ts["name"]
            for pcol in ["initial", "optimal", "pmin", "pmax"]:
                if wnam + "_b" in data["parameters"].index:
                    if data["parameters"].loc[wnam + "_b", pcol] > 0:
                        data["parameters"].loc[wnam + "_b", pcol] = \
                            log(data["parameters"].loc[wnam + "_b", pcol])

        stressmodel = getattr(ps.stressmodels, ts["stressmodel"])
        ts.pop("stressmodel")
        if "rfunc" in ts.keys():
            rfunc_kwargs = {}
            if "rfunc_kwargs" in ts:
                rfunc_kwargs = ts.pop("rfunc_kwargs")
            ts["rfunc"] = getattr(ps.rfunc, ts["rfunc"])(**rfunc_kwargs)
        if "recharge" in ts.keys():
            recharge_kwargs = {}
            if 'recharge_kwargs' in ts:
                recharge_kwargs = ts.pop("recharge_kwargs")
            ts["recharge"] = getattr(
                ps.recharge, ts["recharge"])(**recharge_kwargs)
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
    for param, value in ml.parameters.loc[:, "initial"].items():
        ml.set_parameter(name=param, initial=value)

    return ml


def dump(fname, data, **kwargs):
    """Method to save a pastas-model to a file.

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
