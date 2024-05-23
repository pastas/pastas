"""Import model."""

from importlib import import_module
from logging import getLogger
from os import path

from packaging import version

import pastas as ps

# Type Hinting
from pastas.typing import Model

logger = getLogger(__name__)


def load(fname: str, **kwargs) -> Model:
    """Method to load a Pastas Model from file.

    Parameters
    ----------
    fname: str
        string with the name of the file to be imported including the file extension.
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
        msg = "File not found: %s"
        logger.error(msg, fname)
        raise FileNotFoundError(msg % fname)

    # Dynamic import of the export module
    load_mod = import_module(f"pastas.io{path.splitext(fname)[1]}")

    # Get dicts for all data sources
    data = load_mod.load(fname, **kwargs)

    file_version = data["file_info"]["pastas_version"]

    # A single catch for old pas-files, no longer supported
    if version.parse(file_version) < version.parse("0.23.0"):
        msg = (
            "This file was created with a Pastas version prior to 0.23 "
            "and cannot be loaded with Pastas >= 1.0. Please load and "
            "save the file with Pastas 0.23 first to update the file "
            "format."
        )
        logger.error(msg)
        raise ValueError(msg)

    ml = _load_model(data)

    logger.info(
        "Pastas Model from file %s successfully loaded. This file was created with "
        "Pastas %s. Your current version of Pastas is: %s",
        fname,
        file_version,
        ps.__version__,
    )
    return ml


def _load_model(data: dict) -> Model:
    """Internal method to create a model from a dictionary."""
    # Create model
    oseries = data["oseries"]["series"]
    metadata = data["oseries"]["metadata"]

    if "constant" in data.keys():
        constant = data["constant"]
    else:
        constant = False

    if "name" in data.keys():
        name = data["name"]
    else:
        name = None

    ml = ps.Model(
        oseries=oseries,
        constant=constant,
        name=name,
        metadata=metadata,
    )

    if "settings" in data.keys():
        if "noise" in data["settings"]:
            if not data["settings"]["noise"] and "noisemodel" in data:
                # file is saved before pastas 1.5, and solved with ml.solve(noise=False)
                # remove noisemodel from data
                data.pop("noisemodel")
        ml.settings.update(data["settings"])
    if "file_info" in data.keys():
        ml.file_info.update(data["file_info"])

    # Add stressmodels
    for name, smdata in data["stressmodels"].items():
        sm = _load_stressmodel(smdata, data)
        ml.add_stressmodel(sm)

    # Add transform
    if "transform" in data.keys():
        transform = getattr(ps.transform, data["transform"].pop("class"))
        transform = transform(**data["transform"])
        ml.add_transform(transform)

    # Add noisemodel if present
    if "noisemodel" in data.keys():
        # fixes to read pas-files from before pastas version 1.5
        # TODO: uncomment in pastas 2.0.0
        # if data["noisemodel"]["class"] == "NoiseModel":
        #     data["noisemodel"]["class"] = "ArNoiseModel"
        # if data["noisemodel"]["class"] == "ArmaModel":
        #     data["noisemodel"]["class"] = "ArmaNoiseModel"
        n = getattr(ps.noisemodels, data["noisemodel"].pop("class"))()
        ml.add_noisemodel(n)

    # Add solver object to the model from pas-files < 1.3.0  TODO Deprecate
    if "fit" in data.keys():
        logger.warning(
            "The solver object is stored in the model.solver attribute since Pastas "
            "1.3. Please update your pas-file to the new format by loading and saving "
            "the file with Pastas 1.3."
        )
        solver = getattr(ps.solver, data["fit"].pop("class"))
        ml.solver = solver(**data["fit"])
        ml.solver.set_model(ml)

    # Add solver object to the model
    if "solver" in data.keys():
        solver = getattr(ps.solver, data["solver"].pop("class"))
        ml.solver = solver(**data["solver"])
        ml.solver.set_model(ml)

    # Add parameters, use update to maintain correct order
    ml.parameters = ml.get_init_parameters(noise=ml.settings["noise"])
    ml.parameters.update(data["parameters"])

    # Convert parameters to numeric
    ml.parameters = ml.parameters.infer_objects()

    # When initial values changed
    for param, value in ml.parameters.loc[:, "initial"].items():
        ml.set_parameter(name=param, initial=value)

    return ml


def _load_stressmodel(ts, data):
    # Create and add stress model
    stressmodel = getattr(ps.stressmodels, ts.pop("class"))

    if "rfunc" in ts.keys():
        rfunc_class = ts["rfunc"].pop("class")  # Determine response class
        rfunc_up = ts["rfunc"].pop("up", None)  # get up value
        rfunc_gsf = ts["rfunc"].pop("gain_scale_factor", None)  # get gain_scale_factor
        rfunc = getattr(ps.rfunc, rfunc_class)(**ts["rfunc"])
        rfunc.update_rfunc_settings(up=rfunc_up, gain_scale_factor=rfunc_gsf)
        ts["rfunc"] = rfunc

    if "recharge" in ts.keys():
        recharge_class = ts["recharge"].pop("class")
        ts["recharge"] = getattr(ps.recharge, recharge_class)(**ts["recharge"])

    metadata = []
    settings = []

    # Unpack the stress time series
    if "stress" in ts.keys():
        # Only in the case of the wellmodel stresses are a list
        if isinstance(ts["stress"], list):
            for i, stress in enumerate(ts["stress"]):
                series, meta, setting = _unpack_series(stress)
                ts["stress"][i] = series
                metadata.append(meta)
                settings.append(setting)
        else:
            series, meta, setting = _unpack_series(ts["stress"])
            ts["stress"] = series
            metadata.append(meta)
            settings.append(setting)

    if "prec" in ts.keys():
        series, meta, setting = _unpack_series(ts["prec"])
        ts["prec"] = series
        metadata.append(meta)
        settings.append(setting)

    if "evap" in ts.keys():
        series, meta, setting = _unpack_series(ts["evap"])
        ts["evap"] = series
        metadata.append(meta)
        settings.append(setting)

    if "temp" in ts.keys() and ts["temp"] is not None:
        series, meta, setting = _unpack_series(ts["temp"])
        ts["temp"] = series
        metadata.append(meta)
        settings.append(setting)

    if metadata:
        ts["metadata"] = metadata if len(metadata) > 1 else metadata[0]
    if settings:
        ts["settings"] = settings if len(settings) > 1 else settings[0]

    sm = stressmodel(**ts)
    return sm


def _unpack_series(data: dict):
    """

    Parameters
    ----------
    data: dict
        Dictionary defining the TimeSeries

    Returns
    -------
    series, metadata, setings: dict

    """
    series = data["series"]
    metadata = data["metadata"]
    settings = data["settings"]

    return series, metadata, settings


def dump(fname: str, data: dict, **kwargs):
    """Method to save a pastas-model to a file.

    Parameters
    ----------
    fname: str
        string with the name of the file, including a supported file-extension.
        Currently supported extension are: .pas.
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
    The specific dump-module is automatically chosen based on the provided file
    extension.
    """
    ext = path.splitext(fname)[1]
    dump_mod = import_module("pastas.io" + ext)
    return dump_mod.dump(fname, data, **kwargs)
