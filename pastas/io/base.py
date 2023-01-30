"""Import model."""

from importlib import import_module
from logging import getLogger
from os import path

from numpy import log
from packaging import version
from pandas import to_numeric

import pastas as ps

# Type Hinting
from pastas.typing import Model
from .timeseries_legacy import TimeSeriesOld

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
        logger.error("File not found: %s", fname)

    # Dynamic import of the export module
    load_mod = import_module(f"pastas.io{path.splitext(fname)[1]}")

    # Get dicts for all data sources
    data = load_mod.load(fname, **kwargs)

    ml = _load_model(data)

    file_version = data["file_info"]["pastas_version"]
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

    if "noisemodel" in data.keys():
        noise = True
    else:
        noise = False

    ml = ps.Model(
        oseries=oseries,
        constant=constant,
        noisemodel=noise,
        name=name,
        metadata=metadata,
    )

    if "settings" in data.keys():
        ml.settings.update(data["settings"])
    if "file_info" in data.keys():
        ml.file_info.update(data["file_info"])

    # Add stressmodels
    for name, smdata in data["stressmodels"].items():
        sm = _load_stressmodel(smdata, data)
        ml.add_stressmodel(sm)

    # Add transform
    if "transform" in data.keys():
        # Todo Deal with old files. Remove in pastas 1.0
        if "transform" in data["transform"].keys():
            data["transform"]["class"] = data["transform"].pop("transform")

        transform = getattr(ps.transform, data["transform"].pop("class"))
        transform = transform(**data["transform"])
        ml.add_transform(transform)

    # Add noisemodel if present
    if "noisemodel" in data.keys():
        # Todo Deal with old files. Remove in pastas 1.0
        if "type" in data["noisemodel"].keys():
            data["noisemodel"]["class"] = data["noisemodel"].pop("type")

        n = getattr(ps.noisemodels, data["noisemodel"].pop("class"))()
        ml.add_noisemodel(n)

    # Add fit object to the model
    if "fit" in data.keys():
        # Todo Deal with old files. Remove in pastas 1.0
        if "name" in data["fit"].keys():
            data["fit"]["class"] = data["fit"].pop("name")

        solver = getattr(ps.solver, data["fit"].pop("class"))
        ml.fit = solver(**data["fit"])
        ml.fit.set_model(ml)

    # Add parameters, use update to maintain correct order
    ml.parameters = ml.get_init_parameters(noise=ml.settings["noise"])
    ml.parameters.update(data["parameters"])
    ml.parameters = ml.parameters.apply(to_numeric, errors="ignore")

    # When initial values changed
    for param, value in ml.parameters.loc[:, "initial"].items():
        ml.set_parameter(name=param, initial=value)

    return ml


def _load_stressmodel(ts, data):
    # Todo Deal with old files. Remove in pastas 1.0
    if "stressmodel" in ts.keys():
        ts["class"] = ts.pop("stressmodel")

    # TODO Deal with old StressModel2 files for version 0.22.0. Remove in 0.23.0.
    if ts["class"] == "StressModel2":
        msg = (
            "StressModel2 is removed since Pastas 0.22.0 and is replaced by the "
            "RechargeModel using a Linear recharge model. Make sure to save "
            "this file first using Pastas version 0.22.0 as this file cannot be "
            "loaded in newer Pastas versions. This will automatically update "
            "your model to the newer RechargeModel stress model."
        )
        logger.error(msg=msg)
        raise NotImplementedError(msg)

    # TODO Deal with old parameter value b in HantushWellModel: b_new = np.log(b_old)
    if (ts["class"] == "WellModel") and (
        version.parse(data["file_info"]["pastas_version"]) < version.parse("0.22.0")
    ):
        logger.warning(
            "The value of parameter 'b' in HantushWellModel was modified in 0.22.0: "
            "b_new = log(b_old). The value of 'b' is automatically updated on load."
        )
        wnam = ts["name"]
        for pcol in ["initial", "optimal", "pmin", "pmax"]:
            if wnam + "_b" in data["parameters"].index:
                if data["parameters"].loc[wnam + "_b", pcol] > 0:
                    data["parameters"].loc[wnam + "_b", pcol] = log(
                        data["parameters"].loc[wnam + "_b", pcol]
                    )

    # Deal with old-style response functions (TODO remove in 1.0)
    if version.parse(data["file_info"]["pastas_version"]) < version.parse("0.23.0"):
        if "rfunc" in ts.keys():
            rfunc_kwargs = ts.pop("rfunc_kwargs", {})
            rfunc_kwargs["class"] = ts["rfunc"]
            if "cutoff" in ts.keys():
                rfunc_kwargs["cutoff"] = ts.pop("cutoff")
            ts["rfunc"] = rfunc_kwargs
        if "recharge" in ts.keys():
            recharge_kwargs = ts.pop("recharge_kwargs", {})
            recharge_kwargs["class"] = ts["recharge"]
            ts["recharge"] = recharge_kwargs

    # Create and add stress model
    stressmodel = getattr(ps.stressmodels, ts.pop("class"))

    if "rfunc" in ts.keys():
        rfunc_class = ts["rfunc"].pop("class")  # Determine response class
        ts["rfunc"] = getattr(ps.rfunc, rfunc_class)(**ts["rfunc"])

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

    # Deal with pas-files from Pastas version 0.22. Pastas 0.22.0 was very loose on
    # the input data and would internally fix a lot. Here we choose to recreate the
    # old TimeSeries object, and use the TimeSeries.series.

    if "freq_original" in data.keys():
        msg = (
            "Whoops, looks like an old pas-file using the old TimeSeries format. "
            "Pastas will convert to the new TimeSeries format. However, it can not "
            "be guaranteed that the conversion will result in the exact same results. "
            "If you have the Python scripts used to generate this pas-file, it is "
            "highly recommended to rerun the script using a newer Pastas version ("
            "0.23 or higher). "
        )
        logger.warning(msg)

        # Create an old TimeSeries object
        series = TimeSeriesOld(**data).series

        # Remove deprecated keywords
        settings.pop("norm")
        data.pop("freq_original")

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
