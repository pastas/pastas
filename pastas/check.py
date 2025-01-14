import logging
from typing import Callable, Optional

import numpy as np
from matplotlib.colors import rgb2hex
from pandas import DataFrame, concat

from pastas.model import Model
from pastas.rfunc import RfuncBase
from pastas.stats import tests as diagnostic_tests

logger = logging.getLogger(__name__)


operators = {
    "greater_equal": ">=",
    "less_equal": ">=",
    "greater_than": ">",
    "less_than": "<",
    "equal": "==",
    "not_equal": "!=",
}


def _stat_ufunc_threshold(
    ml: Model,
    ufunc: Callable,
    statistic: str,
    threshold: float,
):
    """Generic function to compare a model statistic with a threshold using a ufunc.

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    ufunc: callable
        Numpy ufunc (e.g. np.greater_than) to compare the model statistic with the
        threshold.
    statistic: str
        Name of the statistic to be compared.
    threshold: float
        Threshold value for the statistic.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the results of the check.
    """
    val = getattr(ml.stats, statistic)()
    check = ufunc(val, threshold)
    label = f"{statistic}{operators[ufunc.__name__]}{threshold}"
    df = DataFrame(
        index=[label],
        columns=["statistic", "operator", "threshold", "dimensions", "pass", "comment"],
    )
    df.index.name = "check"
    df.loc[label] = [val, operators[ufunc.__name__], threshold, "-", check, ""]
    return df


def rsq_geq_threshold(ml: Model, threshold: float = 0.7):
    """Check R^2 >= threshold.

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    threshold: float
        Threshold value for the R^2 statistic.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the results of the check.
    """
    return _stat_ufunc_threshold(ml, np.greater_equal, "rsq", threshold)


def response_memory(
    ml,
    cutoff: float = 0.95,
    factor_length_oseries: float = 0.5,
    names: Optional[list[str] | str] = None,
):
    """Check if response function memory is shorter than fraction of calibration period.

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    cutoff: float, optional
        Cutoff value for the length of the response function. Default is 0.95, which
        means that the response function is cutoff at the time the step response is at
        95% of the gain.
    factor_length_oseries: float, optional
        Factor to multiply the length of the observation series with to get the
        maximum allowed memory. Default is 0.5, e.g. half of the calibration period.
    names: list or str, optional
        List of stressmodel names to check the memory for. Default is None, which
        means all stressmodels are checked.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the results of the check.
    """
    len_oseries_calib = (ml.settings["tmax"] - ml.settings["tmin"]).days

    if names is None:
        names = list(ml.stressmodels.keys())
    elif names is not None and not isinstance(names, list):
        names = [names]

    df = DataFrame(
        columns=["statistic", "operator", "threshold", "dimensions", "pass", "comment"]
    )
    df.index.name = "check"
    # unit = "days"
    dim = "[T]"

    def interp_step(cutoff: float, p: np.ndarray[float], rfunc: RfuncBase):
        """Helper function to interpolate the step response to compute the memory.

        Parameters
        ----------
        cutoff: float
            Compute the time to for this cutoff value for the step response.
        p: array
            Parameters of the response function.
        rfunc: pastas.rfunc
            Response function instance.

        Returns
        -------
        tmem: float
            Time to the cutoff value, i.e. the memory of the response function.
        """
        t = rfunc.get_t(p, dt=1.0, cutoff=1.0 - (1.0 - cutoff) / 10.0)
        step = rfunc.step(p, cutoff=1.0 - (1.0 - cutoff) / 10.0) / sm.rfunc.gain(p)
        tmem = np.interp(cutoff, step, t)
        return tmem

    for sm_name in names:
        sm = ml.stressmodels[sm_name]
        if sm._name == "WellModel":
            # HantushWellModel means the response function changes with distance
            # and therefore the memory is also distance dependent. So we compute
            # the memory for each well in the Wellmodel separately.
            nwells = sm.distances.index.size
            for iw in range(nwells):
                lbl = (
                    f"t{cutoff * 100:.0f}_{sm_name} ({sm.distances.index[iw]}) <"
                    f" {factor_length_oseries} Δt_calib"
                )
                p = sm.get_parameters(ml, istress=iw)
                tmem = interp_step(cutoff, p, sm.rfunc)
                check = tmem < factor_length_oseries * len_oseries_calib
                df.loc[lbl] = [
                    tmem,
                    "<",
                    factor_length_oseries * len_oseries_calib,
                    dim,
                    check,
                    "",
                ]
        elif sm.rfunc._name == "Hantush":
            # get_tmax is a conservative approximation for Hantush,
            # so it is better to interpolate step response to compute the memory
            lbl = f"response_t{cutoff * 100:.0f}_{sm_name}"
            p = ml.get_parameters(sm_name)
            tmem = interp_step(cutoff, p, sm.rfunc)
            check = tmem < factor_length_oseries * len_oseries_calib
            df.loc[lbl] = [
                tmem,
                factor_length_oseries * len_oseries_calib,
                dim,
                check,
                "",
            ]
        else:
            # for response functions where get_tmax is exact
            tmem = ml.get_response_tmax(sm_name, cutoff=cutoff)
            check = tmem < factor_length_oseries * len_oseries_calib
            lbl = f"t{cutoff * 100:.0f}_{sm_name} < {factor_length_oseries} Δt_calib"
            df.loc[lbl] = [
                tmem,
                "<",
                factor_length_oseries * len_oseries_calib,
                dim,
                check,
                "",
            ]
    return df


def uncertainty_gain(
    ml: Model,
    n_std: float = 1.96,
    names: Optional[list[str] | str] = None,
):
    """Check if the gain is larger than n_std times the uncertainty in the gain.

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    n_std: int, optional
        Number of standard errors to compare the the gain to. Default is 1.96.
    names: list or str, optional
        List of stressmodel names to check the gain for. Default is None, which
        means all stressmodels are checked.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the results of the check.
    """
    if names is None:
        names = list(ml.stressmodels.keys())
    elif names is not None and not isinstance(names, list):
        names = [names]

    results = []
    for sm_name in names:
        results.append(_uncertainty_parameter(ml, sm_name + "_A", n_std=n_std))
    df = concat(results)
    df["dimensions"] = "[L] / (unit stress)"
    return df


def parameter_bounds(ml: Model, parameters: Optional[list[str] | str] = None):
    """Check if the optimal parameter values are not on the lower or upper bounds.

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    parameters: list or str, optional
        List of parameter names to check the bounds for. Default is None, which
        means all parameters are checked.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the results of the check.
    """
    if parameters is None:
        parameters = ml.parameters.index.tolist()
    elif isinstance(parameters, str):
        parameters = [parameters]
    df = DataFrame(
        columns=["statistic", "operator", "threshold", "dimensions", "pass", "comment"]
    )
    df.index.name = "check"
    upper, lower = ml._check_parameters_bounds()
    for param in parameters:
        bounds = (
            ml.parameters.loc[param, "pmin"],
            ml.parameters.loc[param, "pmax"],
        )
        check = ~(upper.loc[param] or lower.loc[param])

        df.loc[f"Bounds: {param}"] = (
            ml.parameters.loc[param, "optimal"],
            "within",
            bounds,
            guess_unit_or_dims(param),
            check,
            "",
        )
    return df


def uncertainty_parameters(
    ml: Model,
    parameters: Optional[list[str] | str] = None,
    n_std: float = 1.96,
):
    """Check if parameter value is larger than n_std times the standard deviation.

    Note that it is the modelers responsibility to check if the estimated uncertainty
    is reliable!

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    parameters: list or str, optional
        List of parameter names to check the uncertainty for. Default is None, which
        means all parameters are checked.
    n_std: float, optional
        Number of standard deviations to compare the parameter to. Default is 1.96.

    """
    if parameters is None:
        parameters = ml.parameters.index.tolist()
    elif isinstance(parameters, str):
        parameters = [parameters]

    # loop through parameters
    results = []
    for parameter in parameters:
        results.append(_uncertainty_parameter(ml, parameter, n_std=n_std))
    return concat(results)


def _uncertainty_parameter(ml, parameter, n_std=1.96):
    """Internal method to check if parameter value is larger than n_std * std.

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    parameter: str
        Name of the parameter to check.
    n_std: float, optional
        Number of standard deviations to compare the parameter to. Default is 1.96.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the results of the check.
    """
    df = DataFrame(
        columns=["statistic", "operator", "threshold", "dimensions", "pass", "comment"]
    )
    df.index.name = "check"
    # get stressmodel
    sm_name = "_".join(parameter.split("_")[:-1])
    if sm_name in ml.stressmodels:
        sm = ml.stressmodels[sm_name]
    else:
        sm = None
    # WellModel gain is a special case, because the parameter depends on distance
    if sm is not None and sm._name == "WellModel" and parameter.endswith("_A"):
        nwells = sm.distances.index.size
        for iw in range(nwells):
            params = sm.get_parameters(model=ml, istress=iw)
            p = sm.rfunc.gain(params)
            std = sm.variance_gain(model=ml, istress=iw)
            check = np.abs(p) > (n_std * std)
            df.loc[f"|{parameter} ({sm.distances.index[iw]})| > {n_std}σ"] = [
                p,
                ">",
                n_std * std,
                guess_unit_or_dims(parameter),
                check,
                "Assumes estimate of σ is reliable.",
            ]
    else:
        p = ml.parameters.loc[parameter, "optimal"]
        std = ml.parameters.loc[parameter, "stderr"]
        check = np.abs(p) > (n_std * std)
        df.loc[f"|{parameter}| > {n_std}σ"] = [
            p,
            ">",
            n_std * std,
            guess_unit_or_dims(parameter),
            check,
            "Assumes estimate of σ is reliable.",
        ]
    return df


def guess_unit_or_dims(parameter, return_dims=True):
    """Guess the unit or dimension of a parameter based on its name.

    Parameters
    ----------
    parameter : str
        Name of the parameter.
    return_dims : bool, optional
        if True, return parameter dimensions (default). If False, return
        specific units where they can be determined from the parameter name

    Returns
    -------
    unit or dim: str
        Guessed unit or dimensions of the parameter.
    """
    if "_A" in parameter:
        sm = "_".join(parameter.split("_")[:-1])
        unit = dim = f"[L] / (unit '{sm}' stress)"
    elif parameter == "noise_alpha":
        unit = "days"
        dim = "[T]"
    elif parameter == "constant_d":
        unit = dim = "[L]"
    elif parameter.endswith("_f"):
        unit = "-"
        dim = "[-]"
    else:
        unit = ""
        dim = ""
    return dim if return_dims else unit


def acf_runs_test(ml: Model, p_threshold: float = 0.05):
    """Runs test to check if there is significant autocorrelation in the noise.

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    p_threshold: float, optional
        Threshold value for the p-value of the runs test. Default is 0.05.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the results of the check.
    """
    return _diagnostic_test(ml, "runs_test", alpha=p_threshold)


def acf_stoffer_toloi_test(ml: Model, p_threshold: float = 0.05, **kwargs):
    """Stoffer-Toloi test to check if there is significant autocorrelation in the noise.

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    p_threshold: float, optional
        Threshold value for the p-value of the Stoffer-Toloi test. Default is 0.05.
    **kwargs
        Additional keyword arguments to pass to the Stoffer-Toloi test.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the results of the check.
    """
    return _diagnostic_test(ml, "stoffer_toloi", alpha=p_threshold, **kwargs)


def _diagnostic_test(ml: Model, test: str, alpha: float = 0.05, **kwargs):
    """Internal method to get the result of a diagnostic test.

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    test: str
        Name of the diagnostic test to perform, must be one of
        ["runs_test", "stoffer_toloi", "durbin_watson", "ljung_box"].
    alpha: float, optional
        Threshold value for the p-value of the diagnostic test. Default is 0.05.
    **kwargs
        Additional keyword arguments to pass to the diagnostic test.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the results of the check.
    """
    dtest = getattr(diagnostic_tests, test)
    noise = ml.noise()
    if noise is None:
        logger.warning("No noise model found in model. Using residuals instead.")
        noise = ml.residuals()
    _, p = dtest(noise.iloc[1:], **kwargs)
    check = p > alpha
    label = f"{test} (p > α)"
    df = DataFrame(
        index=[label],
        columns=["statistic", "operator", "threshold", "dimensions", "pass", "comment"],
    )
    df.index.name = "check"
    df.loc[label] = [p, ">", alpha, "-", check, ""]
    return df


def checklist(ml: Model, checks: list[str | Callable | dict], report=True):
    """Run a list of checks on a Pastas model.

    Parameters
    ----------
    ml: pastas.Model
        Pastas model instance.
    checks: list
        List of checks to perform. Each entry in the list can be a string, a callable,
        or a dict:
           * If a string, it must be the name of a function in this module,
             e.g. "rsq_geq_threshold".
           * If a callable, it must be a function that takes a model instance as an
             argument and return a DataFrame with columns:
             ["statistic", "operator", "threshold", "dimensions", "pass", "comment"]].
           * If a dict, it must have a key "func" with a function to perform the check,
             additional dictionary entries are passed to the function as kwargs.
    report: bool, optional
        If True, display a report of the check results. Default is True.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the results of the checks.
    """

    results = []
    for check in checks:
        if isinstance(check, str):
            results.append(globals()[check](ml))
        elif isinstance(check, dict):
            check = check.copy()  # copy so we do not modify original dict
            func = check.pop("func")
            if isinstance(func, str):
                func = globals()[func]  # get function from this module
            results.append(func(ml, **check))  # pass rest of dict to function
        elif callable(check):
            results.append(check(ml))  # call function with model as only argument
        else:
            raise TypeError("Check must be str, callable, or dict.")
    df = concat(results)
    if report:
        print_check_report(df)
    return df


def print_check_report(df):
    """Print a report of the check results.

    The check result is colored red (fail), green (pass) or yellow (pass with
    conditions) based on the outcome.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame with the results of the checks.
    """

    def boolean_row_styler(row, column):
        """Styler function to color rows based on the value in column."""

        colors = [""] * row.size

        # make result based on std deviation check yellow, because we cannot
        # be certain the parameter uncertainty estimate is reliable
        if row[column] and "σ" in row.name:
            colors[row.size - 2] = (
                "background-color: lemonchiffon; color: darkgoldenrod"
            )
        elif row[column]:  # check passed
            colors[row.size - 2] = (
                f"background-color: {rgb2hex((231 / 255, 255 / 255, 239 / 255))}; "
                "color: darkgreen"
            )
        else:  # check failed
            colors[row.size - 2] = (
                f"background-color: {rgb2hex((255 / 255, 238 / 255, 238 / 255))}; "
                "color: darkred"
            )
        return colors

    # try pretty display, otherwise fall back to print
    try:
        from IPython.display import display

        display(df.style.apply(boolean_row_styler, column="pass", axis=1))
    except ModuleNotFoundError:
        print(df)


# list of checks, based on Brakenhoff et al. (2022).
checks_brakenhoff_2022 = [
    {"func": rsq_geq_threshold, "threshold": 0.7},
    {"func": response_memory, "cutoff": 0.95, "factor_length_oseries": 0.5},
    {"func": acf_runs_test},
    {"func": uncertainty_parameters, "n_std": 1.96},
    {"func": parameter_bounds},
]
