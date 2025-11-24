"""This module contains methods to generate forecasts using a Pastas model instance.

Examples
--------
Generate forecasts using ensembles of stress forecasts::

    forecasts = ...  # dictionary or list of dataframes with time series forecasts
    ps.forecast(ml, forecasts)

"""

from logging import getLogger

from numpy import array, empty, exp, linspace, ones
from pandas import DataFrame, DatetimeIndex, MultiIndex, Timedelta, Timestamp, concat

from pastas.noisemodels import ArNoiseModel
from pastas.typing import ArrayLike, Model

logger = getLogger(__name__)


def _check_forecast_data(
    forecasts: dict[str, list[DataFrame]],
) -> tuple[int, Timestamp | str, Timestamp | str, DatetimeIndex]:
    """Internal method to check the integrity of the forecasts data.

    Parameters
    ----------
    forecasts: dict
        Dictionary containing the forecasts data. The keys are the stressmodel names
        and the values are lists of DataFrames containing the forecasts with a datetime
        index and each column a time series (i.e., one ensemble member).

    Returns
    -------
    n: int
        The number of ensemble members in the forecasts.
    tmin: datetime
        The minimum datetime in the forecasts.
    tmax: datetime
        The maximum datetime in the forecasts.
    index: DatetimeIndex
        The datetime index of the forecasts.

    Notes
    -----
    This method checks if the number of columns and indices are the same for all
    DataFrames in the forecasts dictionary. If the number of columns or the indices are
    not the same, a warning is printed and a ValueError is raised.

    """
    # Input validation
    if not isinstance(forecasts, dict) or not forecasts:
        msg = "Forecasts must be a non-empty dictionary"
        logger.error(msg)
        raise ValueError(msg)

    n = None
    tmax = None
    tmin = None
    index = None

    for sm_name, fc_data in forecasts.items():
        if not fc_data:
            msg = f"No forecast data provided for stressmodel '{sm_name}'"
            logger.warning(msg)
            continue

        for fc in fc_data:
            # Check if DataFrame is empty
            if fc.empty:
                msg = f"Empty DataFrame in forecasts for stressmodel '{sm_name}'"
                logger.warning(msg)
                continue

            # Check if the number of columns is the same for all DataFrames
            if n is None:
                n = fc.columns.size
                tmin = fc.index[0]
                tmax = fc.index[-1]
                index = fc.index
                logger.debug(f"First forecast found with {n} ensemble members")
            # If the number of columns is not the same, raise an error
            elif n != fc.columns.size:
                msg = (
                    f"The number of ensemble members is not the same for all forecasts. "
                    f"Expected {n}, got {fc.columns.size} in stressmodel '{sm_name}'."
                )
                logger.error(msg)
                raise ValueError(msg)
            elif tmin != fc.index[0] or tmax != fc.index[-1]:
                msg = (
                    "The time index of the forecasts is not the same for all forecasts."
                    "Please check the forecast data."
                )
                logger.error(msg)
                raise ValueError(msg)

    if n is None:
        msg = "No valid forecast data found in any of the stressmodels"
        logger.error(msg)
        raise ValueError(msg)

    return n, tmin, tmax, index


def forecast(
    ml: Model,
    forecasts: dict[str, list[DataFrame]],
    p: ArrayLike | None = None,
    post_process: bool = False,
) -> DataFrame:
    """Method to forecast the head from ensembles of stress forecasts.

    Parameters
    ----------
    ml: pastas.Model
        Pastas Model instance.
    forecasts: dict
        Dictionary containing the forecasts data. The keys are the stressmodel names
        and the values are lists of DataFrames containing the forecasts with a datetime
        index and each column a time series (i.e., one ensemble member).
    p: array_like, optional
        List of parameter sets to use for the forecasts. If None, a single parameter set is used that defaults to the optimal model parameters. Default is None.
    post_process: bool, optional
        If True, the forecasts are post-processed using the noise model of the model
        instance. Default is False. If True, a noise model should be present in the
        model instance. If no noisemodel is present and post_process is True, an error
        is raised.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame containing the forecasts. The columns are a MultiIndex with the first
        level the ensemble member, the second level the parameter member, and the third level the mean and the variance of each forecast member.

    Notes
    -----
    For efficiency, the iteration over the different ensemble members and parameters sets is done in the following order using a double for-loop:

    1. iterate over the ensemble members
    2. iterate over the parameter sets

    All computations that only depend on the parameter sets and not on the ensemble members are performed outside this double for-loop.

    Please note that only the AR1 noise model is supported at this moment for post-processing.

    """
    # Check the integrity of the forecasts data
    n, tmin, tmax, index = _check_forecast_data(forecasts)
    logger.info(f"Working with {n} ensemble members from {tmin} to {tmax}")

    if post_process and not isinstance(ml.noisemodel, ArNoiseModel):
        if ml.noisemodel is None:
            msg = "No noise model present in the model instance. Please add a noise model to the model instance or set post_process=False."
        else:
            msg = "Only the AR1 noise model is supported for post-processing at this moment. Please use an AR1 noise model or set post_process=False."
        logger.error(msg)
        raise ValueError(msg)

    # Check which parameters sets should be used.
    if p is None:
        logger.info("No parameter provided, using the optimal parameters.")
        # In case no parameters are provided, use optimal values
        p = [ml.parameters.loc[:, "optimal"].values]
        nparam = len(p)
    else:
        if len(p) == 0:
            msg = "Empty parameter list provided"
            logger.error(msg)
            raise ValueError(msg)
        nparam = len(p)
        logger.info(f"Using {nparam} provided parameter sets")

    # Pre-allocate arrays for results to avoid append operations
    forecast_length = len(index)
    result_array = empty((n * nparam * 2, forecast_length))

    # Pre-compute residuals for each parameter set since they only depend on parameters
    logger.info("Pre-computing residuals for each parameter set")

    residuals = {}
    vars = {}
    day = Timedelta("1D")

    if post_process:
        dt = ml.settings["freq_obs"] / day
        t = linspace(1, index.size, index.size)
        correction = {}

    # Preprocess residuals and variances for each parameter set as they only depend on parameters and not on ensemble members
    for i, param in enumerate(p):
        residuals[i] = ml.residuals(tmax=tmin, p=param).dropna()

        if post_process:
            # Compute the time varying variance for the AR1 noise model
            phi = exp(-dt / param[-1])
            denominator = 1.0 - phi**2
            phi_scaling_factor = (1.0 - phi ** (2.0 * t / dt)) / denominator
            vars[i] = ml.noise(tmax=tmin, p=param).var() * phi_scaling_factor

            correction[i] = ml.noisemodel.get_correction(
                residuals[i], [param[-1]], index
            ).values
        else:
            vars[i] = residuals[i].var() * ones(forecast_length)

    # Copy the model so old model is unaffected when replacing the stresses.
    ml = ml.copy()
    idx = 0

    # 1. iterate over the ensemble members
    for member in range(n):
        # Update stresses with ensemble member data
        for sm_name, fc_data in forecasts.items():
            sm = ml.stressmodels[sm_name]  # Select stressmodel
            for i, fc in enumerate(fc_data):
                ts = concat(
                    [
                        sm.stress_tuple[i].series_original.loc[: tmin - day],
                        fc.iloc[:, member],
                    ]
                )
                sm.stress = ts

        # 2. iterate over the parameter sets
        for i, param in enumerate(p):
            # Generate the forecasts
            sim = ml.simulate(tmin=tmin, tmax=tmax, p=param).values

            if post_process:
                # Add the correction from the noise model
                sim = sim + correction[i]

            # Store in pre-allocated array instead of appending to list
            result_array[idx : idx + 2] = array([sim, vars[i]])
            idx += 2  # Bump index by 2 for mean and variance

    # Create DataFrames to store data
    mi = MultiIndex.from_product(
        [range(n), range(nparam), ["mean", "var"]],
        names=["ensemble_member", "param_member", "forecast"],
    )
    df = DataFrame(data=result_array.T, index=index, columns=mi, dtype=float)

    return df


def get_overall_mean_and_variance(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Method to get the overall mean and variance of the forecast ensemble.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the forecasts. The columns are a MultiIndex with the first
        level the ensemble member, the second level the parameter member, and the third
        level the mean and the variance of each forecast member. The index is a
        DatetimeIndex with the time steps of the forecasts.

    Returns
    -------
    overall_mean: pandas.Series
        Series with the mean of the forecasts.
    overall_variance: pandas.Series
        Series with the variance of the forecasts.

    Notes
    -----
    This method is used to get the overall mean and variance of the forecasts. The
    mean and variance are calculated from the ensemble members and parameter members
    using the law of total variance.

    Example
    -------
    Simple example showing how to use the get_overall_mean_and_variance function::

        import pastas as ps
        import pandas as pd
        import numpy as np
        from pastas.forecast import get_overall_mean_and_variance

        # Create a sample DataFrame with forecasts
        index = pd.date_range("2023-01-01", periods=10, freq="D")
        data = np.random.rand(10, 6)
        columns = pd.MultiIndex.from_product(
            [range(3), range(2), ["mean", "var"]],
            names=["ensemble_member", "param_member", "forecast"],
        )
        df = pd.DataFrame(data=data, index=index, columns=columns)

        # Call the function to get the overall mean and variance
        mean, var = get_overall_mean_and_variance(df)
        print(mean)

    """
    means = df.loc[:, (slice(None), slice(None), "mean")]
    variances = df.loc[:, (slice(None), slice(None), "var")]

    overall_mean = means.mean(axis=1)

    # variance of the means
    variance_of_means = means.var(axis=1)

    # mean of the variances
    mean_of_variances = variances.mean(axis=1)

    # total variance
    overall_variance = variance_of_means + mean_of_variances

    return overall_mean, overall_variance
