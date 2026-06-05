"""This module contains the objective function for solvers to calculate residuals or noise."""

from collections.abc import Callable

from pandas import Series

from pastas.typing import ArrayLike, Model


def misfit(
    ml: Model,
    p: ArrayLike,
    noise: bool,
    weights: Series | None = None,
    callback: Callable | None = None,
    returnseparate: bool = False,
) -> ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Shared objective function for solvers to calculate residuals or noise.

    Parameters
    ----------
    p: np.ndarray
        Array of parameter values.
    noise: bool
        If True, minimizes the sum of squared noise computed by the NoiseModel.
    ml: object
        The model instance containing residuals and noise methods.
    weights: pandas.Series, optional
        Weights to scale the residuals or noise.
    callback: Callable, optional
        Function to call after each iteration.
    returnseparate: bool, optional
        If True, returns residuals, noise, and noise weights separately.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray, np.ndarray]
        The calculated residuals or noise, optionally with separate components.
    """
    subtract_mean = not ml.settings["fit_constant"]

    # Get the residuals or the noise
    if noise:
        rv = ml.noise(p=p, subtract_mean=subtract_mean) * ml._noise_weights(
            p=p, subtract_mean=subtract_mean
        )
    else:
        rv = ml.residuals(p=p, subtract_mean=subtract_mean)

    # Apply weights if provided
    if weights is not None:
        weights = weights.reindex(rv.index)
        weights.fillna(1.0, inplace=True)
        rv = rv.multiply(weights)

    # Call the callback function if provided
    if callback is not None:
        callback(p)

    # Return separate components if requested
    if returnseparate:
        return (
            ml.residuals(p=p, subtract_mean=subtract_mean).to_numpy(copy=True),
            ml.noise(p=p, subtract_mean=subtract_mean).to_numpy(copy=True),
            ml._noise_weights(p=p, subtract_mean=subtract_mean).to_numpy(copy=True),
        )

    return rv.to_numpy(copy=True)
