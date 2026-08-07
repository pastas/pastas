"""Module containing the objective function for solvers to calculate residuals or noise."""

from collections.abc import Callable

from pandas import Series

from pastas.typing import ArrayLike, Model


def misfit(
    model: Model,
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
    model: object
        The model instance containing residuals and noise methods.
    p: np.ndarray
        Array of parameter values.
    noise: bool
        If True, minimizes the sum of squared noise computed by the NoiseModel.
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
    # Get the residuals or the noise
    res = model.residuals(p)
    if noise:
        res = model.noise(p=p, res=res) * model._noise_weights(p=p, res=res)

    # Apply weights if provided
    if weights is not None:
        weights = weights.reindex(res.index)
        weights.fillna(1.0, inplace=True)
        res = res.multiply(weights)

    # Call the callback function if provided
    if callback is not None:
        callback(p)

    # Return separate components if requested
    if returnseparate:
        return (
            res.to_numpy(copy=True),
            model.noise(p=p, res=res).to_numpy(copy=True),
            model._noise_weights(p=p, res=res).to_numpy(copy=True),
        )

    return res.to_numpy(copy=True)
