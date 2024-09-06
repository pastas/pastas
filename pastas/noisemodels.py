"""This module contains the noise models available in Pastas.

A Noise model may be used to transform the residual series into a noise series that
better represents white noise.

Examples
--------
By default, a noise model is added to a Pastas model. It is possible to replace the
default model with different models as follows:

>>> n = ps.ArmaNoiseModel()
>>> ml.add_noisemodel(n)

or, to delete the noise model from the model:

>>> ml.del_noisemodel()

See Also
--------
pastas.model.Model.add_noisemodel
"""

from logging import getLogger
from typing import Optional

import numpy as np
from pandas import DataFrame, DatetimeIndex, Series, Timedelta

from pastas.typing import ArrayLike

from .decorators import PastasDeprecationWarning, njit, set_parameter

logger = getLogger(__name__)

__all__ = ["ArNoiseModel", "ArmaNoiseModel"]


class NoiseModelBase:
    _name = "NoiseModelBase"

    def __init__(self) -> None:
        self.nparam = 1
        self.name = "noise"
        self.norm = None
        self.parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )

    def set_init_parameters(self, oseries: Optional[Series] = None) -> None:
        if oseries is not None:
            pinit = np.diff(oseries.index.to_numpy()) / Timedelta("1D")
            pinit = np.median(pinit)
        else:
            pinit = 14.0
        self.parameters.loc["noise_alpha"] = (
            pinit,
            1e-5,
            5000.0,
            True,
            "noise",
            "uniform",
        )

    @set_parameter
    def _set_initial(self, name: str, value: float) -> None:
        """Internal method to set the initial parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "initial"] = value

    @set_parameter
    def _set_pmin(self, name: str, value: float) -> None:
        """Internal method to set the minimum value of the noisemodel.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "pmin"] = value

    @set_parameter
    def _set_pmax(self, name: str, value: float) -> None:
        """Internal method to set the maximum parameter values.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "pmax"] = value

    @set_parameter
    def _set_vary(self, name: str, value: float) -> None:
        """Internal method to set if the parameter is varied.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "vary"] = value

    @set_parameter
    def _set_dist(self, name: str, value: str) -> None:
        """Internal method to set distribution of prior of the parameter.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "dist"] = str(value)

    def to_dict(self) -> dict:
        """Method to return a dict to store the noise model"""
        data = {"class": self._name, "norm": self.norm}
        return data

    @staticmethod
    def weights(res, p) -> int:
        return 1


class ArNoiseModel(NoiseModelBase):
    """Noise model with exponential decay of the residuals and weighting.

    Parameters
    ----------
    norm: boolean, optional
        Boolean to indicate whether weights are normalized according to the Von
        Asmuth and Bierkens (2005) paper. Default is True.

    Notes
    -----
    Calculates the noise :cite:t:`von_asmuth_modeling_2005` according to:

    .. math::

        v(t_1) = r(t_1) - r(t_0) * \\exp(- \\Delta t / \\alpha)

    Calculates the weights as

    .. math::

        w = 1 / \\sqrt{(1 - \\exp(-2 \\Delta t / \\alpha))}

    The units of the alpha parameter is always in days. The first value of the noise
    is the residual (:math:`v(t=0=r(t=0)`). First weight is 1 / sig_residuals (i.e.,
    delt = infty). Normalization of weights as in :cite:t:`von_asmuth_modeling_2005`,
    optional.
    """

    _name = "ArNoiseModel"

    def __init__(self, norm: bool = True) -> None:
        NoiseModelBase.__init__(self)
        self.norm = norm
        self.nparam = 1
        self.set_init_parameters()

    @staticmethod
    def simulate(res: Series, p: ArrayLike) -> Series:
        """Simulate noise from the residuals.

        Parameters
        ----------
        res: pandas.Series
            The residual series.
        p: array_like
            array_like object with the values as floats representing the model
            parameters. Here, Alpha parameter used by the noisemodel.

        Returns
        -------
        noise: pandas.Series
            Series of the noise.
        """
        alpha = p[0]
        odelt = np.diff(res.index.to_numpy()) / Timedelta("1D")
        v = np.append(
            res.values[0], res.values[1:] - np.exp(-odelt / alpha) * res.values[:-1]
        )
        return Series(data=v, index=res.index, name="Noise")

    def weights(self, res: Series, p: ArrayLike) -> Series:
        """Method to calculate the weights for the noise.

        Parameters
        ----------
        res: pandas.Series
            Pandas Series with the residuals to compute the weights for. The Series
            index must be a DatetimeIndex.
        p: array_like
            NumPy array with the parameters used in the noise model.

        Returns
        -------
        w: pandas.Series
            Series of the weights.

        Notes
        -----
        Weights are

        .. math:: w = 1 / sqrt((1 - exp(-2 \\Delta t / \\alpha)))

        which are then normalized so that sum(w) = len(res).
        """
        alpha = p[0]
        # large for first measurement
        odelt = np.append(1e12, np.diff(res.index.to_numpy()) / Timedelta("1D"))
        exp = np.exp(-2.0 / alpha * odelt)  # Twice as fast as 2*odelt/alpha
        w = 1 / np.sqrt(1.0 - exp)  # weights of noise, not noise^2
        if self.norm:
            w *= np.exp(1.0 / (2.0 * odelt.size) * np.sum(np.log(1.0 - exp)))
        return Series(data=w, index=res.index, name="noise_weights")

    def get_correction(
        self, res: Series, p: ArrayLike, tindex: DatetimeIndex
    ) -> Series:
        """Get the correction for a forecast using the noise model.

        Parameters
        ----------
        res : Series
            The residual series.
        p : ArrayLike
            The parameters of the noise model.
        tindex : DatetimeIndex
            The index of the forecast.

        Returns
        -------
        Series
            The correction to the forecast.

        Notes
        -----
        The correction is calculated as:

        .. math::

                correction = \\exp(-\\Delta t / \\alpha) * last_residual

        where :math:`\\Delta t` is the time difference between the last observation
        and the forecast, and :math:`\\alpha` is the noise parameter.

        """
        alpha = p[0]
        last_residual = res.iat[-1]
        last_date = res.index[-1]
        dt = (tindex - last_date).days
        correction = Series(
            index=tindex,
            name="correction",
            dtype=float,
            data=np.exp(-dt / alpha) * last_residual,
        )
        return correction

    def to_dict(self) -> dict:
        """Method to return a dict to store the noise model"""
        data = {"class": self._name, "norm": self.norm}
        return data


@PastasDeprecationWarning(
    remove_version="2.0.0", reason="Please use `ps.ArNoiseModel` instead."
)
def NoiseModel(*args, **kwargs) -> ArNoiseModel:
    n = ArNoiseModel(*args, **kwargs)
    n._name = "NoiseModel"
    return n


class ArmaNoiseModel(NoiseModelBase):
    """ARMA(1,1) Noise model to simulate the noise as defined in
    :cite:t:`collenteur_estimation_2021`.

    Notes
    -----
    Calculates the noise according to:

    .. math::
        \\upsilon_t = r_t - r_{t-1} e^{-\\Delta t/\\alpha} - \\upsilon_{t-1}
        e^{-\\Delta t/\\beta}

    The units of the alpha and beta parameters are always in days.

    Warnings
    --------
    This model has only been tested on regular time steps and should not be used for
    irregular time steps yet.
    """

    _name = "ArmaNoiseModel"

    def __init__(self) -> None:
        NoiseModelBase.__init__(self)
        self.nparam = 2
        self.set_init_parameters()

    def set_init_parameters(self, oseries: Series = None) -> None:
        if oseries is not None:
            pinit = np.diff(oseries.index.to_numpy()) / Timedelta("1D")
            pinit = np.median(pinit)
        else:
            pinit = 14.0
        self.parameters.loc["noise_alpha"] = (
            pinit,
            1e-9,
            5000.0,
            True,
            "noise",
            "uniform",
        )
        self.parameters.loc["noise_beta"] = (
            1.0,
            -np.inf,
            np.inf,
            True,
            "noise",
            "uniform",
        )

    def simulate(self, res: Series, p: ArrayLike) -> Series:
        alpha = p[0]
        beta = p[1]

        # Calculate the time steps
        odelt = np.diff(res.index.to_numpy()) / Timedelta("1D")
        a = self.calculate_noise(res.values, odelt, alpha, beta)
        return Series(index=res.index, data=a, name="Noise")

    @staticmethod
    @njit
    def calculate_noise(
        res: ArrayLike, odelt: ArrayLike, alpha: float, beta: float
    ) -> ArrayLike:
        # Create an array to store the noise
        a = np.zeros_like(res)
        a[0] = res[0]

        if beta == 0.0:  # Prevent division by zero errors
            beta = 1e-24

        pm = beta / np.abs(beta)

        # We have to loop through each value
        for i in range(1, res.size):
            a[i] = (
                res[i]
                - res[i - 1] * np.exp(-odelt[i - 1] / alpha)
                - a[i - 1] * pm * np.exp(-odelt[i - 1] / np.abs(beta))
            )
        return a


@PastasDeprecationWarning(
    remove_version="2.0.0", reason="Please use `ps.ArmaNoiseModel` instead."
)
def ArmaModel(*args, **kwargs) -> ArmaNoiseModel:
    n = ArmaNoiseModel(*args, **kwargs)
    n._name = "ArmaModel"
    return n
