"""Module for noise models.

Noise models may be used to transform the residual
series into a noise series that better represents
white noise.

Examples
--------
A noise model can be added to a Pastas model.::

    n = ps.ArmaNoiseModel(model=ml)

Or delete the noise model from the model::

    ml.del_noisemodel()

"""

from abc import ABC, abstractmethod
from logging import getLogger

import numpy as np
from pandas import DataFrame, DatetimeIndex, Series, Timedelta

from pastas.typing import ArrayLike, Model

from .decorators import (
    check_argument_model,
    deprecate_class_func_or_method,
    njit,
    set_parameter,
)

logger = getLogger(__name__)

__all__ = ["ArNoiseModel", "ArmaNoiseModel"]


class NoiseModelBase(ABC):
    """Base class for noise models."""

    def __init__(self, model: Model, name: str, norm: bool | None = None) -> None:
        self.model = model
        self.name = name
        self.norm = norm
        self.parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])

    @property
    def _name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def nparam(self) -> int:
        """Number of parameters of the noise model."""

    @abstractmethod
    def simulate(self, res: Series, p: ArrayLike) -> Series:
        """Simulate noise from the residual series."""

    @abstractmethod
    def set_init_parameters(self, oseries: Series | None = None) -> None:
        """Set initial noise model parameters."""

    @set_parameter
    def _set_initial(self, name: str, value: float) -> None:
        """Set the initial parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "initial"] = value

    @set_parameter
    def _set_pmin(self, name: str, value: float) -> None:
        """Set the minimum value of the noisemodel.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "pmin"] = value

    @set_parameter
    def _set_pmax(self, name: str, value: float) -> None:
        """Set the maximum parameter values.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "pmax"] = value

    @set_parameter
    def _set_vary(self, name: str, value: float) -> None:
        """Set if the parameter is varied.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "vary"] = value

    def _set_model(self, model: Model) -> None:
        """Set the Pastas Model for the noise model."""
        self.model = model

    def to_dict(self) -> dict:
        """Return a dict to store the noise model."""
        return {"class": self._name, "norm": self.norm}

    def weights(self, res: Series, p: ArrayLike) -> Series | int:
        _, _ = res, p
        return 1


class ArNoiseModel(NoiseModelBase):
    r"""Noise model with exponential decay of the residuals and weighting.

    Parameters
    ----------
    model: pastas.Model
        The Pastas Model instance to which the noise model is added.
    name: str, optional
        Name of the noise model. Default is "noise".
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

    @check_argument_model
    def __init__(self, model: Model, name: str = "noise", norm: bool = True) -> None:
        super().__init__(model=model, name=name, norm=norm)
        self.set_init_parameters()
        if model is not None:
            self.model._add_noisemodel(self)

    def set_init_parameters(self, oseries: Series | None = None) -> None:
        """Set initial parameters for the noise model.

        Parameters
        ----------
        oseries : pandas.Series, optional
            Observation series used to estimate initial parameters. If None,
            default values are used. Default is None.
        """
        if oseries is not None:
            pinit = np.diff(oseries.index.to_numpy()) / Timedelta("1D")
            pinit = np.median(pinit)
        else:
            pinit = 14.0
        self.parameters.loc[f"{self.name}_alpha"] = (
            pinit,
            1e-5,
            5000.0,
            True,
            self.name,
        )

    @property
    def nparam(self) -> int:
        """Number of parameters for the noise model.

        Returns
        -------
        int
            Number of parameters (1 for ArNoiseModel).
        """
        return 1

    def simulate(self, res: Series, p: ArrayLike) -> Series:
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
        resv = res.to_numpy()
        v = np.append(resv[0], resv[1:] - np.exp(-odelt / alpha) * resv[:-1])
        return Series(data=v, index=res.index, name="Noise")

    def weights(self, res: Series, p: ArrayLike) -> Series:
        r"""Calculate the weights for the noise.

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

        .. math::

            w = 1 / sqrt((1 - exp(-2 \\Delta t / \\alpha)))

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
        self,
        res: Series,
        p: ArrayLike,
        tindex: DatetimeIndex,
    ) -> Series:
        r"""Get correction for a forecast using the noise model.

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
        """Return a dict to store the noise model."""
        return super().to_dict()


@deprecate_class_func_or_method(
    version="2.0.0",
    reason="Please use `ps.ArNoiseModel` instead.",
)
def NoiseModel(*args, **kwargs) -> ArNoiseModel:
    n = ArNoiseModel(*args, **kwargs)
    return n


class ArmaNoiseModel(NoiseModelBase):
    r"""ARMA(1,1) Noise model to simulate the noise as defined in :cite:t:`collenteur_estimation_2021`.

    Warnings
    --------
    This model has only been tested on regular time steps and should not be used for
    irregular time steps yet.

    Notes
    -----
    Calculates the noise according to:

    .. math::

        \\upsilon_t = r_t - r_{t-1} e^{-\\Delta t/\\alpha} - \\upsilon_{t-1}
        e^{-\\Delta t/\\beta}

    The units of the alpha and beta parameters are always in days.
    """

    @check_argument_model
    def __init__(self, model: Model, name: str = "noise", norm: bool = True) -> None:
        super().__init__(model=model, name=name, norm=norm)
        self.set_init_parameters()
        if model is not None:
            self.model._add_noisemodel(self)

    @property
    def nparam(self) -> int:
        """Number of parameters for the noise model.

        Returns
        -------
        int
            Number of parameters (2 for ArmaNoiseModel: alpha and beta).
        """
        return 2

    def set_init_parameters(self, oseries: Series | None = None) -> None:
        """Set initial parameters for the noise model.

        Parameters
        ----------
        oseries : pandas.Series, optional
            Observation series used to estimate initial parameters. If None,
            default values are used. Default is None.
        """
        if oseries is not None:
            pinit = np.diff(oseries.index.to_numpy()) / Timedelta("1D")
            pinit = np.median(pinit)
        else:
            pinit = 14.0
        self.parameters.loc[f"{self.name}_alpha"] = (
            pinit,
            1e-9,
            5000.0,
            True,
            self.name,
        )
        self.parameters.loc[f"{self.name}_beta"] = (
            1.0,
            -np.inf,
            np.inf,
            True,
            self.name,
        )

    def simulate(self, res: Series, p: ArrayLike) -> Series:
        """Simulate noise from the residual series.

        Parameters
        ----------
        res : pandas.Series
            The residual series.
        p : array_like
            array_like object with the values as floats representing the model
            parameters. Here, Alpha and Beta parameters used by the noisemodel.

        Returns
        -------
        noise : pandas.Series
            Series of the noise.
        """
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
        """Calculate the noise values for the ARMA(1,1) noise model.

        Parameters
        ----------
        res : array_like
            Array of residual values.
        odelt : array_like
            Array of time steps between observations in days.
        alpha : float
            Alpha parameter for the noise model.
        beta : float
            Beta parameter for the noise model.

        Returns
        -------
        array_like
            Array of noise values.
        """
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

    def to_dict(self) -> dict:
        """Return a dict to store the noise model."""
        return super().to_dict()


@deprecate_class_func_or_method(
    version="2.0.0",
    reason="Please use `ps.ArmaNoiseModel` instead.",
)
def ArmaModel(*args, **kwargs) -> ArmaNoiseModel:
    n = ArmaNoiseModel(*args, **kwargs)
    return n
