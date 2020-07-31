"""The following noise models are available in Pastas. A Noise model may be
used to transform the residual series into a noise series that better
represents white noise.

Supported Noise Models
----------------------

.. autosummary::
    :nosignatures:
    :toctree: ./generated

    NoiseModel
    ArmaModel

Examples
--------
By default, a noise model is added to a Pastas model. It is possible to
replace the default model with different models as follows:

>>> n = ps.ArmaModel()
>>> ml.add_noisemodel(n)

or, to delete the noise model from the model:

>>> ml.del_noisemodel()

See Also
--------
pastas.model.Model.add_noisemodel

"""

import numpy as np
from pandas import Timedelta, DataFrame, Series

from .decorators import set_parameter, njit

__all__ = ["NoiseModel", "NoiseModelOld", "ArmaModel"]


class NoiseModelBase:
    _name = "NoiseModelBase"

    def __init__(self):
        self.nparam = 1
        self.name = "noise"
        self.parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])

    def set_init_parameters(self, oseries=None):
        if oseries is not None:
            pinit = oseries.index.to_series().diff() / Timedelta(1, "D")
            pinit = pinit.median()
        else:
            pinit = 14.0
        self.parameters.loc["noise_alpha"] = (pinit, 1e-5, 5000, True, "noise")

    @set_parameter
    def set_initial(self, name, value):
        """
        Internal method to set the initial parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        self.parameters.loc[name, "initial"] = value

    @set_parameter
    def set_pmin(self, name, value):
        """
        Internal method to set the minimum value of the noisemodel.

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        self.parameters.loc[name, "pmin"] = value

    @set_parameter
    def set_pmax(self, name, value):
        """
        Internal method to set the maximum parameter values.

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        self.parameters.loc[name, "pmax"] = value

    @set_parameter
    def set_vary(self, name, value):
        """
        Internal method to set if the parameter is varied.

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        self.parameters.loc[name, "vary"] = value

    def to_dict(self):
        return {"type": self._name}

    @staticmethod
    def weights(res, parameters):
        return 1


class NoiseModel(NoiseModelBase):
    """
    Noise model with exponential decay of the residuals and weighting.
    Differences compared to NoiseModelOld:

    1. First value is residual
    2. First weight is 1 / sig_residuals (i.e., delt = infty)
    3. Normalization of weights as in Von Asmuth and Bierkens (2005), optional

    Parameters
    ----------
    norm: boolean
        Boolean to indicate whether weights are normalized according to
        the Von Asmuth and Bierkens (2005) paper.

    Notes
    -----
    Calculates the noise [1]_ according to:

    .. math::

        v(t_1) = r(t_1) - r(t_0) * \\exp(- \\Delta t / \\alpha)

    Calculates the weights as

    .. math::

        w = 1 / \\sqrt{(1 - \\exp(-2 \\Delta t / \\alpha))}

    The units of the alpha parameter is always in days.

    References
    ----------
    .. [1] von Asmuth, J. R., and M. F. P. Bierkens (2005), Modeling
           irregularly spaced residual series as a continuous stochastic
           process, Water Resour. Res., 41, W12404, doi:10.1029/2004WR003726.

    """
    _name = "NoiseModel"

    def __init__(self, norm=True):
        NoiseModelBase.__init__(self)
        self.norm = norm
        self.nparam = 1
        self.set_init_parameters()

    @staticmethod
    def simulate(res, parameters):
        """
        Simulate noise from the residuals.

        Parameters
        ----------
        res: pandas.Series
            The residual series.
        parameters: array-like
            Alpha parameters used by the noisemodel.

        Returns
        -------
        noise: pandas.Series
            Series of the noise.

        """
        alpha = parameters[0]
        odelt = (res.index[1:] - res.index[:-1]).values / Timedelta("1d")
        v = Series(index=res.index, dtype="float64", name="Noise")
        v.iloc[0] = res.values[0]
        v.iloc[1:] = res.values[1:] - np.exp(-odelt / alpha) * res.values[:-1]
        return v

    def weights(self, res, parameters):
        """
        Method to calculate the weights for the noise.

        Parameters
        ----------
        res: pandas.Series
            Pandas Series with the residuals to compute the weights for. The
            Series index must be a DatetimeIndex.
        parameters: numpy.ndarray
            numpy array with the parameters used in the noise mdoel.

        Returns
        -------
        w: pandas.Series
            Series of the weights.

        Notes
        -----
        Weights are

        .. math::

        w = 1 / sqrt((1 - exp(-2 \\Delta t / \\alpha)))

        which are then normalized so that sum(w) = len(res)

        """
        alpha = parameters[0]
        odelt = np.empty(res.size)
        odelt[0] = 1e12  # large for first measurement
        odelt[1:] = (res.index[1:] - res.index[:-1]).values / Timedelta("1d")
        exp = np.exp(-2.0 / alpha * odelt)  # Twice as fast as 2*odelt/alpha
        # weights of noise, not noise^2
        w = Series(data=1 / np.sqrt(1.0 - exp), index=res.index,
                   dtype="float64", name="noise_weights")
        if self.norm:
            w = w.multiply(np.exp(1.0 / (2.0 * odelt.size) *
                                  np.sum(np.log(1.0 - exp))))
        return w


class NoiseModelOld(NoiseModelBase):
    """
    Noise model with exponential decay of the residual and weighting.

    Notes
    -----
    Calculates the noise [1]_ according to:

    .. math::

        v(t1) = r(t1) - r(t0) * exp(- (\\frac{\\Delta t}{\\alpha})

    Note that in the referenced paper, alpha is defined as the inverse of
    alpha used in Pastas. The unit of the alpha parameter is always in days.

    References
    ----------
    .. [1] von Asmuth, J. R., and M. F. P. Bierkens (2005), Modeling
           irregularly spaced residual series as a continuous stochastic
           process, Water Resour. Res., 41, W12404, doi:10.1029/2004WR003726.

    """
    _name = "NoiseModel"

    def __init__(self):
        NoiseModelBase.__init__(self)
        self.nparam = 1
        self.set_init_parameters()

    def simulate(self, res, parameters):
        """
        Simulate noise from the residuals.

        Parameters
        ----------
        res: pandas.Series
            The residual series.
        parameters: array-like
            Alpha parameters used by the noisemodel.

        Returns
        -------
        noise: pandas.Series
            Series of the noise.

        """
        alpha = parameters[0]
        odelt = (res.index[1:] - res.index[:-1]).values / Timedelta("1d")
        # res.values is needed else it gets messed up with the dates
        res.iloc[1:] -= np.exp(-odelt / alpha) * res.values[:-1]
        res.name = "Noise"
        return res

    @staticmethod
    def weights(res, parameters):
        """
        Method to calculate the weights for the noise.

        Based on the sum of weighted squared noise (SWSI) method.

        Parameters
        ----------
        res: pandas.Series
            The residual series.
        parameters: array-like
            Alpha parameters used by the noisemodel.

        Returns
        -------
        w: numpy.ndarray
            Array with the weights.

        """
        alpha = parameters[0]
        odelt = (res.index[1:] - res.index[:-1]).values / Timedelta("1d")
        # divide power by 2 as nu / sigma is returned
        power = 1.0 / (2.0 * odelt.size)
        exp = np.exp(-2.0 / alpha * odelt)  # Twice as fast as 2*odelt/alpha
        w = np.exp(power * np.sum(np.log(1.0 - exp))) / np.sqrt(1.0 - exp)
        w = np.insert(w, 0, 0)  # Set first weight to zero
        w = Series(w, res.index)
        return w


class ArmaModel(NoiseModelBase):
    """
    ARMA(1,1) Noise model to simulate the noise.

    Notes
    -----
    Calculates the noise according to:

    .. math::

        \\upsilon_t = r_t - r_{t-1} e^{-\\Delta t/\\alpha} - \\upsilon_{t-1}
        e^{-\\Delta t/\\beta}

    The unit of the alpha parameter is always in days.

    Warnings
    --------
    This model has only been tested on regular time steps and should not be
    used for irregular time steps yet.

    """
    _name = "ArmaModel"

    def __init__(self):
        NoiseModelBase.__init__(self)
        self.nparam = 2
        self.set_init_parameters()

    def set_init_parameters(self, oseries=None):
        self.parameters.loc["noise_alpha"] = (10, 1e-9, np.inf, True, "noise")
        self.parameters.loc["noise_beta"] = (10, 1e-9, np.inf, True, "noise")

    def simulate(self, res, parameters):
        alpha = parameters[0]
        beta = parameters[1]

        # Calculate the time steps
        odelt = (res.index[1:] - res.index[:-1]).values / Timedelta("1d")
        a = self.calculate_noise(res.values, odelt, alpha, beta)
        return Series(index=res.index, data=a, name="Noise")

    @staticmethod
    @njit
    def calculate_noise(res, odelt, alpha, beta):
        # Create an array to store the noise
        a = np.zeros_like(res)
        a[0] = res[0]
        # We have to loop through each value
        for i in range(1, res.size):
            a[i] = res[i] - res[i - 1] * np.exp(-odelt[i - 1] / alpha) - \
                   a[i - 1] * np.exp(-odelt[i - 1] / beta)
        return a
