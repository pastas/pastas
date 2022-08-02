"""This module contains the noise models available in Pastas.

A Noise model may be used to transform the residual series into a noise
series that better represents white noise.

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
from pandas import DataFrame, Series, Timedelta

from .decorators import njit, set_parameter
from .utils import check_numba

__all__ = ["NoiseModel", "ArmaModel"]


class NoiseModelBase:
    _name = "NoiseModelBase"

    def __init__(self):
        self.nparam = 1
        self.name = "noise"
        self.parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])

    def set_init_parameters(self, oseries=None):
        if oseries is not None:
            pinit = np.diff(oseries.index.to_numpy()) / Timedelta("1D")
            pinit = np.median(pinit)
        else:
            pinit = 14.0
        self.parameters.loc["noise_alpha"] = (pinit, 1e-5, 5000.0, True,
                                              "noise")

    @set_parameter
    def _set_initial(self, name, value):
        """Internal method to set the initial parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "initial"] = value

    @set_parameter
    def _set_pmin(self, name, value):
        """Internal method to set the minimum value of the noisemodel.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "pmin"] = value

    @set_parameter
    def _set_pmax(self, name, value):
        """Internal method to set the maximum parameter values.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "pmax"] = value

    @set_parameter
    def _set_vary(self, name, value):
        """Internal method to set if the parameter is varied.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "vary"] = value

    def to_dict(self):
        return {"type": self._name}

    @staticmethod
    def weights(res, p):
        return 1


class NoiseModel(NoiseModelBase):
    """Noise model with exponential decay of the residuals and weighting.

    Parameters
    ----------
    norm: boolean, optional
        Boolean to indicate whether weights are normalized according to
        the Von Asmuth and Bierkens (2005) paper. Default is True.

    Notes
    -----
    Calculates the noise [1]_ according to:

    .. math::

        v(t_1) = r(t_1) - r(t_0) * \\exp(- \\Delta t / \\alpha)

    Calculates the weights as

    .. math::

        w = 1 / \\sqrt{(1 - \\exp(-2 \\Delta t / \\alpha))}

    The units of the alpha parameter is always in days. The first value of
    the noise is the residual ($v(t=0=r(t=0)$). First weight is
    1 / sig_residuals (i.e., delt = infty). Normalization of weights as in
    Von Asmuth and Bierkens (2005), optional.

    Differences compared to NoiseModelOld:

    1. First value is residual
    2. First weight is 1 / sig_residuals (i.e., delt = infty)
    3. Normalization of weights as in Von Asmuth and Bierkens (2005), optional

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
    def simulate(res, p):
        """Simulate noise from the residuals.

        Parameters
        ----------
        res: pandas.Series
            The residual series.
        p: array_like
            array_like object with the values as floats representing the
            model parameters. Here, Alpha parameter used by the noisemodel.

        Returns
        -------
        noise: pandas.Series
            Series of the noise.
        """
        alpha = p[0]
        odelt = np.diff(res.index.to_numpy()) / Timedelta("1D")
        v = np.append(res.values[0], res.values[1:] - np.exp(-odelt / alpha)
                      * res.values[:-1])
        return Series(data=v, index=res.index, name="Noise")

    def weights(self, res, p):
        """Method to calculate the weights for the noise.

        Parameters
        ----------
        res: pandas.Series
            Pandas Series with the residuals to compute the weights for. The
            Series index must be a DatetimeIndex.
        p: numpy.ndarray
            numpy array with the parameters used in the noise model.

        Returns
        -------
        w: pandas.Series
            Series of the weights.

        Notes
        -----
        Weights are

        .. math:: w = 1 / sqrt((1 - exp(-2 \\Delta t / \\alpha)))

        which are then normalized so that sum(w) = len(res)
        """
        alpha = p[0]
        # large for first measurement
        odelt = np.append(1e12, np.diff(res.index.to_numpy()) /
                          Timedelta("1D"))
        exp = np.exp(-2.0 / alpha * odelt)  # Twice as fast as 2*odelt/alpha
        w = 1 / np.sqrt(1.0 - exp)  # weights of noise, not noise^2
        if self.norm:
            w *= np.exp(1.0 / (2.0 * odelt.size) * np.sum(np.log(1.0 - exp)))
        return Series(data=w, index=res.index, name="noise_weights")


class ArmaModel(NoiseModelBase):
    """ARMA(1,1) Noise model to simulate the noise as defined in.

    [collenteur_2020]_.

    Notes
    -----
    Calculates the noise according to:

    .. math::
        \\upsilon_t = r_t - r_{t-1} e^{-\\Delta t/\\alpha} - \\upsilon_{t-1}
        e^{-\\Delta t/\\beta}

    The units of the alpha and beta parameters are always in days.

    Warnings
    --------
    This model has only been tested on regular time steps and should not be
    used for irregular time steps yet.

    References
    ----------
    .. [collenteur_2020] Collenteur, R., Bakker, M., Klammler, G., and Birk,
       S. (in review, 2020.) Estimating groundwater recharge from
       groundwater levels using non-linear transfer function noise models
       and comparison to lysimeter data, Hydrol. Earth Syst. Sci. Discuss.
       https://doi.org/10.5194/hess-2020-392
    """
    _name = "ArmaModel"

    def __init__(self):
        check_numba()
        NoiseModelBase.__init__(self)
        self.nparam = 2
        self.set_init_parameters()

    def set_init_parameters(self, oseries=None):
        if oseries is not None:
            pinit = np.diff(oseries.index.to_numpy()) / Timedelta("1D")
            pinit = np.median(pinit)
        else:
            pinit = 14.0
        self.parameters.loc["noise_alpha"] = (pinit, 1e-9, 5000.0, True,
                                              "noise")
        self.parameters.loc["noise_beta"] = (1., -np.inf, np.inf, True,
                                             "noise")

    def simulate(self, res, p):
        alpha = p[0]
        beta = p[1]

        # Calculate the time steps
        odelt = np.diff(res.index.to_numpy()) / Timedelta("1D")
        a = self.calculate_noise(res.values, odelt, alpha, beta)
        return Series(index=res.index, data=a, name="Noise")

    @staticmethod
    @njit
    def calculate_noise(res, odelt, alpha, beta):
        # Create an array to store the noise
        a = np.zeros_like(res)
        a[0] = res[0]

        if beta == 0.0:  # Prevent division by zero errors
            beta = 1e-24

        pm = beta / np.abs(beta)

        # We have to loop through each value
        for i in range(1, res.size):
            a[i] = res[i] - res[i - 1] * np.exp(-odelt[i - 1] / alpha) - \
                   a[i - 1] * pm * np.exp(-odelt[i - 1] / np.abs(beta))
        return a
