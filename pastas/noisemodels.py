"""The noisemodels module contains all noisemodels available in Pastas.


Author: R.A. Collenteur, 2017

"""

from abc import ABC
from logging import getLogger

import numpy as np
import pandas as pd

from .decorators import set_parameter
from .utils import get_dt

logger = getLogger(__name__)

__all__ = ["NoiseModel", "NoiseModel2"]


class NoiseModelBase(ABC):
    _name = "NoiseModelBase"

    def __init__(self):
        self.nparam = 0
        self.name = "noise"
        self.parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])

    @set_parameter
    def set_initial(self, name, value):
        """Internal method to set the initial parameter value

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        if name in self.parameters.index:
            self.parameters.loc[name, 'initial'] = value
        else:
            print('Warning:', name, 'does not exist')

    @set_parameter
    def set_min(self, name, value):
        """Internal method to set the minimum value of the noisemodel.

        Notes
        -----
        The preferred method for parameter setting is through the model.


        """
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmin'] = value
        else:
            print('Warning:', name, 'does not exist')

    @set_parameter
    def set_max(self, name, value):
        """Internal method to set the maximum parameter values.

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmax'] = value
        else:
            print('Warning:', name, 'does not exist')

    @set_parameter
    def set_vary(self, name, value):
        """Internal method to set if the parameter is varied during
        optimization.

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        self.parameters.loc[name, 'vary'] = value

    def dump(self):
        data = dict()
        data["type"] = self._name
        return data


class NoiseModel(NoiseModelBase):
    _name = "NoiseModel"
    __doc__ = """Noise model with exponential decay of the residual .

    Notes
    -----
    Calculates the noise [1] according to:

    .. math::
        v(t1) = r(t1) - r(t0) * exp(- (t1 - t0) / alpha)
    
    Note that in the referenced paper, alpha is defined as the inverse of 
    alpha used in Pastas.
    
    Examples
    --------
    It can happen that the noisemodel is used during model calibration
    to explain most of the variation in the data. A recommended solution is to
    scale the initial parameter with the model timestep, E.g.::

    >>> n = NoiseModel()
    >>> n.set_initial("noise_alpha", 1.0 * ml.get_dt(ml.freq))

    References
    ----------
    .. [1] von Asmuth, J. R., and M. F. P. Bierkens (2005), Modeling irregularly spaced residual series as a continuous stochastic process, Water Resour. Res., 41, W12404, doi:10.1029/2004WR003726.

    """

    def __init__(self):
        NoiseModelBase.__init__(self)
        self.nparam = 1
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters.loc['noise_alpha'] = (14.0, 0, 5000, 1, 'noise')

    def simulate(self, res, parameters, freq):
        """

        Parameters
        ----------
        res : pandas.Series
            The residual series.
        parameters : array-like, optional
            Alpha parameters used by the noisemodel.

        Returns
        -------
        noise: pandas.Series
            Series of the noise.

        """
        dt = get_dt(freq)
        odelt = res.index.to_series().diff() / pd.Timedelta(dt, "D")
        odelt = odelt.iloc[1:]
        alpha = parameters[0]
        noise = pd.Series(data=res)
        # res.values is needed else it gets messed up with the dates
        noise.iloc[1:] -= np.exp(-odelt / alpha) * res.values[:-1]

        weights = self.weights(alpha, odelt)
        noise = noise.multiply(weights, fill_value=0.0)
        noise.name = "Noise"
        return noise

    def weights(self, alpha, odelt):
        """Method to calculate the weights for the noise based on the
        sum of weighted squared noise (SWSI) method.

        Parameters
        ----------
        alpha
        odelt:

        Returns
        -------

        """
        # divide power by 2 as nu / sigma is returned
        power = 1.0 / (2.0 * odelt.size)
        exp = np.exp(-2.0 / alpha * odelt)  # Twice as fast as 2*odelt/alpha
        w = np.exp(power * np.sum(np.log(1.0 - exp))) / np.sqrt(1.0 - exp)
        return w


class NoiseModel2(NoiseModelBase):
    """
    Noise model with exponential decay of the residual.

    Notes
    -----
    Calculates the noise [1] according to:

    .. math::
        v(t1) = r(t1) - r(t0) * exp(- (t1 - t0) / alpha)

    Examples
    --------
    It can happen that the noisemodel is used during model calibration
    to explain most of the variation in the data. A recommended solution is to
    scale the initial parameter with the model timestep, E.g.::

    >>> n = NoiseModel()
    >>> n.set_initial("noise_alpha", 1.0 * ml.get_dt(ml.freq))

    References
    ----------
    .. [A] von Asmuth, J. R., and M. F. P. Bierkens (2005), Modeling irregularly spaced residual series as a continuous stochastic process, Water Resour. Res., 41, W12404, doi:10.1029/2004WR003726.

    """
    _name = "NoiseModel2"

    def __init__(self):
        NoiseModelBase.__init__(self)
        self.nparam = 1
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters.loc['noise_alpha'] = (14.0, 0, 5000, 1, 'noise')

    def simulate(self, res, parameters, freq):
        """

        Parameters
        ----------
        res : pandas.Series
            The residual series.
        parameters : array_like, optional
            Alpha parameters used by the noisemodel.

        Returns
        -------
        noise: pandas.Series
            Series of the noise.

        """
        odelt = res.index.to_series().diff() / pd.Timedelta(1, freq)
        odelt = odelt.iloc[1:]
        noise = pd.Series(res)
        alpha = parameters[0]
        # res.values is needed else it gets messed up with the dates
        noise.iloc[1:] -= np.exp(-odelt / alpha) * res.values[:-1]
        noise.name = "Noise"
        return noise
