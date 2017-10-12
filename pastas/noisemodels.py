"""The noisemodels module contains all noisemodels available in Pastas.


Author: R.A. Collenteur, 2017

"""

import logging
from abc import ABC

import numpy as np
import pandas as pd

from .decorators import set_parameter

logger = logging.getLogger(__name__)

all = ["NoiseModel", "NoiseModel2"]


class NoiseModelBase(ABC):
    _name = "NoiseModelBase"

    def __init__(self):
        self.nparam = 0
        self.name = "noise"

    @set_parameter
    def set_initial(self, name, value):
        """Method to set the initial parameter value

        Examples
        --------

        >>> ts.set_initial('parameter_name', 200)

        """
        if name in self.parameters.index:
            self.parameters.loc[name, 'initial'] = value
        else:
            print('Warning:', name, 'does not exist')

    @set_parameter
    def set_min(self, name, value):
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmin'] = value
        else:
            print('Warning:', name, 'does not exist')

    @set_parameter
    def set_max(self, name, value):
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmax'] = value
        else:
            print('Warning:', name, 'does not exist')

    @set_parameter
    def set_vary(self, name, value):
        self.parameters.loc[name, 'pmax'] = value

    def dump(self):
        data = dict()
        data["type"] = self._name
        return data


class NoiseModel(NoiseModelBase):
    _name = "NoiseModel"
    __doc__ = """Noise model with exponential decay of the residual.

    Notes
    -----
    Calculates the innovations [1] according to:

    .. math::
        v(t1) = r(t1) - r(t0) * exp(- (t1 - t0) / alpha)

    Examples
    --------
    It can happen that the noisemodel is used in during the model calibration
    to explain most of the variation in the data. A recommended solution is to
    scale the initial parameter with the model timestep, E.g.::

    >>> n = NoiseModel()
    >>> n.set_initial("noise_alpha", 1.0 * ml.get_dt(ml.freq))

    References
    ----------
    von Asmuth, J. R., and M. F. P. Bierkens (2005), Modeling irregularly spaced residual series as a continuous stochastic process, Water Resour. Res., 41, W12404, doi:10.1029/2004WR003726.

    """

    def __init__(self):
        NoiseModelBase.__init__(self)
        self.nparam = 1
        self.set_init_parameters()

    def fix_parameter(self, name):
        if name in self.parameters.index:
            self.parameters.loc[name, 'vary'] = 0
        else:
            logger.warning('%s does not exist' % name)

    def set_init_parameters(self):
        self.parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.parameters.loc['noise_alpha'] = (14.0, 0, 5000, 1, 'noise')

    def simulate(self, res, delt, p, tindex=None):
        """

        Parameters
        ----------
        res : pandas.Series
            The residual series.
        delt : pandas.Series
            Time steps between observations.
        tindex : None, optional
            Time indices used for simulation.
        p : array-like, optional
            Alpha parameters used by the noisemodel.

        Returns
        -------
        innovations: pandas.Series
            Series of the innovations.

        """
        innovations = pd.Series(res, index=res.index, name="Innovations")
        # res.values is needed else it gets messed up with the dates
        innovations[1:] -= np.exp(-delt[1:] / p[0]) * res.values[:-1]

        weights = self.weights(p, delt)
        innovations = innovations.multiply(weights, fill_value=0.0)

        if tindex is not None:
            innovations = innovations[tindex]
        return innovations

    def weights(self, p, delt):
        """Method to calculate the weights for the innovations based on the
        sum of weighted squares innovations (SWSI) method.

        Parameters
        ----------
        p: nump.array
            array or iterable containing the parameters.
        delt:


        Returns
        -------

        """
        alpha = p[-1]
        power = (1.0 / (2.0 * (delt[1:].size - 1.0)))
        w = np.exp(
            power * np.sum(np.log(1.0 - np.exp(-2.0 * delt[1:] / alpha)))) / \
            np.sqrt(1.0 - np.exp(-2.0 * delt[1:] / alpha))
        return w


class NoiseModel2(NoiseModelBase):
    _name = "NoiseModel2"
    __doc__ = """Noise model with exponential decay of the residual.

    Notes
    -----
    Calculates the innovations [1] according to:

    .. math::
        v(t1) = r(t1) - r(t0) * exp(- (t1 - t0) / alpha)

    Examples
    --------
    It can happen that the noisemodel is used in during the model calibration
    to explain most of the variation in the data. A recommended solution is to
    scale the initial parameter with the model timestep, E.g.::

    >>> n = NoiseModel()
    >>> n.set_initial("noise_alpha", 1.0 * ml.get_dt(ml.freq))

    References
    ----------
    von Asmuth, J. R., and M. F. P. Bierkens (2005), Modeling irregularly spaced residual series as a continuous stochastic process, Water Resour. Res., 41, W12404, doi:10.1029/2004WR003726.

    """

    def __init__(self):
        NoiseModelBase.__init__(self)
        self.nparam = 1
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.parameters.loc['noise_alpha'] = (14.0, 0, 5000, 1, 'noise')

    def simulate(self, res, delt, p, tindex=None):
        """

        Parameters
        ----------
        res : pandas.Series
            The residual series.
        delt : pandas.Series
            Time steps between observations.
        tindex : None, optional
            Time indices used for simulation.
        p : array-like, optional
            Alpha parameters used by the noisemodel.

        Returns
        -------
        innovations: pandas.Series
            Series of the innovations.

        """
        innovations = pd.Series(res, index=res.index, name="Innovations")
        # res.values is needed else it gets messed up with the dates
        innovations[1:] -= np.exp(-delt[1:] / p[0]) * res.values[:-1]
        if tindex is not None:
            innovations = innovations[tindex]
        return innovations
