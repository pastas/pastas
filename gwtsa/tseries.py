import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

"""The tseries.py file contains all time series objects.

Each class defined in tseries should at least have the following:

Attributes
----------
stress : Pandas Series
    Series object of the stress with index in datetime format.
rfunc : Object
    Response function defined in rfunc.py, E.g. Gamma().
npoints : int
    length of the stress Series.
nparam : int
    Number of parameters for this tseries object. Depending on
    rfunc.nparam plus the parameters in used in this class.
name : str
    Name of this tseries object. Used as prefix for the parameters.
parameters : Pandas Dataframe
    Dataframe containing the parameters.

Methods
-------
simulate

"""


class Tseries:
    """Time series model consisting of the convolution of one stress with one
    response function.

    """
    def __init__(self, stress, rfunc, name):
        self.stress = stress
        self.rfunc = rfunc
        self.npoints = len(self.stress)
        self.nparam = rfunc.nparam
        self.name = name
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, tindex=None, p=None):
        """ Simulates the head contribution.

        Parameters
        ----------
        tindex:
            Time indices to simulate the model.
        p: Optional[array-like]
            Parameters used for simulation. If p is not
            provided, parameters attribute will be used.

        Returns
        -------
        Pandas Series Object
            The simulated head contribution.

        """
        if p is None: p = np.array(self.parameters.value)
        b = self.rfunc.block(p)
        h = pd.Series(fftconvolve(self.stress, b, 'full')[:self.npoints],
                      index=self.stress.index)
        if tindex is not None:
            h = h[tindex]
        return h


class Tseries2:
    """Time series model consisting of the convolution of two stresses with
    one response function.

    stress = stress1 + factor * stress2

    Last parameters is factor

    """
    def __init__(self, stress1, stress2, rfunc, name):
        self.stress1 = stress1
        self.stress2 = stress2
        self.rfunc = rfunc
        self.nparam = self.rfunc.nparam + 1
        self.name = name
        self.parameters = self.rfunc.set_parameters(self.name)
        self.parameters.loc[self.name + '_f'] = (-1.0, -5.0, 0.0, 1)

    def simulate(self, tindex=None, p=None):
        if p is None: p = np.array(self.parameters.value)
        b = self.rfunc.block(p[:-1])  # nparam-1 depending on rfunc
        stress = self.stress1 + p[-1] * self.stress2
        stress.fillna(stress.mean(), inplace=True)
        self.npoints = len(stress)
        h = pd.Series(fftconvolve(stress, b, 'full')[:self.npoints],
                      index=stress.index)
        if tindex is not None:
            h = h[tindex]
        return h


class Constant:
    """A constant value.

    Parameters
    ----------
    value : Optional[float]
        Initial estimate of the parameter value. E.g. The minimum of the observed
        series.

    """
    def __init__(self, value=0.0):
        self.nparam = 1
        self.parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
        self.parameters.loc['constant_d'] = (value, np.nan, np.nan, 1)

    def simulate(self, tindex=None, p=None):
        if p is None: p = np.array(self.parameters.value)
        return p


class NoiseModel:
    """Noise model with exponential decay of the residual.

    Notes
    -----
    Calculates the innovations [1] according to:
    v(t1) = r(t1) - r(t0) * exp(- (t1 - t0) / alpha)

    References
    ----------
    .. [1] von Asmuth, J. R., and M. F. P. Bierkens (2005), Modeling irregularly
    spaced residual series as a continuous stochastic process, Water Resour.
    Res., 41, W12404, doi:10.1029/2004WR003726.

    """
    def __init__(self):
        self.nparam = 1
        self.parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
        self.parameters.loc['noise_alpha'] = (14.0, 0, 5000, 1)

    def simulate(self, res, delt, tindex=None, p=None):
        """

        Parameters
        ----------
        res : Pandas Series
            The residual series.
        delt : Pandas Series
            Time steps between observations.
        tindex : Optional[None]
            Time indices used for simulation.
        p : Optional[array-like]
            Alpha parameters used by the noisemodel.

        Returns
        -------
        Pandas Series
            Series of the innovations.
        """
        if p is None: p = np.array(self.parameters.value)
        innovations = pd.Series(res, index=res.index)
        innovations[1:] -= np.exp(-delt[1:] / p) * res.values[:-1]  # values is
        # needed else it gets messed up with the dates
        return innovations
