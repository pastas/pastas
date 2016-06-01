"""
tseries module
    Constains class for time series objects.

    Each response function class needs the following:

    Attributes
    ----------
    nparam : int
        Number of parameters.
    name : str
        Name of this tseries object. Used as prefix for the parameters.
    parameters : pandas Dataframe
        Dataframe containing the parameters.

    Methods
    -------
    simulate : Returns pandas Series Object with simulate values
               Input: tindex: Optional pandas TimeIndex. Time index to simulate
               values
               p: Optional[array-like]. Parameters used for simulation. If p is not
               provided, parameters attribute will be used.
               Returns: pandas Series of simulated values
"""

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from checks import check_tseries


class TseriesBase:
    """Tseries Base class called by each Tseries object.

    """

    def __init__(self, rfunc, name, xy, metadata):
        self.rfunc = rfunc
        self.nparam = rfunc.nparam
        self.name = name
        self.xy = xy
        self.metadata = metadata

    def set_parameters(self, **kwargs):
        """Method to set the parameters value

        Usage
        -----
        >>> ts2.set_parameters(recharge_A=200)

        """
        for i in kwargs:
            self.parameters.loc['%s' % i, 'value'] = kwargs[i]


class Tseries(TseriesBase):
    """
    Time series model consisting of the convolution of one stress with one
    response function.

    Parameters
    ----------
    stress
    rfunc
    name
    metadata
    xy
    freq
    fillnan

    """

    def __init__(self, stress, rfunc, name, metadata=None, xy=(0, 0), freq=None,
                 fillnan='mean'):
        TseriesBase.__init__(self, rfunc, name, xy, metadata)
        self.stress = check_tseries(stress, freq, fillnan)
        self.set_init_parameters()

    def set_init_parameters(self):
        """
        Set the initial parameters back to their default values.

        """
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
        if p is None:
            p = np.array(self.parameters.value)
        b = self.rfunc.block(p)
        self.npoints = len(self.stress)
        h = pd.Series(fftconvolve(self.stress, b, 'full')[:self.npoints],
                      index=self.stress.index)
        if tindex is not None:
            h = h[tindex]
        return h


class Recharge(TseriesBase):
    """Time series model to calculate the groundwater recharge as a stress.

    Parameters
    ----------
    precip
    evap
    rfunc
    recharge
    name
    metadata
    xy
    freq
    fillnan
    """

    def __init__(self, precip, evap, rfunc, recharge,
                 name='Recharge', metadata=None, xy=(0, 0), freq=['D', 'D'],
                 fillnan=['mean', 'interpolate']):
        TseriesBase.__init__(self, rfunc, name, xy, metadata)

        # Check and name the time series
        P = check_tseries(precip, freq[0], fillnan[0])
        E = check_tseries(evap, freq[1], fillnan[1])

        # Select data where both series are available
        self.precip = P[P.index & E.index]
        self.evap = E[P.index & E.index]
        self.evap.name = 'Evaporation'
        self.precip.name = 'Precipitation'

        # The recharge calculation needs arrays
        self.precip_array = np.array(self.precip)
        self.evap_array = np.array(self.evap)

        self.recharge = recharge
        self.set_init_parameters()
        self.nparam = self.rfunc.nparam + self.recharge.nparam

    def set_init_parameters(self):
        self.parameters = pd.concat([self.rfunc.set_parameters(self.name),
                                     self.recharge.set_parameters(self.name)])

    def simulate(self, tindex=None, p=None):
        if p is None:
            p = np.array(self.parameters.value)
        b = self.rfunc.block(p[:-self.recharge.nparam])  # Block response
        rseries = self.recharge.simulate(self.precip_array, self.evap_array,
                                         p[-self.recharge.nparam:])
        self.npoints = len(rseries)
        h = pd.Series(fftconvolve(rseries, b, 'full')[:self.npoints],
                      index=self.precip.index, name='Recharge')
        if tindex is not None:
            h = h[tindex]
        return h

    def simulate_recharge(self, p=None):
        if p is None:
            p = np.array(self.parameters.value)
        rseries = self.recharge.simulate(self.precip_array, self.evap_array,
                                         p[-self.recharge.nparam:])
        rseries = pd.Series(rseries, index=self.precip.index,
                            name='Recharge')
        return rseries


class Well(TseriesBase):
    """Time series model consisting of the convolution of two stresses with
    one response function.


    """

    # TODO implement this function
    def __init__(self, stress, rfunc, r, name, metadata=None, stressnames=None,
                 xy=(0, 0), freq=None, fillna='mean'):
        TseriesBase.__init__(self, rfunc, name, xy, metadata)

        # Check stresses
        self.stress = []
        if type(stress) is pd.Series:
            stress = [stress]
        for i in range(len(stress)):
            self.stress.append(check_tseries(stress, freq, fillna))

        self.set_init_parameters()
        self.r = r

    def set_init_parameters(self):
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, tindex=None, p=None):
        if p is None:
            p = np.array(self.parameters.value)
        h = pd.Series(data=0, index=self.stress[0].index)
        for i in range(len(self.stress)):
            self.npoints = len(self.stress[i])
            b = self.rfunc.block(p, self.r[i])  # nparam-1 depending on rfunc
            h += fftconvolve(self.stress[i], b, 'full')[:self.npoints]
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
        if p is None:
            p = np.array(self.parameters.value)
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
        if p is None:
            p = np.array(self.parameters.value)
        innovations = pd.Series(res, index=res.index)
        # weights of innovations, see Eq. A17 in reference [1]
        power = (1.0 / (2.0 * (len(delt) - 1)))
        w = np.exp(power * np.sum(np.log(1 - np.exp(-2 * delt[1:] / p)))) / \
            np.sqrt(1.0 - np.exp(-2 * delt[1:] / p))
        # res.values is needed else it gets messed up with the dates
        innovations[1:] -= w * np.exp(-delt[1:] / p) * res.values[:-1]
        if tindex is not None:
            innovations = innovations[tindex]
        return innovations
