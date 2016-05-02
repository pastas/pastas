import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

"""
tseries module.
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
           Input: tindex: Optional pandas TimeIndex. Time index to simulate values
           p: Optional[array-like]. Parameters used for simulation. If p is not
           provided, parameters attribute will be used.
           Returns: pandas Series of simulated values
"""


class TseriesBase:
    """Tseries Base class called by each Tseries object.
    """

    def __init__(self, stress, rfunc, name, stressnames, xy, metadata, freq,
                 fillna):
        self.stress = self.check_stresses(stress, freq, fillna)
        self.rfunc = rfunc
        self.nparam = rfunc.nparam
        self.name = name
        if stressnames is None:
            self.stress_names = [k.name for k in self.stress]
        else:
            self.stress_names = stressnames
        self.xy = xy
        self.metadata = metadata

    def check_stresses(self, stress, freq, fillna):
        """ Check the stress series on missing values and constant frequency.

        Returns
        -------
        list of stresses:
            - Checked for Missing values
            - Checked for frequency of stress
        """
        if type(stress) is pd.Series:
            stress = [stress]
        stresses = []
        for k in stress:
            assert isinstance(k, pd.Series), 'Expected a Pandas Series, ' \
                                             'got %s' % type(k)
            # Deal with frequency of the stress series
            if freq:
                k = k.asfreq(freq)
            else:
                freq = pd.infer_freq(k.index)
                k = k.asfreq(freq)

            # Deal with nan-values in stress series
            if k.hasnans:
                print '%i nan-value(s) was/were found and filled with: %s' % (
                    k.isnull(
                    ).values.sum(), fillna)
                if fillna == 'interpolate':
                    k.interpolate('time')
                elif type(fillna) == float:
                    print fillna, 'init'
                    k.fillna(fillna, inplace=True)
                else:
                    k.fillna(k.mean(), inplace=True)  # Default option
            stresses.append(k)
        return stresses

    def set_parameters(self, **kwargs):
        """Method to set the parameters value

        Usage
        -----
        E.g. ts2.set_parameters(recharge_A=200)

        """
        for i in kwargs:
            self.parameters.loc['%s' % i, 'value'] = kwargs[i]


class Tseries(TseriesBase):
    """Time series model consisting of the convolution of one stress with one
    response function.

    """

    def __init__(self, stress, rfunc, name, metadata=None, stressnames=None,
                 xy=(0, 0), freq=None, fillna='mean'):
        TseriesBase.__init__(self, stress, rfunc, name, metadata, stressnames, xy,
                             freq, fillna)
        self.set_init_parameters()
        self.stress = self.stress[0]  # unpack stress list for this Tseries

    def set_init_parameters(self):
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


class Tseries2(TseriesBase):
    """Time series model consisting of the convolution of two stresses with
    one response function.

    stress = stress1 + factor * stress2

    Last parameters is factor

    """

    def __init__(self, stress, rfunc, name, metadata=None, stressnames=None,
                 xy=(0, 0), freq=None, fillna='mean'):
        TseriesBase.__init__(self, stress, rfunc, name, metadata, stressnames,
                             xy, freq, fillna)
        self.nparam += 1
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = self.rfunc.set_parameters(self.name)
        self.parameters.loc[self.name + '_f'] = (-1.0, -5.0, 0.0, 1)

    def simulate(self, tindex=None, p=None):
        if p is None:
            p = np.array(self.parameters.value)
        b = self.rfunc.block(p[:-1])  # nparam-1 depending on rfunc
        stress = self.stress[0] + p[-1] * self.stress[1]
        stress.fillna(stress.mean(), inplace=True)
        self.npoints = len(stress)
        h = pd.Series(fftconvolve(stress, b, 'full')[:self.npoints],
                      index=stress.index)
        if tindex is not None:
            h = h[tindex]
        return h


class Tseries3(TseriesBase):
    def __init__(self, stress, rfunc, recharge, name, metadata=None,
                 stressnames=None,
                 xy=(0, 0), freq=None, fillna='mean'):
        TseriesBase.__init__(self, stress, rfunc, name, stressnames, xy,
                             metadata, freq, fillna)
        self.recharge = recharge
        self.set_init_parameters()
        self.nparam = self.rfunc.nparam + self.recharge.nparam

    def set_init_parameters(self):
        self.parameters = pd.concat([self.rfunc.set_parameters(self.name),
                                     self.recharge.set_parameters(self.name)])

    def simulate(self, tindex=None, p=None):
        if p is None:
            p = np.array(self.parameters.value)
        b = self.rfunc.block(p[:-3])
        self.npoints = len(self.stress[0])
        P = np.array(self.stress[0])
        E = np.array(self.stress[1])
        self.rseries = self.recharge.simulate(P, E, p[-self.recharge.nparam:])
        self.npoints = len(self.rseries)
        h = pd.Series(fftconvolve(self.rseries, b, 'full')[:self.npoints],
                      index=self.stress[0].index)
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
