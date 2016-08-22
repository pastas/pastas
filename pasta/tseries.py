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
from rfunc import One


class TseriesBase:
    """Tseries Base class called by each Tseries object.

    """

    def __init__(self, rfunc, name, xy, metadata, tmin, tmax, cutoff):
        self.rfunc = rfunc(cutoff)
        self.nparam = self.rfunc.nparam
        self.name = name
        self.xy = xy
        self.metadata = metadata
        self.tmin = tmin
        self.tmax = tmax

    def set_initial(self, name, value):
        """Method to set the initial parameter value

        Usage
        -----
        >>> ts.set_initial('parametername', 200)

        """
        if name in self.parameters.index:
            self.parameters.loc[name, 'initial'] = value

    def set_min(name, value):
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmin'] = value
        
    def set_max(name, value):
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmax'] = value

    def fix_parameter(self, name):
        if name in self.parameters.index:
            self.parameters.loc[name, 'vary'] = 0


class Tseries(TseriesBase):
    """
    Time series model consisting of the convolution of one stress with one
    response function.

    Parameters
    ----------
    stress: pd.Series
        pandas Series object containing the stress.
    rfunc: rfunc class
        Response function used in the convolution with the stess.
    name: str
        Name of the stress
    metadata: Optional[dict]
        dictionary containing metadata about the stress.
    xy: Optional[tuple]
        XY location in lon-lat format used for making maps.
    freq: Optional[str]
        Frequency to which the stress series are transformed. By default,
        the frequency is inferred from the data and that frequency is used.
        The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
        -aliases
    fillnan: Optional[str or float]
        Methods or float number to fill nan-values. Default values is
        'mean'. Currently supported options are: 'interpolate', float,
        and 'mean'. Interpolation is performed with a standard linear
        interpolation.

    """

    def __init__(self, stress, rfunc, name, metadata=None, xy=(0, 0), freq=None,
                 fillnan='mean', cutoff=0.99):
        self.stress = check_tseries(stress, freq, fillnan, name=name)
        TseriesBase.__init__(self, rfunc, name, xy, metadata,
                             self.stress.index.min(), self.stress.index.max(),
                             cutoff)
        self.set_init_parameters()

    def set_init_parameters(self):
        """
        Set the initial parameters back to their default values.

        """
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, p, tindex=None):
        """ Simulates the head contribution.

        Parameters
        ----------
        p: 1D array
           Parameters used for simulation.
        tindex: Optional[Pandas time series]
           Time indices to simulate the model.

        Returns
        -------
        Pandas Series Object
            The simulated head contribution.

        """
        b = self.rfunc.block(p)
        self.npoints = len(self.stress)  # Why recompute?
        h = pd.Series(fftconvolve(self.stress, b, 'full')[:self.npoints],
                      index=self.stress.index)
        if tindex is not None:
            h = h[tindex]
        return h


class Recharge(TseriesBase):
    """Time series model performing convolution on groundwater recharge
    calculated from precipitation and evaporation with one response function.

    Parameters
    ----------
    precip: pd.Series
        pandas Series object containing the precipitation stress.
    evap: pd.Series
        pandas Series object containing the evaporationstress.
    rfunc: rfunc class
        Response function used in the convolution with the stess.
    recharge: recharge_func class object
    name: str
        Name of the stress
    metadata: Optional[dict]
        dictionary containing metadata about the stress.
    xy: Optional[tuple]
        XY location in lon-lat format used for making maps.
    freq: Optional[list of str]
        Frequency to which the stress series are transformed. By default,
        the frequency is inferred from the data and that frequency is used.
        The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
        -aliases
    fillnan: Optional[list of str or float]
        Methods or float number to fill nan-values. Default value for
        precipitation is 'mean' and default for evaporation is 'interpolate'.
        Currently supported options are: 'interpolate', float,
        and 'mean'. Interpolation is performed with a standard linear
        interpolation.

    Notes
    -----
    Please check the documentation of the recharge_func classes for more details.

    References
    ----------
    [1] R.A. Collenteur [2016] Non-linear time series analysis of deep groundwater
    levels: Application to the Veluwe. MSc. thesis, TU Delft.
    http://repository.tudelft.nl/view/ir/uuid:baf4fc8c-6311-407c-b01f-c80a96ecd584/

    """

    def __init__(self, precip, evap, rfunc, recharge,
                 name='Recharge', metadata=None, xy=(0, 0), freq=[None, None],
                 fillnan=['mean', 'interpolate'], cutoff=0.99):

        # Check and name the time series
        P = check_tseries(precip, freq[0], fillnan[0], name=name + '_P')
        E = check_tseries(evap, freq[1], fillnan[1], name=name + '_E')

        # Select data where both series are available
        self.precip = P[P.index & E.index]
        self.evap = E[P.index & E.index]
        self.evap.name = 'Evaporation'
        self.precip.name = 'Precipitation'

        # Store tmin and tmax
        TseriesBase.__init__(self, rfunc, name, xy, metadata,
                             self.precip.index.min(), self.precip.index.max(),
                             cutoff)
        self.set_init_parameters()

        # The recharge calculation needs arrays
        self.precip_array = np.array(self.precip)
        self.evap_array = np.array(self.evap)

        self.recharge = recharge()
        self.set_init_parameters()
        self.nparam = self.rfunc.nparam + self.recharge.nparam
        self.stress = self.simulate_recharge()

    def set_init_parameters(self):
        self.parameters = pd.concat([self.rfunc.set_parameters(self.name),
                                     self.recharge.set_parameters(self.name)])

    def simulate(self, p, tindex=None):
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
        """
        Returns the simulated recharge with the parameters saves in the recharge
        object.

        Parameters
        ----------
        p: optional[pd.DataFrame]
            Array containing the parameters of the recharge model.
        Returns
        -------
            Other simulated recharge series as a pd. Series object.

        """
        rseries = self.recharge.simulate(self.precip_array, self.evap_array,
                                         p[-self.recharge.nparam:])
        rseries = pd.Series(rseries, index=self.precip.index,
                            name='Recharge')
        return rseries


class Well(TseriesBase):
    """Time series model consisting of the convolution of one or more stresses with
    one response function.

    Parameters
    ----------
    stress: list
        list of pandas Series objects containing the stresses.
    rfunc: rfunc class
        Response function used in the convolution with the stess.
    name: str
        Name of the stress
    metadata: Optional[dict]
        dictionary containing metadata about the stress.
    xy: Optional[tuple]
        XY location in lon-lat format used for making maps.
    freq: Optional[str]
        Frequency to which the stress series are transformed. By default,
        the frequency is inferred from the data and that frequency is used.
        The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
        -aliases
    fillnan: Optional[str or float]
        Methods or float number to fill nan-values. Default values is
        'mean'. Currently supported options are: 'interpolate', float,
        and 'mean'. Interpolation is performed with a standard linear
        interpolation.

    """

    # TODO implement this function
    def __init__(self, stress, rfunc, r, name, metadata=None,
                 xy=(0, 0), freq=None, fillna='mean', cutoff=0.99):

        # Check stresses
        self.stress = []
        if type(stress) is pd.Series:
            stress = [stress]
        for i in range(len(stress)):
            self.stress.append(check_tseries(stress, freq, fillna, name))

        self.set_init_parameters()
        self.r = r

        TseriesBase.__init__(self, rfunc, name, xy, metadata,
                             self.stress.index.min(), self.stress.index.max(),
                             cutoff)
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, p=None, tindex=None):
        h = pd.Series(data=0, index=self.stress[0].index)
        for i in range(len(self.stress)):
            self.npoints = len(self.stress[i])
            b = self.rfunc.block(p, self.r[i])  # nparam-1 depending on rfunc
            h += fftconvolve(self.stress[i], b, 'full')[:self.npoints]
        if tindex is not None:
            h = h[tindex]
        return h


class Constant(TseriesBase):
    """A constant value that is added to the time series model.

    Parameters
    ----------
    value : Optional[float]
        Initial estimate of the parameter value. E.g. The minimum of the observed
        series.

    """

    def __init__(self, name='Constant', xy=None, metadata=None, value=0.0, pmin=-5,
                 pmax=+5):
        self.nparam = 1
        self.value = value
        self.pmin = self.value + pmin
        self.pmax = self.value + pmax
        self.set_init_parameters()
        TseriesBase.__init__(self, One, name, xy, metadata,
                             pd.Timestamp.min, pd.Timestamp.max, 0)
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = pd.DataFrame(columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.parameters.loc['constant_d'] = (self.value, self.pmin, self.pmax, 1, 'constant')

    def simulate(self, p=None, t=None):
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
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = pd.DataFrame(columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.parameters.loc['noise_alpha'] = (14.0, 0, 5000, 1, 'noise')

    def set_parameters(self, **kwargs):
        for i in kwargs:
            self.parameters.loc['%s' % i, 'value'] = kwargs[i]

    def fix_parameters(self, **kwargs):
        for i in kwargs:
            if (kwargs[i] is not 0) and (kwargs[i] is not 1):
                print 'vary should be 1 or 0, not %s' % kwargs[i]
            self.parameters.loc['%s' % i, 'vary'] = kwargs[i]

    def simulate(self, res, delt, p, tindex=None):
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
