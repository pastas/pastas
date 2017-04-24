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

from __future__ import print_function, division

from warnings import warn

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from .checks import check_tseries
from .rfunc import One
from .utils import get_dt


class TseriesBase:
    """Tseries Base class called by each Tseries object.

    """

    def __init__(self, rfunc, name, xy, metadata, tmin, tmax, up, meanstress,
                 cutoff):
        self.rfunc = rfunc(up, meanstress, cutoff)
        self.nparam = self.rfunc.nparam
        self.name = name
        self.xy = xy
        self.metadata = metadata
        self.tmin = tmin
        self.tmax = tmax
        self.freq = None
        self.stress = pd.DataFrame()

    def set_initial(self, name, value):
        """Method to set the initial parameter value

        Usage
        -----
        >>> ts.set_initial('parametername', 200)

        """
        if name in self.parameters.index:
            self.parameters.loc[name, 'initial'] = value
        else:
            print('Warning:', name, 'does not exist')

    def set_min(self, name, value):
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmin'] = value
        else:
            print('Warning:', name, 'does not exist')

    def set_max(self, name, value):
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmax'] = value
        else:
            print('Warning:', name, 'does not exist')

    def fix_parameter(self, name):
        if name in self.parameters.index:
            self.parameters.loc[name, 'vary'] = 0
        else:
            print('Warning:', name, 'does not exist')

    def change_frequency(self,freq):
        # change the frequency
        # TODO: next lines of code ony are correct when the frequencies are a multiple of each other, fix this
        # this is for example not the case when monthly data is resampled to weekly data. it would be better to
        # work with weights. A week at the end of the month would then consist of the weighted data of that
        # month and the next. Right now, this week will get the value of the next month. Did not find a Pandas
        # method to perform this weighted mean.
        if get_dt(freq) > get_dt(self.freq):
            # downsample (for example from day to week), use mean
            # make sure the labels are still at the end of each period, and data at the right side of the bucket
            # is included (see http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.resample.html)
            self.stress = self.stress.resample(freq, label='right', closed='right').mean()
        elif get_dt(freq) < get_dt(self.freq):
            # upsample (for exmaple from week to day), use bfill
            self.stress = self.stress.resample(freq).bfill()
        self.freq = freq

    def get_stress(self, p=None, tindex=None):
        """
        Returns the stress or stresses of the time series object as a pandas
        DataFrame. If the time series object has multiple stresses each column
        represents a stress.

        Returns
        -------
        stress: pd.Dataframe()
            Pandas dataframe of the stress(es)

        """
        if tindex is not None:
            return self.stress[tindex]
        else:
            return self.stress


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

    def __init__(self, stress, rfunc, name, metadata=None, xy=(0, 0),
                 freq=None, fillnan='mean', up=True, cutoff=0.99):
        stress = check_tseries(stress, freq, fillnan, name=name)
        TseriesBase.__init__(self, rfunc, name, xy, metadata,
                             stress.index.min(), stress.index.max(),
                             up, stress.mean(), cutoff)
        self.freq = stress.index.freqstr
        self.stress[name] = stress
        self.set_init_parameters()

    def set_init_parameters(self):
        """
        Set the initial parameters (back) to their default values.

        """
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, p, tindex=None, dt=1):
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
        b = self.rfunc.block(p, dt)
        self.npoints = self.stress.index.size  # Why recompute?
        h = pd.Series(fftconvolve(self.stress[self.name], b, 'full')[
                      :self.npoints], index=self.stress.index, name=self.name)
        if tindex is not None:
            h = h[tindex]
        return h


class Tseries2(TseriesBase):
    """
    Time series model consisting of the convolution of two stresses with one
    response function. The first stress causes the head to go up and the second
    stress causes the head to go down.

    Parameters
    ----------
    stress1: pd.Series
        pandas Series object containing stress 1.
    stress2: pd.Series
        pandas Series object containing stress 2.
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

    def __init__(self, stress0, stress1, rfunc, name, metadata=None, xy=(0, 0),
                 freq=None, fillnan=('mean', 'interpolate'),
                 up=True, cutoff=0.99):
        # First check the series, then determine tmin and tmax
        stress0 = check_tseries(stress0, freq, fillnan[0], name=name)
        stress1 = check_tseries(stress1, freq, fillnan[1], name=name)

        # Select indices where both series are available
        index = stress0.index & stress1.index
        if index.size is 0:
            warn('The two stresses that were provided have no overlapping time'
                 ' indices. Please make sure time indices overlap or apply to '
                 'separate time series objects.')

        TseriesBase.__init__(self, rfunc, name, xy, metadata, index.min(),
                             index.max(), up, stress0.mean() - stress1.mean(),
                             cutoff)

        self.stress["stress0"] = stress0[index]
        self.stress["stress1"] = stress1[index]

        self.freq = stress0.index.freqstr
        self.set_init_parameters()

    def set_init_parameters(self):
        """
        Set the initial parameters back to their default values.

        """
        self.parameters = self.rfunc.set_parameters(self.name)
        self.parameters.loc[self.name + '_f'] = (-1.0, -2.0, 2.0, 1, self.name)
        self.nparam += 1

    def simulate(self, p, tindex=None, dt=1):
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
        b = self.rfunc.block(p[:-1], dt)
        self.npoints = self.stress.index.size  # Why recompute?
        h = pd.Series(
            fftconvolve(
                self.stress["stress0"] + p[-1] * self.stress["stress1"],
                b, 'full')[:self.npoints], index=self.stress.index,
            name=self.name)
        if tindex is not None:
            h = h[tindex]
        return h

    def get_stress(self, p=None, tindex=None):
        if p is not None:
            stress = self.stress[0] + p[-1] * self.stress[1]

            if tindex is not None:
                return stress[tindex]
            else:
                return stress
        else:
            print("parameter to calculate the stress is unknown")


class Recharge(TseriesBase):
    """Time series model performing convolution on groundwater recharge
    calculated from precipitation and evaporation with a single response function.

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
                 name, metadata=None, xy=(0, 0), freq=(None, None),
                 fillnan=('mean', 'interpolate'), cutoff=0.99):
        # Check and name the time series
        P = check_tseries(precip, freq[0], fillnan[0], name=name + '_P')
        E = check_tseries(evap, freq[1], fillnan[1], name=name + '_E')

        # Select indices where both series are available
        index = P.index & E.index

        if index.size is 0:
            raise Warning('The two stresses that were provided have no '
                          'overlapping time indices. Please make sure time '
                          'indices overlap or apply to separate time series '
                          'objects.')

        # Store tmin and tmax
        TseriesBase.__init__(self, rfunc, name, xy, metadata, index.min(),
                             index.max(), True,
                             precip.mean() - evap.mean(), cutoff)

        self.stress[P.name] = P[index]
        self.stress[E.name] = E[index]
        self.freq = self.stress.index.freqstr

        # The recharge calculation needs arrays
        self.precip_array = np.array(self.stress[P.name])
        self.evap_array = np.array(self.stress[E.name])

        self.recharge = recharge()
        self.set_init_parameters()
        self.nparam = self.rfunc.nparam + self.recharge.nparam

    def set_init_parameters(self):
        self.parameters = pd.concat([self.rfunc.set_parameters(self.name),
                                     self.recharge.set_parameters(self.name)])

    def simulate(self, p, tindex=None, dt=1):
        dt = int(dt)
        b = self.rfunc.block(p[:-self.recharge.nparam], dt)  # Block response
        rseries = self.recharge.simulate(self.precip_array, self.evap_array,
                                         p[-self.recharge.nparam:])
        self.npoints = len(rseries)
        h = pd.Series(fftconvolve(rseries, b, 'full')[:self.npoints],
                      index=self.stress.index, name=self.name)
        if tindex is not None:
            h = h[tindex]
        return h

    def get_stress(self, p=None, tindex=None):
        """
        Returns the stress or stresses of the time series object as a pandas
        DataFrame. If the time series object has multiple stresses each column
        represents a stress.

        Parameters
        ----------
        tindex: pd TimeIndex

        Returns
        -------
        stress: pd DataFrame
            DataFrame containing the stresses with the required time indices.

        """

        # If parameters are not provided, don't calculate the recharge.
        if p is not None:
            rseries = self.recharge.simulate(self.precip_array,
                                             self.evap_array,
                                             p[-self.recharge.nparam:])

            stress = pd.Series(rseries, index=self.stress.index,
                               name=self.name)

            if tindex is not None:
                return stress[tindex]
            else:
                return stress
        else:
            print("parameter to calculate the stress is unknown")


class Well(TseriesBase):
    """Time series model consisting of the convolution of one or more stresses
    with one response function.

    Parameters
    ----------
    stress: pd.DataFrame
        Pandas DataFrame object containing the stresses.
    rfunc: rfunc class
        Response function used in the convolution with the stresses.
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

    Notes
    -----
    This class implements convolution of multiple series with a the same
    response function. This is often applied when dealing with multiple
    wells in a time series model.

    """

    def __init__(self, stress, rfunc, name, r=None, metadata=None,
                 xy=(0, 0), freq=None, fillna='mean', up=True, cutoff=0.99):
        # Check if number of stresses and radii match
        if len(stress.keys()) != len(r) and r:
            warn("The number of stresses applied does not match the number "
                 "of radii provided.")
        else:
            self.r = r

        # Check stresses
        if type(stress) is pd.Series:
            stress = [stress]

        TseriesBase.__init__(self, rfunc, name, xy, metadata,
                             self.stress.index.min(), self.stress.index.max(),
                             up, self.stress.mean(), cutoff)

        for i in stress:
            self.stress.append(check_tseries(stress[i], freq, fillna, name))

        self.freq = self.stress[0].index.freqstr
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, p=None, tindex=None, dt=1):
        h = pd.Series(data=0, index=self.stress[0].index, name=self.name)
        for i in self.stress:
            self.npoints = self.stress.index.size
            b = self.rfunc.block(p, self.r[i])  # nparam-1 depending on rfunc
            h += fftconvolve(self.stress[i], b, 'full')[:self.npoints]
        if tindex is not None:
            h = h[tindex]
        return h


class TseriesStep(TseriesBase):
    """
    A stress consisting of a step resonse from a specified time. The amplitude and form (if rfunc is not One) of the step is calibrated. Before t_step the response is zero.
    """

    def __init__(self, t_step, name, rfunc=One, xy=None, metadata=None,
                 up=True):
        assert t_step is not None, 'Error: Need to specify time of step (for now this will not be optimized)'

        TseriesBase.__init__(self, rfunc, name, xy, metadata,
                             pd.Timestamp.min, pd.Timestamp.max, up, 1, None)
        self.t_step = t_step
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = self.rfunc.set_parameters(self.name)
        self.parameters.loc['step_t'] = (
            self.t_step.value, pd.Timestamp.min.value, pd.Timestamp.max.value,
            0, self.name)
        self.nparam += 1

    def simulate(self, p, tindex=None, dt=1):
        assert tindex is not None, 'Error: Need an index'
        h = pd.Series(0, tindex, name=self.name)
        td = tindex - pd.Timestamp(p[-1])
        h[td.days > 0] = self.rfunc.step(p[:-1], td[td.days > 0].days)
        return h


class TseriesNoConv(TseriesBase):
    """
    Time series model consisting of the calculation of one stress with one
    response function, without the use of convolution (so it is slooooow)

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

    def __init__(self, stress, rfunc, name, metadata=None, xy=(0, 0),
                 freq=None, fillnan='mean', up=True, cutoff=0.99):
        stress = check_tseries(stress, freq, fillnan, name=name)
        TseriesBase.__init__(self, rfunc, name, xy, metadata,
                             stress.index.min(), stress.index.max(),
                             up, stress.mean(), cutoff)
        self.freq = stress.index.freqstr
        self.stress[name] = stress
        self.set_init_parameters()

    def set_init_parameters(self):
        """
        Set the initial parameters (back) to their default values.

        """
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, p, tindex=None, dt=None):
        """ Simulates the head contribution, without convolution.

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
        h = pd.Series(0, tindex, name=self.name)
        stress = self.stress.diff()
        if self.stress.values[0] != 0:
            stress = stress.set_value(
                stress.index[0] - (stress.index[1] - stress.index[0]),
                stress.columns, 0)
            stress = stress.sort_index()
        # set the index at the beginning of each step
        stress = stress.shift(-1).dropna()
        # remove steps that do not change
        stress = stress.loc[~(stress == 0).all(axis=1)]
        # tmax = self.rfunc.calc_tmax(p)
        for i in stress.index:
            erin = (h.index > i)  # & ((h.index-i).days<tmax)
            if any(erin):
                r = stress.loc[i][0] * self.rfunc.step(p, (
                h.index[erin] - i).days)
                h[erin] += r
                # h[np.invert(erin) & (h.index > i)] = r[-1]
        return h


class Constant(TseriesBase):
    """A constant value that is added to the time series model.

    Parameters
    ----------
    value : Optional[float]
        Initial estimate of the parameter value. E.g. The minimum of the observed
        series.

    """

    def __init__(self, name, xy=None, metadata=None, value=0.0,
                 pmin=np.nan, pmax=np.nan):
        self.nparam = 1
        self.value = value
        self.pmin = pmin
        self.pmax = pmax
        TseriesBase.__init__(self, One, name, xy, metadata,
                             pd.Timestamp.min, pd.Timestamp.max, 1, 0, 0)
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.parameters.loc['constant_d'] = (
            self.value, self.pmin, self.pmax, 1, self.name)

    def simulate(self, p=None):
        return p


class NoiseModel:
    """Noise model with exponential decay of the residual.

    Notes
    -----
    Calculates the innovations [1] according to:

    ..math:: v(t1) = r(t1) - r(t0) * exp(- (t1 - t0) / alpha)

    It can happen that the noisemodel is used in during the model calibration
    to explain most of the variation in the data. A recommended solution is to
    scale the initial parameter with the model timestep, E.g.:
    >>> n = NoiseModel()
    >>> n.set_initial("noise_alpha", 1.0 * ml.get_dt(ml.freq))

    References
    ----------
    .. [1] von Asmuth, J. R., and M. F. P. Bierkens (2005), Modeling irregularly
    spaced residual series as a continuous stochastic process, Water Resour.
    Res., 41, W12404, doi:10.1029/2004WR003726.

    """

    def __init__(self):
        self.nparam = 1
        self.set_init_parameters()

    def set_initial(self, name, value):
        """Method to set the initial parameter value

        Usage
        -----
        >>> ts.set_initial('parametername', 200)

        """
        if name in self.parameters.index:
            self.parameters.loc[name, 'initial'] = value
        else:
            print('Warning:', name, 'does not exist')

    def set_min(self, name, value):
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmin'] = value
        else:
            print('Warning:', name, 'does not exist')

    def set_max(self, name, value):
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmax'] = value
        else:
            print('Warning:', name, 'does not exist')

    def fix_parameter(self, name):
        if name in self.parameters.index:
            self.parameters.loc[name, 'vary'] = 0
        else:
            print('Warning:', name, 'does not exist')

    def set_init_parameters(self):
        self.parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.parameters.loc['noise_alpha'] = (14.0, 0, 5000, 1, 'noise')

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
        innovations = pd.Series(res, index=res.index, name="Innovations")
        # weights of innovations, see Eq. A17 in reference [1]
        power = (1.0 / (2.0 * (len(delt) - 1)))
        w = np.exp(power * np.sum(np.log(1 - np.exp(-2 * delt[1:] / p)))) / \
            np.sqrt(1.0 - np.exp(-2 * delt[1:] / p))
        # res.values is needed else it gets messed up with the dates
        innovations[1:] -= w * np.exp(-delt[1:] / p) * res.values[:-1]
        if tindex is not None:
            innovations = innovations[tindex]
        return innovations


class NoiseModel2:
    """Noise model with exponential decay of the residual.

    Notes
    -----
    Calculates the innovations [1] according to:

    ..math:: v(t1) = r(t1) - r(t0) * exp(- (t1 - t0) / alpha)

    It can happen that the noisemodel is used in during the model calibration
    to explain most of the variation in the data. A recommended solution is to
    scale the initial parameter with the model timestep, E.g.:
    >>> n = NoiseModel()
    >>> n.set_initial("noise_alpha", 1.0 * ml.get_dt(ml.freq))

    References
    ----------
    .. [1] von Asmuth, J. R., and M. F. P. Bierkens (2005), Modeling irregularly
    spaced residual series as a continuous stochastic process, Water Resour.
    Res., 41, W12404, doi:10.1029/2004WR003726.

    """

    def __init__(self):
        self.nparam = 1
        self.set_init_parameters()

    def set_initial(self, name, value):
        """Method to set the initial parameter value

        Usage
        -----
        >>> ts.set_initial('parametername', 200)

        """
        if name in self.parameters.index:
            self.parameters.loc[name, 'initial'] = value
        else:
            print('Warning:', name, 'does not exist')

    def set_min(self, name, value):
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmin'] = value
        else:
            print('Warning:', name, 'does not exist')

    def set_max(self, name, value):
        if name in self.parameters.index:
            self.parameters.loc[name, 'pmax'] = value
        else:
            print('Warning:', name, 'does not exist')

    def fix_parameter(self, name):
        if name in self.parameters.index:
            self.parameters.loc[name, 'vary'] = 0
        else:
            print('Warning:', name, 'does not exist')

    def set_init_parameters(self):
        self.parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.parameters.loc['noise_alpha'] = (14.0, 0, 5000, 1, 'noise')

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
        innovations: pd Series
            Series of the innovations.
        """
        innovations = pd.Series(res, index=res.index, name="Innovations")
        # res.values is needed else it gets messed up with the dates
        innovations[1:] -= np.exp(-delt[1:] / p) * res.values[:-1]
        if tindex is not None:
            innovations = innovations[tindex]
        return innovations