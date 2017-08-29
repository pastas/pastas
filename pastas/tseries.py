"""tseries module contains class for time series objects.

"""

from __future__ import print_function, division

from warnings import warn

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from .rfunc import One
from .timeseries import TimeSeries


class TseriesBase:
    _name = "TseriesBase"
    __doc__ = """Tseries Base class called by each Tseries object.

    Attributes
    ----------
    nparam : int
        Number of parameters.
    name : str
        Name of this tseries object. Used as prefix for the parameters.
    parameters : pandas.Dataframe
        Dataframe containing the parameters.

    """

    def __init__(self, rfunc, name, metadata, tmin, tmax, up, meanstress,
                 cutoff):
        self.rfunc = rfunc(up, meanstress, cutoff)
        self.nparam = self.rfunc.nparam
        self.name = name
        self.metadata = metadata
        self.tmin = tmin
        self.tmax = tmax
        self.freq = None
        self.stress = dict()

    def set_initial(self, name, value):
        """Method to set the initial parameter value.

        Examples
        --------

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

    def update_stress(self, **kwargs):
        """Method to change the frequency of the individual TimeSeries in
        the Pandas DataFrame.

        Parameters
        ----------
        freq

        Returns
        -------

        """
        for stress in self.stress.values():
            stress.update_series(**kwargs)

        if "freq" in kwargs:
            self.freq = kwargs["freq"]

    def get_stress(self, p=None, tindex=None):
        """Returns the stress or stresses of the time series object as a pandas
        DataFrame.

        If the time series object has multiple stresses each column
        represents a stress.

        Returns
        -------
        stress: pd.Dataframe
            Pandas dataframe of the stress(es)

        """
        if tindex is not None:
            return self.stress[tindex]
        else:
            return self.stress

    def export(self):
        data = dict()
        data["type"] = "TseriesBase"

        return data


class Tseries(TseriesBase):
    _name = "Tseries"
    __doc__ = """Time series model consisting of the convolution of one stress with one
    response function.

    Parameters
    ----------
    stress: pandas.Series
        pandas Series object containing the stress.
    rfunc: rfunc class
        Response function used in the convolution with the stress.
    name: str
        Name of the stress
    metadata: dict, optional
        dictionary containing metadata about the stress.
    xy: tuple, optional
        XY location in lon-lat format used for making maps.
    freq: str, optional
        Frequency to which the stress series are transformed. By default,
        the frequency is inferred from the data and that frequency is used.
        The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
        -aliases
    fillnan: str or float, optional
        Methods or float number to fill nan-values. Default values is
        'mean'. Currently supported options are: 'interpolate', float,
        and 'mean'. Interpolation is performed with a standard linear
        interpolation.
    norm_stress: Boolean, optional
        normalize the stress by subtracting the mean. For example this is
        convenient when simulating river levels.
    fill_before: Boolean, optional
        fill the stress-series before the beginning of this series with a certain value
    fill_after: Boolean, optional
        fill the stress-series after the end of this series with a certain value

    """

    def __init__(self, stress, rfunc, name, metadata=None, up=True,
                 cutoff=0.99, kind=None, settings=None):
        stress = TimeSeries(stress, name=name, kind=kind, settings=settings)

        TseriesBase.__init__(self, rfunc, name, metadata, stress.index.min(),
                             stress.index.max(), up, stress.mean(), cutoff)
        self.freq = stress.settings["freq"]
        self.stress[name] = stress
        self.set_init_parameters()

    def set_init_parameters(self):
        """Set the initial parameters (back) to their default values.

        """
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, p, tindex=None, dt=1):
        """Simulates the head contribution.

        Parameters
        ----------
        p: 1D array
           Parameters used for simulation.
        tindex: pandas.Series, optional
           Time indices to simulate the model.

        Returns
        -------
        pandas.Series
            The simulated head contribution.

        """
        b = self.rfunc.block(p, dt)
        stress = list(self.stress.values())[0]
        self.npoints = stress.index.size
        h = pd.Series(fftconvolve(stress, b, 'full')[:self.npoints],
                      index=stress.index, name=self.name)
        if tindex is not None:
            h = h[tindex]
        return h

    def export(self):
        """Method to export the Tseries object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the
            Tseries object.

        """
        data = dict()
        data["tseries_type"] = self._name
        data["stress"] = self.stress[self.name].series_original
        data["rfunc"] = self.rfunc._name
        data["name"] = self.name
        data["metadata"] = self.metadata
        data["up"] = True if self.rfunc.up == 1 else False
        data["cutoff"] = self.rfunc.cutoff
        data["kind"] = self.stress[self.name].kind
        data["settings"] = self.stress[self.name].settings
        return data


class Tseries2(TseriesBase):
    _name = "Tseries2"
    __doc__ = """Time series model consisting of the convolution of two stresses with one
    response function. The first stress causes the head to go up and the second
    stress causes the head to go down.

    Parameters
    ----------
    stress1: pandas.Series
        pandas Series object containing stress 1.
    stress2: pandas.Series
        pandas Series object containing stress 2.
    rfunc: rfunc class
        Response function used in the convolution with the stress.
    name: str
        Name of the stress
    metadata: dict, optional
        dictionary containing metadata about the stress.
    xy: tuple, optional
        XY location in lon-lat format used for making maps.
    freq: str, optional
        Frequency to which the stress series are transformed. By default,
        the frequency is inferred from the data and that frequency is used.
        The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
        -aliases
    fillnan: str or float, optional
        Methods or float number to fill nan-values. Default values is
        'mean'. Currently supported options are: 'interpolate', float,
        and 'mean'. Interpolation is performed with a standard linear
        interpolation.

    """

    def __init__(self, stress0, stress1, rfunc, name, metadata=None, up=True,
                 cutoff=0.99, kind=("prec", "evap"), settings=(None, None)):
        # First check the series, then determine tmin and tmax
        ts0 = TimeSeries(stress0, name="stress0", kind=kind[0],
                         settings=settings[0])
        ts1 = TimeSeries(stress1, name="stress1", kind=kind[1],
                         settings=settings[1])

        # Select indices from validated stress where both series are available.
        index = ts0.series.index & ts1.series.index

        # First check the series, then determine tmin and tmax
        stress0 = TimeSeries(stress0[index], name="stress0", kind=kind[0],
                             settings=settings[0])
        stress1 = TimeSeries(stress1[index], name="stress1", kind=kind[1],
                             settings=settings[1])

        if index.size is 0:
            warn('The two stresses that were provided have no overlapping time'
                 ' indices. Please make sure time indices overlap or apply to '
                 'separate time series objects.')

        TseriesBase.__init__(self, rfunc, name, metadata, index.min(),
                             index.max(), up, stress0.mean() - stress1.mean(),
                             cutoff)
        self.stress["stress0"] = stress0
        self.stress["stress1"] = stress1

        self.freq = stress0.settings["freq"]
        self.set_init_parameters()

    def set_init_parameters(self):
        """Set the initial parameters back to their default values.

        """
        self.parameters = self.rfunc.set_parameters(self.name)
        self.parameters.loc[self.name + '_f'] = (-1.0, -2.0, 2.0, 1, self.name)
        self.nparam += 1

    def simulate(self, p, tindex=None, dt=1):
        """Simulates the head contribution.

        Parameters
        ----------
        p: 1D array
           Parameters used for simulation.
        tindex: pandas.Series, optional
           Time indices to simulate the model.

        Returns
        -------
        pandas.Series
            The simulated head contribution.

        """
        b = self.rfunc.block(p[:-1], dt)
        self.npoints = self.stress["stress0"].index.size
        h = pd.Series(
            fftconvolve(self.stress["stress0"] + p[-1] * self.stress[
                "stress1"], b, 'full')[:self.npoints],
            index=self.stress["stress0"].index,
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

    def export(self):
        """Method to export the Tseries object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the
            Tseries object.

        """
        data = dict()
        data["tseries_type"] = self._name
        data["stress0"] = self.stress["stress0"].series_original
        data["stress1"] = self.stress["stress1"].series_original
        data["rfunc"] = self.rfunc._name
        data["name"] = self.name
        data["metadata"] = self.metadata
        data["up"] = True if self.rfunc.up == 1 else False
        data["cutoff"] = self.rfunc.cutoff
        data["kind"] = (self.stress["stress0"].kind,
                        self.stress["stress1"].kind)
        data["settings"] = (self.stress["stress0"].settings,
                            self.stress["stress1"].settings)
        return data


class Recharge(TseriesBase):
    _name = "Recharge"
    __doc__ = """Time series model performing convolution on groundwater recharge
    calculated from precipitation and evaporation with a single response function.

    Parameters
    ----------
    precip: pandas.Series
        pandas Series object containing the precipitation stress.
    evap: pandas.Series
        pandas Series object containing the evaporationstress.
    rfunc: rfunc class
        Response function used in the convolution with the stess.
    recharge: recharge_func class object
    name: str
        Name of the stress
    metadata: dict, optional
        dictionary containing metadata about the stress.
    xy: tuple, optional
        XY location in lon-lat format used for making maps.
    freq: list of str, optional
        Frequency to which the stress series are transformed. By default,
        the frequency is inferred from the data and that frequency is used.
        The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
        -aliases
    fillnan: list of str or float, optional
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
    R.A. Collenteur [2016] Non-linear time series analysis of deep groundwater levels: Application to the Veluwe. MSc. thesis, TU Delft. http://repository.tudelft.nl/view/ir/uuid:baf4fc8c-6311-407c-b01f-c80a96ecd584/

    """

    def __init__(self, prec, evap, rfunc, recharge, name, metadata=None,
                 cutoff=0.99, kind=("prec", "evap"), settings=(None, None)):
        # Check and name the time series
        prec1 = TimeSeries(prec, name="prec", kind=kind[0],
                           settings=settings[0])
        evap1 = TimeSeries(evap, name="evap", kind=kind[0],
                           settings=settings[0])

        # Select indices where both series are available
        index = prec1.series.index & evap1.series.index

        if index.size is 0:
            raise Warning('The two stresses that were provided have no '
                          'overlapping time indices. Please make sure time '
                          'indices overlap or apply to separate time series '
                          'objects.')

        # Store tmin and tmax
        TseriesBase.__init__(self, rfunc, name, metadata, index.min(),
                             index.max(), True, prec.mean() - evap.mean(),
                             cutoff)

        self.stress["prec"] = TimeSeries(prec[index], name="prec",
                                         kind=kind[0], settings=settings[0])
        self.stress["evap"] = TimeSeries(evap[index], name="evap",
                                         kind=kind[0], settings=settings[0])

        self.freq = self.stress["prec"].settings["freq"]

        self.recharge = recharge()
        self.set_init_parameters()
        self.nparam = self.rfunc.nparam + self.recharge.nparam

    def set_init_parameters(self):
        self.parameters = pd.concat([self.rfunc.set_parameters(self.name),
                                     self.recharge.set_parameters(self.name)])

    def simulate(self, p, tindex=None, dt=1):
        dt = int(dt)
        b = self.rfunc.block(p[:-self.recharge.nparam], dt)  # Block response
        # The recharge calculation needs arrays
        precip_array = np.array(self.stress["prec"])
        evap_array = np.array(self.stress["evap"])
        rseries = self.recharge.simulate(precip_array, evap_array,
                                         p[-self.recharge.nparam:])
        self.npoints = len(rseries)
        h = pd.Series(fftconvolve(rseries, b, 'full')[:self.npoints],
                      index=self.stress["prec"].index, name=self.name)
        if tindex is not None:
            h = h[tindex]
        return h

    def get_stress(self, p=None, tindex=None):
        """Returns the stress or stresses of the time series object as a pandas
        DataFrame. If the time series object has multiple stresses each column
        represents a stress.

        Parameters
        ----------
        tindex: pandas.TimeIndex

        Returns
        -------
        stress: pandas.DataFrame
            DataFrame containing the stresses with the required time indices.

        """

        # If parameters are not provided, don't calculate the recharge.
        if p is not None:
            precip_array = np.array(self.stress["prec"])
            evap_array = np.array(self.stress["evap"])
            rseries = self.recharge.simulate(precip_array,
                                             evap_array,
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
    _name = "Well"
    __doc__ = """Time series model consisting of the convolution of one or more
    stresses
    with one response function.

    Parameters
    ----------
    stress: pandas.DataFrame
        Pandas DataFrame object containing the stresses.
    rfunc: rfunc class
        Response function used in the convolution with the stresses.
    name: str
        Name of the stress
    metadata: dict, optional
        dictionary containing metadata about the stress.
    xy: tuple, optional
        XY location in lon-lat format used for making maps.
    freq: str, optional
        Frequency to which the stress series are transformed. By default,
        the frequency is inferred from the data and that frequency is used.
        The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
        -aliases
    fillnan: str or float, optional
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

    def __init__(self, stress, rfunc, name, radius, metadata=None,
                 up=True, cutoff=0.99, kind="well", settings=None):
        # Check if number of stresses and radii match
        if len(stress.keys()) != len(radius) and radius:
            warn("The number of stresses applied does not match the number "
                 "of radii provided.")
        else:
            self.r = radius

        # Check stresses
        if isinstance(stress, pd.Series):
            stress = [stress]

        TseriesBase.__init__(self, rfunc, name, metadata,
                             self.stress.index.min(), self.stress.index.max(),
                             up, self.stress.mean(), cutoff)

        for i, well in enumerate(stress):
            self.stress[name + str(i)] = TimeSeries(well, name=name, kind=kind,
                                                    settings=settings)

        self.freq = self.stress[name + "0"].settings["freq"]
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
    _name = "TseriesStep"
    __doc__ = """A stress consisting of a step resonse from a specified time. The
    amplitude and form (if rfunc is not One) of the step is calibrated. Before
    t_step the response is zero.

    """

    def __init__(self, t_step, name, rfunc=One, metadata=None, up=True):
        assert t_step is not None, 'Error: Need to specify time of step (for now this will not be optimized)'

        TseriesBase.__init__(self, rfunc, name, metadata, pd.Timestamp.min,
                             pd.Timestamp.max, up, 1.0, None, 0.0, 1.0)
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
    _name = "TseriesNoConv"
    __doc__ = """Time series model consisting of the calculation of one stress with one
    response function, without the use of convolution (so it is slooooow)

    Parameters
    ----------
    stress: pandas.Series
        pandas Series object containing the stress.
    rfunc: pastas.rfunc
        Response function used in the convolution with the stess.
    name: str
        Name of the stress
    metadata: dict, optional
        dictionary containing metadata about the stress.
    xy: tuple, optional
        XY location in lon-lat format used for making maps.
    freq: str, optional
        Frequency to which the stress series are transformed. By default,
        the frequency is inferred from the data and that frequency is used.
        The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    fillnan: str or float, optional
        Methods or float number to fill nan-values. Default values is
        'mean'. Currently supported options are: 'interpolate', float,
        and 'mean'. Interpolation is performed with a standard linear
        interpolation.

    """

    def __init__(self, stress, rfunc, name, metadata=None, up=True,
                 cutoff=0.99, kind=None, settings=None):
        stress = TimeSeries(stress, name=name, kind=kind, settings=settings)
        TseriesBase.__init__(self, rfunc, name, metadata, stress.index.min(),
                             stress.index.max(), up, stress.mean(), cutoff)
        self.freq = stress.settings["freq"]
        self.stress[name] = stress
        self.set_init_parameters()

    def set_init_parameters(self):
        """Set the initial parameters (back) to their default values.

        """
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, p, tindex=None, dt=None):
        """ Simulates the head contribution, without convolution.

        Parameters
        ----------
        p: array_like
           Parameters used for simulation.
        tindex: pandas.Series, optional
           Time indices to simulate the model.

        Returns
        -------
        pandas.Series
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
    _name = "Constant"
    __doc__ = """A constant value that is added to the time series model.

    Parameters
    ----------
    value : float, optional
        Initial estimate of the parameter value. E.g. The minimum of the
        observed series.

    """

    def __init__(self, name, metadata=None, value=0.0,
                 pmin=np.nan, pmax=np.nan):
        self.nparam = 1
        self.value = value
        self.pmin = pmin
        self.pmax = pmax
        self.name = "constant"
        TseriesBase.__init__(self, One, name, metadata,
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
        self.nparam = 1
        self.name = "noise"
        self.set_init_parameters()

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
        innovations[1:] -= np.exp(-delt[1:] / p) * res.values[:-1]
        if tindex is not None:
            innovations = innovations[tindex]
        return innovations

    def export(self):
        data = dict()
        data["type"] = "NoiseModel"
        return data
