"""The stressmodels module contains all the stressmodels that available in
Pastas.

Supported Stressmodels
----------------------
The following stressmodels are supported and tested:

- StressModel
- StressModel2
- FactorModel
- StepModel

All other stressmodels are for research purposes only and are not (yet)
fully supported and tested.

TODO
----
- Test and support StepModel
- Test and support LinearTrend
- Test and support WellModel


"""

from logging import getLogger

from importlib import import_module

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from .decorators import set_parameter
from .rfunc import One
from .timeseries import TimeSeries

logger = getLogger(__name__)

__all__ = ["StressModel", "StressModel2", "Constant", "StepModel",
           "LinearTrend", "FactorModel", "RechargeModel"]


class StressModelBase:
    """StressModel Base class called by each StressModel object.

    Attributes
    ----------
    nparam : int
        Number of parameters.
    name : str
        Name of this stressmodel object. Used as prefix for the parameters.
    parameters : pandas.DataFrame
        Dataframe containing the parameters.

    """
    _name = "StressModelBase"

    def __init__(self, rfunc, name, tmin, tmax, up, meanstress, cutoff):
        self.rfunc = rfunc(up, meanstress, cutoff)
        self.parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.name = name
        self.tmin = tmin
        self.tmax = tmax
        self.freq = None
        self.stress = []

    @property
    def nparam(self):
        return self.parameters.index.size

    def set_init_parameters(self):
        """Set the initial parameters (back) to their default values.

        """
        pass

    @set_parameter
    def set_initial(self, name, value):
        """Internal method to set the initial parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        self.parameters.loc[name, 'initial'] = value

    @set_parameter
    def set_pmin(self, name, value):
        """Internal method to set the lower bound of the parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        self.parameters.loc[name, 'pmin'] = value

    @set_parameter
    def set_pmax(self, name, value):
        """Internal method to set the upper bound of the parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        self.parameters.loc[name, 'pmax'] = value

    @set_parameter
    def set_vary(self, name, value):
        """Internal method to set if the parameter is varied during
        optimization.

        Notes
        -----
        The preferred method for parameter setting is through the model.

        """
        self.parameters.loc[name, 'vary'] = value

    def update_stress(self, **kwargs):
        """Method to change the frequency of the individual TimeSeries in
        the Pandas DataFrame.

        Parameters
        ----------
        freq

        Returns
        -------

        """
        for stress in self.stress:
            stress.update_series(**kwargs)

        if "freq" in kwargs:
            self.freq = kwargs["freq"]

    def handle_stress(self, stress, settings):
        """Method to handle user provided stress in init

        Parameters
        ----------
        stress: pandas.Series, pastas.TimeSeries or iterable
        settings: dict or iterable

        Returns
        -------
        stress: dict
            dictionary with strings

        """
        data = []

        if isinstance(stress, pd.Series):
            data.append(TimeSeries(stress, settings))
        elif isinstance(stress, dict):
            for i, value in enumerate(stress.values()):
                data.append(TimeSeries(value, settings=settings[i]))
        elif isinstance(stress, list):
            for i, value in enumerate(stress):
                data.append(TimeSeries(value, settings=settings[i]))
        else:
            logger.warning("provided stress format is unknown. Provide a"
                           "Series,dict or list.")
        return data

    def dump_stress(self, series=True):
        """Method to dump all stresses in the stresses list.

        Parameters
        ----------
        data: dict
            Dictionary for the data to go into.
        series: Boolean
            True if time series are to be exported, False if only the name
            of the time series are needed. Settings are always exported.

        Returns
        -------
        data: dict
            dictionary with the dump of the stresses.

        """
        data = []

        for stress in self.stress:
            data.append(stress.dump(series=series))

        return data

    def get_stress(self, p=None, original=False, **kwargs):
        """Returns the stress or stresses of the time series object as a pandas
        DataFrame.

        If the time series object has multiple stresses each column
        represents a stress.

        Returns
        -------
        stress: pd.Dataframe
            Pandas dataframe of the stress(es)

        """
        if original:
            return self.stress[0].series_original
        else:
            return self.stress[0].series

    def dump(self, series=True):
        """Method to export the StressModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the
            StressModel object.

        """
        data = {
            "stressmodel": self._name,
            "name": self.name,
            "stress": self.dump_stress(series)
        }
        return data


class StressModel(StressModelBase):
    """Time series model consisting of the convolution of one stress with one
    response function.

    Parameters
    ----------
    stress: pandas.Series
        pandas Series object containing the stress.
    rfunc: rfunc class
        Response function used in the convolution with the stress.
    name: str
        Name of the stress.
    up: Boolean, optional
        True if response function is positive (default), False if negative.
    cutoff: float, optional
        float between 0 and 1 to determine how long the response is (default
        is 99% of the actual response time). Used to reduce computation times.
    settings: dict or str, optional
        The settings of the stress. This can be a string referring to a
        predefined settings dict, or a dict with the settings to apply.
        Refer to the docstring of pastas.Timeseries for further information.
    metadata: dict, optional
        dictionary containing metadata about the stress. This is passed onto
        the TimeSeries object.
    meanstress: float, optional
        The mean stress determines the initial parameters of rfunc. The initial
        parameters are chosen in such a way that the gain of meanstress is 1.

    Examples
    --------
    >>> import pastas as ps
    >>> import pandas as pd
    >>> sm = ps.StressModel(stress=pd.Series(), rfunc=ps.Gamma, name="Prec", \
    >>> settings="prec")

    See Also
    --------
    pastas.rfunc
    pastas.TimeSeries

    """
    _name = "StressModel"

    def __init__(self, stress, rfunc, name, up=True, cutoff=0.99,
                 settings=None, metadata=None, meanstress=None):
        if isinstance(stress, list):
            stress = stress[0]  # Temporary fix Raoul, 2017-10-24

        stress = TimeSeries(stress, settings=settings, metadata=metadata)

        if meanstress is None:
            meanstress = stress.series.std()

        StressModelBase.__init__(self, rfunc, name, stress.series.index.min(),
                                 stress.series.index.max(), up, meanstress,
                                 cutoff)
        self.freq = stress.settings["freq"]
        self.stress = [stress]
        self.set_init_parameters()

    def set_init_parameters(self):
        """Set the initial parameters (back) to their default values.

        """
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1):
        """Simulates the head contribution.

        Parameters
        ----------
        p: 1D array
           Parameters used for simulation.
        tmin: str, optional
        tmax: str, optional
        freq: str, optional

        Returns
        -------
        pandas.Series
            The simulated head contribution.

        """
        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)
        b = self.rfunc.block(p, dt)
        stress = self.stress[0].series
        npoints = stress.index.size
        h = pd.Series(data=fftconvolve(stress, b, 'full')[:npoints],
                      index=stress.index, name=self.name, fastpath=True)
        return h

    def dump(self, series=True):
        """Method to export the StressModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the
            StressModel object.

        """
        data = {
            "stressmodel": self._name,
            "rfunc": self.rfunc._name,
            "name": self.name,
            "up": True if self.rfunc.up == 1 else False,
            "cutoff": self.rfunc.cutoff,
            "stress": self.dump_stress(series)
        }
        return data


class StressModel2(StressModelBase):
    """Time series model consisting of the convolution of two stresses with one
    response function. The first stress causes the head to go up and the second
    stress causes the head to go down.

    Parameters
    ----------
    stress: list of pandas.Series or list of pastas.TimeSeries
        list of pandas.Series or pastas.TimeSeries objects containing the
        stresses.
    rfunc: pastas.rfunc instance
        Response function used in the convolution with the stress.
    name: str
        Name of the stress
    up: Boolean, optional
        True if response function is positive (default), False if negative.
    cutoff: float
        float between 0 and 1 to determine how long the response is (default
        is 99% of the actual response time). Used to reduce computation times.
    settings: Tuple with two dicts
        The settings of the individual TimeSeries.
    settings: list of dicts or strs, optional
        The settings of the stresses. This can be a string referring to a
        predefined settings dict, or a dict with the settings to apply.
        Refer to the docstring of pastas.Timeseries for further information.
        Default is ("prec", "evap").
    metadata: list of dicts, optional
        dictionary containing metadata about the stress. This is passed onto
        the TimeSeries object.

    Notes
    -----
    The order in which the stresses are provided is the order the metadata
    and settings dictionaries or string are passed onto the TimeSeries
    objects. By default, the precipitation stress is the first and the
    evaporation stress the second stress.

    See Also
    --------
    pastas.rfunc
    pastas.TimeSeries

    """
    _name = "StressModel2"

    def __init__(self, stress, rfunc, name, up=True, cutoff=0.99,
                 settings=("prec", "evap"), metadata=(None, None),
                 meanstress=None):
        # First check the series, then determine tmin and tmax
        stress0 = TimeSeries(stress[0], settings=settings[0],
                             metadata=metadata[0])
        stress1 = TimeSeries(stress[1], settings=settings[1],
                             metadata=metadata[1])

        # Select indices from validated stress where both series are available.
        index = stress0.series.index.intersection(stress1.series.index)
        if index.empty:
            msg = ('The two stresses that were provided have no '
                  'overlapping time indices. Please make sure the '
                  'indices of the time series overlap.')
            logger.error(msg)
            raise Exception(msg)

        # First check the series, then determine tmin and tmax
        stress0.update_series(tmin=index.min(), tmax=index.max())
        stress1.update_series(tmin=index.min(), tmax=index.max())

        if meanstress is None:
            meanstress = (stress0.series - stress1.series).std()

        StressModelBase.__init__(self, rfunc, name, index.min(), index.max(),
                                 up, meanstress, cutoff)
        self.stress.append(stress0)
        self.stress.append(stress1)

        self.freq = stress0.settings["freq"]
        self.set_init_parameters()

    def set_init_parameters(self):
        """Set the initial parameters back to their default values.

        """
        self.parameters = self.rfunc.set_parameters(self.name)
        self.parameters.loc[self.name + '_f'] = (-1.0, -2.0, 0.0, 1, self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1, istress=None):
        """Simulates the head contribution.

        Parameters
        ----------
        p: 1D array
           Parameters used for simulation.
        tmin: str, optional
        tmax: str, optional
        freq: str, optional

        Returns
        -------
        pandas.Series
            The simulated head contribution.

        """
        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)
        b = self.rfunc.block(p[:-1], dt)
        stress = self.get_stress(p=p, istress=istress)
        npoints = stress.index.size
        h = pd.Series(data=fftconvolve(stress, b, 'full')[:npoints],
                      index=stress.index, name=self.name, fastpath=True)
        if istress is not None:
            if self.stress[istress].name is not None:
                h.name = h.name + ' (' + self.stress[istress].name + ')'
        # see whether it makes a difference to subtract gain * mean_stress
        # h -= self.rfunc.gain(p) * stress.mean()
        return h

    def get_stress(self, p=None, original=False, istress=None):
        if istress is None:
            return self.stress[0].series.add(p[-1] * self.stress[1].series)
        elif istress == 0:
            return self.stress[0].series
        else:
            return p[-1] * self.stress[1].series

    def dump(self, series=True):
        """Method to export the StressModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the
            StressModel object.

        """
        data = {
            "stressmodel": self._name,
            "rfunc": self.rfunc._name,
            "name": self.name,
            "up": True if self.rfunc.up == 1 else False,
            "cutoff": self.rfunc.cutoff,
            "stress": self.dump_stress(series)
        }
        return data


class StepModel(StressModelBase):
    """Stressmodel that simulates a step trend.

    Parameters
    ----------
    start: str
        String with the start date of the step, e.g. '2018-01-01'. This
        value is fixed by default. Use ml.set_vary("step_tstart", 1) to vary
        the start time of the step trend.
    name: str
        String with the name of the stressmodel.
    rfunc: pastas.rfunc.RfuncBase
        Pastas response function used to simulate the effect of the step.
        Default is rfunc.One()

    Notes
    -----
    This step trend is calculated as follows. First, a binary series is
    created, with zero values before tstart, and ones after the start. This
    series is convoluted with the block response to simulate a step trend.

    """
    _name = "StepModel"

    def __init__(self, tstart, name, rfunc=One, up=True):
        StressModelBase.__init__(self, rfunc, name, pd.Timestamp.min,
                                 pd.Timestamp.max, up, 1.0, 0.99)
        self.tstart = pd.Timestamp(tstart)
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = self.rfunc.set_parameters(self.name)
        tmin = pd.Timestamp.min.toordinal()
        tmax = pd.Timestamp.max.toordinal()
        tinit = self.tstart.toordinal()

        self.parameters.loc[self.name + "_tstart"] = (tinit, tmin, tmax,
                                                      0, self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1):
        tstart = pd.Timestamp.fromordinal(int(p[-1]), freq="D")
        tindex = pd.date_range(tmin, tmax, freq=freq)
        h = pd.Series(0, tindex, name=self.name)
        h.loc[h.index > tstart] = 1

        b = self.rfunc.block(p[:-1], dt)
        npoints = h.index.size
        h = pd.Series(data=fftconvolve(h, b, 'full')[:npoints],
                      index=h.index, name=self.name, fastpath=True)
        return h

    def dump(self, series=True):
        data = {
            "stressmodel": self._name,
            'tstart': self.tstart,
            'name': self.name,
            "up": True if self.rfunc.up == 1 else False,
            'rfunc': self.rfunc._name
        }
        return data


class LinearTrend(StressModelBase):
    """Stressmodel that simulates a linear trend.

    name: str
        String with the name of the stressmodel
    start: str
        String with a date to start the trend, will be transformed to an
        ordinal number internally. E.g. "2018-01-01"
    end: str
        String with a date to end the trend, will be transformed to an ordinal
        number internally. E.g. "2018-01-01"

    """
    _name = "LinearTrend"

    def __init__(self, name="linear_trend", start=0, end=0):
        StressModelBase.__init__(self, One, name, pd.Timestamp.min,
                                 pd.Timestamp.max, 1, 0, 0)
        self.start = start
        self.end = end
        self.set_init_parameters(start, end)

    def set_init_parameters(self, start, end):
        start = pd.Timestamp(start).toordinal()
        end = pd.Timestamp(end).toordinal()
        tmin = pd.Timestamp.min.toordinal()
        tmax = pd.Timestamp.max.toordinal()

        self.parameters.loc[self.name + "_a"] = (
            0, -np.inf, np.inf, 1, self.name)
        self.parameters.loc[self.name + "_tstart"] = (
            start, tmin, tmax, 1, self.name)
        self.parameters.loc[self.name + "_tend"] = (
            end, tmin, tmax, 1, self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1):
        tindex = pd.date_range(tmin, tmax, freq=freq)

        if p[1] < tindex[0].toordinal():
            tmin = tindex[0]
        else:
            tmin = pd.Timestamp.fromordinal(int(p[1]))

        if p[2] >= tindex[-1].toordinal():
            tmax = tindex[-1]
        else:
            tmax = pd.Timestamp.fromordinal(int(p[2]))

        trend = tindex.to_series().diff() / pd.Timedelta(1, "D")
        trend.loc[:tmin] = 0
        trend.loc[tmax:] = 0
        trend = trend.cumsum() * p[0]
        return trend

    def dump(self, series=None):
        data = {
            "stressmodel": self._name,
            'start': self.start,
            "end": self.end,
            'name': self.name,
        }
        return data


class Constant(StressModelBase):
    """A constant value that is added to the time series model.

    Parameters
    ----------
    value : float, optional
        Initial estimate of the parameter value. E.g. The minimum of the
        observed series.

    """
    _name = "Constant"

    def __init__(self, name="constant", value=0.0, pmin=np.nan, pmax=np.nan):
        self.value = value
        self.pmin = pmin
        self.pmax = pmax
        StressModelBase.__init__(self, One, name, pd.Timestamp.min,
                                 pd.Timestamp.max, 1, 0, 0)
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters.loc[self.name + "_d"] = (
            self.value, self.pmin, self.pmax, 1, self.name)

    def simulate(self, p=None):
        return p


class WellModel(StressModelBase):
    """Time series model consisting of the convolution of one or more
    stresses with one response function. The distance from an influence to
    the location of the oseries has to be provided for each

    Parameters
    ----------
    stress: pandas.DataFrame
        Pandas DataFrame object containing the stresses.
    rfunc: rfunc class
        Response function used in the convolution with the stresses.
    name: str
        Name of the stress

    Notes
    -----
    This class implements convolution of multiple series with a the same
    response function. This is often applied when dealing with multiple
    wells in a time series model.

    """
    _name = "WellModel"

    def __init__(self, stress, rfunc, name, radius, up=False, cutoff=0.99,
                 settings="well"):

        meanstress = 1.0  # ? this should be something logical

        tmin = pd.Timestamp.max
        tmax = pd.Timestamp.min

        StressModelBase.__init__(self, rfunc, name, tmin, tmax,
                                 up, meanstress, cutoff)

        if settings is None or isinstance(settings, str):
            settings = len(stress) * [None]

        self.stress = self.handle_stress(stress, settings)

        # Check if number of stresses and radii match
        if len(self.stress) != len(radius) and radius:
            logger.error("The number of stresses applied does not match the "
                         "number of radii provided.")
        else:
            self.radius = radius

        self.freq = self.stress[0].settings["freq"]

        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = self.rfunc.set_parameters(self.name)

    def simulate(self, p=None, tmin=None, tmax=None, freq=None, dt=1,
                 istress=None):
        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)
        h = pd.Series(data=0, index=self.stress[0].series.index,
                      name=self.name)
        stresses = self.get_stress(istress=istress)
        radii = self.get_radii(irad=istress)
        for stress, radius in zip(stresses, radii):
            npoints = stress.index.size
            # TODO Make response function that take the radius as input
            # b = self.rfunc.block(p, dt=dt, radius=radius)
            b = self.rfunc.block(p, dt)
            c = fftconvolve(stress, b, 'full')[:npoints]
            h = h.add(pd.Series(c, index=stress.index), fill_value=0.0)

        return h

    def get_stress(self, p=None, istress=None):
        if istress is None:
            return [s.series for s in self.stress]
        else:
            return [self.stress[istress].series]

    def get_radii(self, irad=None):
        if irad is None:
            return self.radius
        else:
            return [self.radius[irad]]


class FactorModel(StressModelBase):
    """Model that multiplies a stress by a single value. The indepedent series
    do not have to be equidistant and are allowed to have gaps.

    Parameters
    ----------
    stress: pandas.Series or pastas.TimeSeries
        Stress which will be multiplied by a factor. The stress does not
        have to be equidistant.
    name: str
        String with the name of the stressmodel.
    settings: dict or str
        Dict or String that is forwarded to the TimeSeries object created
        from the stress.
    metadata: dict
        Dictionary with metadata, forwarded to the TimeSeries object created
        from the stress.

    """
    _name = "FactorModel"

    def __init__(self, stress, name="factor", settings=None, metadata=None):
        if isinstance(stress, list):
            stress = stress[0]  # Temporary fix Raoul, 2017-10-24
        tmin = stress.series_original.index.min()
        tmax = stress.series_original.index.max()
        StressModelBase.__init__(self, One, name, tmin=tmin, tmax=tmax,
                                 up=True, meanstress=1, cutoff=0.99)
        self.value = 1  # Initial value
        stress = TimeSeries(stress, settings=settings, metadata=metadata)
        self.stress = [stress]
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters.loc[self.name + "_f"] = (
            self.value, -np.inf, np.inf, 1, self.name)

    def simulate(self, p=None, tmin=None, tmax=None, freq=None, dt=1):
        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)
        return self.stress[0].series * p[0]

    def dump(self, series=True):
        """Method to export the StressModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the
            StressModel object.

        """
        data = {
            "stressmodel": self._name,
            "name": self.name,
            "stress": self.dump_stress(series)
        }
        return data


class RechargeModel(StressModelBase):
    """Stressmodel simulating the effect of groundwater recharge on the
    groundwater head.

    Parameters
    ----------
    prec: pandas.Series or pastas.TimeSeries
        pandas.Series or pastas.TimeSeries objects containing the
        precipitation series.
    evap: pandas.Series or pastas.TimeSeries
        pandas.Series or pastas.TimeSeries objects containing the
        evaporation series.
    rfunc: pastas.rfunc instance
        Response function used in the convolution with the stress.
    name: str
        Name of the stress
    recharge: string, optional
        String with the name of the recharge model. Options are: "Linear" (
        default).
    up: Boolean, optional
        True if response function is positive (default), False if negative.
    cutoff: float
        float between 0 and 1 to determine how long the response is (default
        is 99% of the actual response time). Used to reduce computation times.
    settings: list of dicts or strs, optional
        The settings of the precipitation and evaporation time series,
        in this order. This can be a string referring to a predefined
        settings dict, or a dict with the settings to apply. Refer to the
        docstring of pastas.Timeseries for further information. Default is (
        "prec", "evap").
    metadata: list of dicts, optional
        dictionary containing metadata about the stress. This is passed onto
        the TimeSeries object.

    See Also
    --------
    pastas.rfunc
    pastas.TimeSeries
    pastas.recharge

    Notes
    -----
    This stressmodel computes the contribution of precipitation and
    potential evaporation in two steps. In the first step a recharge flux is
    computed by a method determined by the recharge input argument. In de
    second step this recharge flux is convoluted with a response function to
    obtain the final contribution.

    """
    _name = "Recharge"

    def __init__(self, prec, evap, rfunc, name, recharge="Linear", up=True,
                 cutoff=0.99, settings=("prec", "evap"), metadata=(None, None),
                 meanstress=None):
        # First check the series, then determine tmin and tmax
        stress0 = TimeSeries(prec, settings=settings[0], metadata=metadata[0])
        stress1 = TimeSeries(evap, settings=settings[1], metadata=metadata[1])

        # Select indices from validated stress where both series are available.
        index = stress0.series.index.intersection(stress1.series.index)
        if index.size is 0:
            logger.error('The two stresses that were provided have no '
                         'overlapping time indices. Please make sure the '
                         'indices of the time series overlap.')

        # First check the series, then determine tmin and tmax
        stress0.update_series(tmin=index.min(), tmax=index.max())
        stress1.update_series(tmin=index.min(), tmax=index.max())

        if meanstress is None:
            meanstress = (stress0.series - stress1.series).std()

        StressModelBase.__init__(self, rfunc, name, index.min(), index.max(),
                                 up, meanstress, cutoff)
        self.prec = stress0
        self.evap = stress1

        self.freq = stress0.settings["freq"]

        # Dynamically load the required recharge model from string
        recharge = getattr(import_module("pastas.recharge"), recharge)
        self.recharge = recharge()

        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = pd.concat([self.rfunc.set_parameters(self.name),
                                     self.recharge.set_parameters(self.name)])

    def update_stress(self, **kwargs):
        """Method to change the frequency of the individual TimeSeries in
        the Pandas DataFrame.

        Parameters
        ----------
        freq

        Returns
        -------

        """
        self.prec.update_series(**kwargs)
        self.evap.update_series(**kwargs)

        if "freq" in kwargs:
            self.freq = kwargs["freq"]

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1):
        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)
        b = self.rfunc.block(p[:-self.recharge.nparam], dt)
        stress = self.get_stress(p[-self.recharge.nparam:])
        npoints = stress.index.size
        h = pd.Series(data=fftconvolve(stress, b, 'full')[:npoints],
                      index=stress.index, name=self.name, fastpath=True)
        return h

    def get_stress(self, p=None, original=False, istress=None):
        if istress is None:
            prec = self.prec.series
            evap = self.evap.series
            stress = self.recharge.simulate(prec, evap, p)

            stress = pd.Series(data=stress, index=prec.index,
                               name="recharge", fastpath=True)
            return stress
        elif istress == 0:
            return self.prec.series
        else:
            return self.evap.series
