"""The stressmodels module contains all the stressmodels that available in
Pastas.

Supported Stressmodels
----------------------
The following stressmodels are supported and tested:

- StressModel
- StressModel2
- FactorModel
- StepModel
- WellModel

All other stressmodels are for research purposes only and are not (yet)
fully supported and tested.

TODO
----
- Test and support StepModel
- Test and support LinearTrend

"""

from importlib import import_module
from logging import getLogger

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from .decorators import set_parameter
from .rfunc import One, Exponential, HantushWellModel
from .timeseries import TimeSeries
from .utils import validate_name

logger = getLogger(__name__)

__all__ = ["StressModel", "StressModel2", "Constant", "StepModel",
           "LinearTrend", "FactorModel", "RechargeModel"]


class StressModelBase:
    """StressModel Base class called by each StressModel object.

    Attributes
    ----------
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
        self.name = validate_name(name)
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
        self.parameters.loc[name, 'vary'] = bool(value)

    def update_stress(self, **kwargs):
        """Method to update the settings of the individual TimeSeries.

        Notes
        -----
        For the individual options for the different settings please refer to
        the docstring from the TimeSeries.update_series() method.

        See Also
        --------
        ps.TimeSeries.update_series

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
                           "Series, dict or list.")
        return data

    def dump_stress(self, series=True):
        """Method to dump all stresses in the stresses list.

        Parameters
        ----------
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
            data.append(stress.to_dict(series=series))

        return data

    def get_stress(self, p=None, tmin=None, tmax=None, freq=None,
                   istress=None, **kwargs):
        """Returns the stress or stresses of the time series object as a pandas
        DataFrame.

        If the time series object has multiple stresses each column
        represents a stress.

        Returns
        -------
        stress: pd.Dataframe
            Pandas dataframe of the stress(es)

        """
        return self.stress[0].series

    def to_dict(self, series=True):
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

    def get_nsplit(self):
        """Determine in how many timeseries the contribution can be splitted"""
        if hasattr(self, 'nsplit'):
            return self.nsplit
        else:
            return len(self.stress)

    def get_block(self, p, dt, tmin, tmax):
        """Internal method to get the block-response from the respnse function"""
        if tmin is not None and tmax is not None:
            day = pd.to_timedelta(1, 'd')
            maxtmax = (pd.Timestamp(tmax) - pd.Timestamp(tmin)) / day
        else:
            maxtmax = None
        b = self.rfunc.block(p, dt, maxtmax=maxtmax)
        return b


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
    up: Boolean or None, optional
        True if response function is positive (default), False if negative.
        None if you don't want to define if response is positive or negative.
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
                            settings="prec")

    See Also
    --------
    pastas.rfunc
    pastas.timeseries.TimeSeries

    """
    _name = "StressModel"

    def __init__(self, stress, rfunc, name, up=True, cutoff=0.999,
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
        self.parameters = self.rfunc.get_init_parameters(self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1):
        """Simulates the head contribution.

        Parameters
        ----------
        p: 1D array
           Parameters used for simulation.
        tmin: str, optional
        tmax: str, optional
        freq: str, optional
        dt: int, optional

        Returns
        -------
        pandas.Series
            The simulated head contribution.

        """
        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)
        b = self.get_block(p, dt, tmin, tmax)
        stress = self.stress[0].series
        npoints = stress.index.size
        h = pd.Series(data=fftconvolve(stress, b, 'full')[:npoints],
                      index=stress.index, name=self.name, fastpath=True)
        return h

    def to_dict(self, series=True):
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
            "up": self.rfunc.up,
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
    up: Boolean or None, optional
        True if response function is positive (default), False if negative.
        None if you don't want to define if response is positive or negative.
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

    def __init__(self, stress, rfunc, name, up=True, cutoff=0.999,
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
        self.parameters = self.rfunc.get_init_parameters(self.name)
        self.parameters.loc[self.name + '_f'] = \
            (-1.0, -2.0, 0.0, True, self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1, istress=None):
        """Simulates the head contribution.

        Parameters
        ----------
        p: 1D array
           Parameters used for simulation.
        tmin: str, optional
        tmax: str, optional
        freq: str, optional
        dt: int, optional
        istress: int, optional

        Returns
        -------
        pandas.Series
            The simulated head contribution.

        """
        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)
        b = self.get_block(p[:-1], dt, tmin, tmax)
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

    def get_stress(self, p=None, istress=None, **kwargs):
        if istress is None:
            if p is None:
                p = self.parameters.initial.values
            return self.stress[0].series.add(p[-1] * self.stress[1].series)
        elif istress == 0:
            return self.stress[0].series
        else:
            return p[-1] * self.stress[1].series

    def to_dict(self, series=True):
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
            "up": self.rfunc.up,
            "cutoff": self.rfunc.cutoff,
            "stress": self.dump_stress(series)
        }
        return data


class StepModel(StressModelBase):
    """Stressmodel that simulates a step trend.

    Parameters
    ----------
    tstart: str
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

    def __init__(self, tstart, name, rfunc=One, up=None):
        StressModelBase.__init__(self, rfunc, name, pd.Timestamp.min,
                                 pd.Timestamp.max, up, 1.0, 0.99)
        self.tstart = pd.Timestamp(tstart)
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = self.rfunc.get_init_parameters(self.name)
        tmin = pd.Timestamp.min.toordinal()
        tmax = pd.Timestamp.max.toordinal()
        tinit = self.tstart.toordinal()

        self.parameters.loc[self.name + "_tstart"] = (tinit, tmin, tmax,
                                                      False, self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1):
        tstart = pd.Timestamp.fromordinal(int(p[-1]), freq="D")
        tindex = pd.date_range(tmin, tmax, freq=freq)
        h = pd.Series(0, tindex, name=self.name)
        h.loc[h.index > tstart] = 1

        b = self.get_block(p[:-1], dt, tmin, tmax)
        npoints = h.index.size
        h = pd.Series(data=fftconvolve(h, b, 'full')[:npoints],
                      index=h.index, name=self.name, fastpath=True)
        return h

    def to_dict(self, series=True):
        data = {
            "stressmodel": self._name,
            'tstart': self.tstart,
            'name': self.name,
            "up": self.rfunc.up,
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
        self.set_init_parameters()

    def set_init_parameters(self):
        start = pd.Timestamp(self.start).toordinal()
        end = pd.Timestamp(self.end).toordinal()
        tmin = pd.Timestamp.min.toordinal()
        tmax = pd.Timestamp.max.toordinal()

        self.parameters.loc[self.name + "_a"] = (
            0, -np.inf, np.inf, True, self.name)
        self.parameters.loc[self.name + "_tstart"] = (
            start, tmin, tmax, True, self.name)
        self.parameters.loc[self.name + "_tend"] = (
            end, tmin, tmax, True, self.name)

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

    def to_dict(self, series=None):
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

    def __init__(self, name="constant", initial=0.0):
        StressModelBase.__init__(self, One, name, pd.Timestamp.min,
                                 pd.Timestamp.max, None, initial, 0)
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = self.rfunc.get_init_parameters(self.name)

    @staticmethod
    def simulate(p=None):
        return p


class WellModel(StressModelBase):
    """Time series model consisting of the convolution of one or more
    stresses with one response function. The distance from an influence to
    the location of the oseries has to be provided for each stress.

    Parameters
    ----------
    stress : list
        list containing the stresses timeseries.
    rfunc : pastas.rfunc
        WellModel only works with Hantush!
    name : str
        Name of the stressmodel.
    distances : list or list-like
        list of distances to oseries, must be ordered the same as the
        stresses.
    up : bool, optional
        whether positive stress has increasing or decreasing effect on
        the model, by default False, in which case positive stress lowers
        e.g. the groundwater level.
    cutoff : float, optional
        percentage at which to cutoff the step response, by default 0.99.
    settings : str, list of dict, optional
        settings of the timeseries, by default "well".
    sort_wells : bool, optional
        sort wells from closest to furthest, by default True.

    Notes
    -----
    This class implements convolution of multiple series with a the same
    response function. This can be applied when dealing with multiple
    wells in a time series model.

    """
    _name = "WellModel"

    def __init__(self, stress, rfunc, name, distances, up=False, cutoff=0.999,
                 settings="well", sort_wells=True):
        if not issubclass(rfunc, HantushWellModel):
            raise NotImplementedError("WellModel only supports rfunc "
                                      "HantushWellModel!")

        # sort wells by distance
        self.sort_wells = sort_wells
        if self.sort_wells:
            stress = [s for _, s in sorted(zip(distances, stress),
                                           key=lambda pair: pair[0])]
            if isinstance(settings, list):
                settings = [s for _, s in sorted(zip(distances, settings),
                                                 key=lambda pair: pair[0])]
            distances.sort()

        # get largest std for meanstress
        meanstress = np.max([s.series.std() for s in stress])

        tmin = pd.Timestamp.max
        tmax = pd.Timestamp.min

        StressModelBase.__init__(self, rfunc, name, tmin, tmax,
                                 up, meanstress, cutoff)

        if settings is None or isinstance(settings, str):
            settings = len(stress) * [None]

        self.stress = self.handle_stress(stress, settings)

        # Check if number of stresses and distances match
        if len(self.stress) != len(distances):
            msg = "The number of stresses applied does not match the  number" \
                  "of distances provided."
            logger.error(msg)
            raise ValueError(msg)
        else:
            self.distances = distances

        self.freq = self.stress[0].settings["freq"]
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = self.rfunc.get_init_parameters(self.name)
        # ensure lambda can't get too small
        # r/lambda <= 702 else get_tmax() will yield np.inf
        self.parameters.loc[self.name + "_lab", "pmin"] = \
            np.max(self.distances) / 702.
        # set initial lambda to largest distance
        self.parameters.loc[self.name + "_lab", "initial"] = \
            np.max(self.distances)

    def simulate(self, p=None, tmin=None, tmax=None, freq=None, dt=1,
                 istress=None):
        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)
        h = pd.Series(data=0, index=self.stress[0].series.index,
                      name=self.name)
        stresses = self.get_stress(istress=istress)
        distances = self.get_distances(istress=istress)
        for stress, r in zip(stresses, distances):
            npoints = stress.index.size
            p_with_r = np.concatenate([p, np.asarray([r])])
            b = self.get_block(p_with_r, dt, tmin, tmax)
            c = fftconvolve(stress, b, 'full')[:npoints]
            h = h.add(pd.Series(c, index=stress.index,
                                fastpath=True), fill_value=0.0)
        if istress is not None:
            if self.stress[istress].name is not None:
                h.name = self.stress[istress].name
        return h

    def get_stress(self, p=None, istress=None, **kwargs):
        if istress is None:
            return [s.series for s in self.stress]
        else:
            return [self.stress[istress].series]

    def get_distances(self, istress=None):
        if istress is None:
            return self.distances
        else:
            return [self.distances[istress]]

    def get_parameters(self, model=None, istress=None):
        """ Get parameters including distance to observation point
        and return as array (dimensions (nstresses, 4))

        Parameters
        ----------
        model : pastas.Model, optional
            if not None (default), use optimal model parameters
        istress : int, optional
            if not None (default), return all parameters

        Returns
        -------
        p : np.array
            parameters for each stress as row of array, if istress is used
            returns only one row.

        """
        if model is None:
            p = self.parameters.initial.values
        else:
            p = model.get_parameters(self.name)

        distances = np.array(self.get_distances(istress=istress))
        if len(distances) > 1:
            p_with_r = np.concatenate([np.tile(p, (len(distances), 1)),
                                       distances[:, np.newaxis]], axis=1)
        else:
            p_with_r = np.r_[p, distances]
        return p_with_r

    def to_dict(self, series=True):
        """Method to export the WellModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the
            WellModel object.

        """
        data = {
            "stressmodel": self._name,
            "rfunc": self.rfunc._name,
            "name": self.name,
            "up": True if self.rfunc.up is 1 else False,
            "distances": self.distances,
            "cutoff": self.rfunc.cutoff,
            "stress": self.dump_stress(series),
            "sort_wells": self.sort_wells
        }
        return data


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
                                 up=True, meanstress=1, cutoff=0.999)
        self.value = 1.  # Initial value
        stress = TimeSeries(stress, settings=settings, metadata=metadata)
        self.stress = [stress]
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters.loc[self.name + "_f"] = (
            self.value, -np.inf, np.inf, True, self.name)

    def simulate(self, p=None, tmin=None, tmax=None, freq=None, dt=1):
        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)
        return self.stress[0].series * p[0]

    def to_dict(self, series=True):
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
    rfunc: pastas.rfunc instance, optional
        Response function used in the convolution with the stress. Default
        is Exponential.
    name: str, optional
        Name of the stress. Default is "recharge".
    recharge: string, optional
        String with the name of the recharge model. Options are: "Linear" (
        default).
    temp: pandas.Series or pastas.TimeSeries, optional
        pandas.Series or pastas.TimeSeries objects containing the
        temperature series. It depends on the recharge model is this
        argument is required or not.
    cutoff: float, optional
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
    computed by a method determined by the recharge input argument. In the
    second step this recharge flux is convoluted with a response function to
    obtain the contribution of recharge to the groundwater levels.

    """
    _name = "RechargeModel"

    def __init__(self, prec, evap, rfunc=Exponential, name="recharge",
                 recharge="Linear", temp=None, cutoff=0.999,
                 settings=("prec", "evap", "evap"),
                 metadata=(None, None, None)):
        # Store the precipitation and evaporation time series
        self.prec = TimeSeries(prec, settings=settings[0],
                               metadata=metadata[0])
        self.evap = TimeSeries(evap, settings=settings[1],
                               metadata=metadata[1])

        # Check if both series have a regular time step
        if self.prec.freq_original is None:
            msg = "Frequency of the precipitation series could not be " \
                  "determined. Please provide a time series with a regular " \
                  "time step."
            raise IndexError(msg)
        if self.evap.freq_original is None:
            msg = "Frequency of the evaporation series could not be " \
                  "determined. Please provide a time series with a regular " \
                  "time step."
            raise IndexError(msg)

        # Dynamically load the required recharge model from string
        recharge_mod = getattr(import_module("pastas.recharge"), recharge)
        self.recharge = recharge_mod()

        # Store a temperature time series if needed or set to None
        if self.recharge.temp is True:
            if temp is None:
                msg = "Recharge module {} requires a temperature series. " \
                      "No temperature series were provided".format(recharge)
                raise TypeError(msg)
            else:
                self.temp = TimeSeries(temp, settings=settings[2],
                                       metadata=metadata[2])
        else:
            self.temp = None

        # Select indices from validated stress where both series are available.
        index = self.prec.series.index.intersection(self.evap.series.index)
        if index.empty:
            msg = ('The stresses that were provided have no overlapping '
                   'time indices. Please make sure the indices of the time '
                   'series overlap.')
            logger.error(msg)
            raise Exception(msg)

        # Calculate initial recharge estimation for initial rfunc parameters
        p = self.recharge.get_init_parameters().initial.values
        meanstress = self.get_stress(p=p, tmin=index.min(), tmax=index.max(),
                                     freq=self.prec.settings["freq"]).std()

        StressModelBase.__init__(self, rfunc=rfunc, name=name,
                                 tmin=index.min(), tmax=index.max(),
                                 meanstress=meanstress, cutoff=cutoff,
                                 up=True)

        self.stress = [self.prec, self.evap]
        if self.temp:
            self.stress.append(self.temp)
        self.freq = self.prec.settings["freq"]
        self.set_init_parameters()

        self.nsplit = 1

    def set_init_parameters(self):
        self.parameters = pd.concat(
            [self.rfunc.get_init_parameters(self.name),
             self.recharge.get_init_parameters(self.name)
             ])

    def update_stress(self, **kwargs):
        """Method to update the settings of the individual TimeSeries.

        Notes
        -----
        For the individual options for the different settings please refer to
        the docstring from the TimeSeries.update_series() method.

        See Also
        --------
        ps.TimeSeries.update_series

        """
        self.prec.update_series(**kwargs)
        self.evap.update_series(**kwargs)
        if self.temp is not None:
            self.temp.update_series(**kwargs)

        if "freq" in kwargs:
            self.freq = kwargs["freq"]

    def simulate(self, p=None, tmin=None, tmax=None, freq=None, dt=1):
        """Method to simulate the contribution of the groundwater
        recharge to the head.

        Parameters
        ----------
        p: array of floats
        tmin: string, optional
        tmax: string, optional
        freq: string, optional
        dt: float, optional
            Time step to use in the recharge calculation.

        Returns
        -------

        """
        if p is None:
            p = self.parameters.initial.values
        b = self.get_block(p[:-self.recharge.nparam], dt, tmin, tmax)
        stress = self.get_stress(p=p, tmin=tmin, tmax=tmax, freq=freq).values
        return pd.Series(data=fftconvolve(stress, b, 'full')[:stress.size],
                         index=self.prec.series.index, name=self.name,
                         fastpath=True)

    def get_stress(self, p=None, tmin=None, tmax=None, freq=None,
                   istress=None, **kwargs):
        """Method to obtain the recharge stress calculated by the recharge
        model.

        Parameters
        ----------
        p: array, optional
            array with the parameters values. Must be the length self.nparam.
        istress: int, optional
            Return one of the stresses used for the recharge calculation.
            0 for precipitation, 1 for evaporation and 2 for temperature.
        tmin: string, optional
        tmax: string, optional
        freq: string, optional
        kwargs

        Returns
        -------
        stress: pandas.Series
            When no istress is selected, this return the estimated recharge
            flux that is convoluted with a response function on the
            "simulate" method.

        """
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax

        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)

        if istress is None:
            prec = self.prec.series.values
            evap = self.evap.series.values
            if self.temp is not None:
                temp = self.temp.series.values
            else:
                temp = None
            if p is None:
                p = self.parameters.initial.values

            stress = self.recharge.simulate(prec=prec, evap=evap, temp=temp,
                                            p=p[-self.recharge.nparam:])

            return pd.Series(data=stress, index=self.prec.series.index,
                             name="recharge", fastpath=True)
        elif istress == 0:
            return self.prec.series
        elif istress == 1:
            return self.evap.series
        else:
            return self.temp.series

    def to_dict(self, series=True):
        data = {
            "stressmodel": self._name,
            "prec": self.prec.to_dict(),
            "evap": self.evap.to_dict(),
            "rfunc": self.rfunc._name,
            "name": self.name,
            "recharge": self.recharge._name,
            "cutoff": self.rfunc.cutoff,
            "temp": self.temp.to_dict() if self.temp else None
        }
        return data
