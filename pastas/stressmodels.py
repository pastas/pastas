"""This module contains all the stress models that available in
Pastas. Stress models are used to translate an input time series into a
contribution that explains (part of) the output series.

Supported Stress models
-----------------------
The following stressmodels are currently supported and tested:

.. autosummary::
    :nosignatures:
    :toctree: ./generated

    StressModel
    StressModel2
    RechargeModel
    FactorModel
    StepModel
    WellModel
    TarsoModel

Examples
--------

>>> sm = ps.StressModel(stress, rfunc=ps.Gamma, name="sm1")
>>> ml.add_stressmodel(stressmodel=sm)

See Also
--------
pastas.model.Model.add_stressmodel

Warnings
--------
All other stressmodels are for research purposes only and are not (yet)
fully supported and tested.

"""

from logging import getLogger

import numpy as np
from pandas import date_range, Series, Timedelta, DataFrame, concat, Timestamp
from scipy.signal import fftconvolve

from .decorators import set_parameter, njit
from .recharge import Linear
from .rfunc import One, Exponential, HantushWellModel
from .timeseries import TimeSeries
from .utils import validate_name

logger = getLogger(__name__)

__all__ = ["StressModel", "StressModel2", "Constant", "StepModel",
           "LinearTrend", "FactorModel", "RechargeModel", "WellModel"]


class StressModelBase:
    """StressModel Base class called by each StressModel object.

    Attributes
    ----------
    name: str
        Name of this stressmodel object. Used as prefix for the parameters.
    parameters: pandas.DataFrame
        Dataframe containing the parameters.

    """
    _name = "StressModelBase"

    def __init__(self, name, tmin, tmax, rfunc=None):
        self.name = validate_name(name)
        self.tmin = tmin
        self.tmax = tmax
        self.freq = None

        self.rfunc = rfunc
        self.parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])

        self.stress = []

    @property
    def nparam(self):
        return self.parameters.index.size

    def set_init_parameters(self):
        """Set the initial parameters (back) to their default values."""

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
        ps.timeseries.TimeSeries.update_series

        """
        for stress in self.stress:
            stress.update_series(**kwargs)

        if "freq" in kwargs:
            self.freq = kwargs["freq"]

    def dump_stress(self, series=True):
        """Method to dump all stresses in the stresses list.

        Parameters
        ----------
        series: bool, optional
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
        stress: pandas.Dataframe
            Pandas dataframe of the stress(es)

        """
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax

        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)

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
        """Internal method to get the block-response function"""
        if tmin is not None and tmax is not None:
            day = Timedelta(1, 'D')
            maxtmax = (Timestamp(tmax) - Timestamp(tmin)) / day
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
    up: bool or None, optional
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
    >>> sm = ps.StressModel(stress=pd.Series(), rfunc=ps.Gamma, name="Prec",
    >>>                     settings="prec")

    See Also
    --------
    pastas.rfunc
    pastas.timeseries.TimeSeries

    """
    _name = "StressModel"

    def __init__(self, stress, rfunc, name, up=True, cutoff=0.999,
                 settings=None, metadata=None, meanstress=None):
        if isinstance(stress, list):
            stress = stress[0]  # TODO Temporary fix Raoul, 2017-10-24

        stress = TimeSeries(stress, settings=settings, metadata=metadata)

        if meanstress is None:
            meanstress = stress.series.std()

        rfunc = rfunc(up=up, cutoff=cutoff, meanstress=meanstress)

        StressModelBase.__init__(self, name=name,
                                 tmin=stress.series.index.min(),
                                 tmax=stress.series.index.max(), rfunc=rfunc)
        self.freq = stress.settings["freq"]
        self.stress = [stress]
        self.set_init_parameters()

    def set_init_parameters(self):
        """Set the initial parameters (back) to their default values.

        """
        self.parameters = self.rfunc.get_init_parameters(self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1.0):
        """Simulates the head contribution.

        Parameters
        ----------
        p: numpy.ndarray
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
        h = Series(data=fftconvolve(stress, b, 'full')[:npoints],
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
    stress: list of pandas.Series or list of pastas.timeseries
        list of two pandas.Series or pastas.timeseries objects containing the
        stresses. Usually the first is the precipitation and the second the
        evaporation.
    rfunc: pastas.rfunc instance
        Response function used in the convolution with the stress.
    name: str
        Name of the stress
    up: bool or None, optional
        True if response function is positive (default), False if negative.
        None if you don't want to define if response is positive or negative.
    cutoff: float, optional
        float between 0 and 1 to determine how long the response is (default
        is 99% of the actual response time). Used to reduce computation times.
    settings: Tuple with two dicts, optional
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
    pastas.timeseries

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

        rfunc = rfunc(up=up, cutoff=cutoff, meanstress=meanstress)

        StressModelBase.__init__(self, name=name, tmin=index.min(),
                                 tmax=index.max(), rfunc=rfunc)
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
        p: numpy.ndarray
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
        b = self.get_block(p[:-1], dt, tmin, tmax)
        stress = self.get_stress(p=p, tmin=tmin, tmax=tmax, freq=freq,
                                 istress=istress)
        if istress == 1:
            stress = p[-1] * stress
        npoints = stress.index.size
        h = Series(data=fftconvolve(stress, b, 'full')[:npoints],
                   index=stress.index, name=self.name, fastpath=True)
        if istress is not None:
            if self.stress[istress].name is not None:
                h.name = h.name + ' (' + self.stress[istress].name + ')'
        return h

    def get_stress(self, p=None, tmin=None, tmax=None, freq=None,
                   istress=None, **kwargs):
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax

        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)

        if istress is None:
            if p is None:
                p = self.parameters.initial.values
            return self.stress[0].series.add(p[-1] * self.stress[1].series)
        elif istress == 0:
            return self.stress[0].series
        else:
            return self.stress[1].series

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
    tstart: str or Timestamp
        String with the start date of the step, e.g. '2018-01-01'. This
        value is fixed by default. Use ml.set_vary("step_tstart", 1) to vary
        the start time of the step trend.
    name: str
        String with the name of the stressmodel.
    rfunc: pastas.rfunc.RfuncBase, optional
        Pastas response function used to simulate the effect of the step.
        Default is rfunc.One, an instant effect.
    up: bool, optional
        Force a direction of the step. Default is None.

    Notes
    -----
    This step trend is calculated as follows. First, a binary series is
    created, with zero values before tstart, and ones after the start. This
    series is convoluted with the block response to simulate a step trend.

    """
    _name = "StepModel"

    def __init__(self, tstart, name, rfunc=One, up=True, cutoff=0.999):
        rfunc = rfunc(up=up, cutoff=cutoff, meanstress=1.0)

        StressModelBase.__init__(self, name=name, tmin=Timestamp.min,
                                 tmax=Timestamp.max, rfunc=rfunc)
        self.tstart = Timestamp(tstart)
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = self.rfunc.get_init_parameters(self.name)
        tmin = Timestamp.min.toordinal()
        tmax = Timestamp.max.toordinal()
        tinit = self.tstart.toordinal()

        self.parameters.loc[self.name + "_tstart"] = (tinit, tmin, tmax,
                                                      False, self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1):
        tstart = Timestamp.fromordinal(int(p[-1]), freq="D")
        tindex = date_range(tmin, tmax, freq=freq)
        h = Series(0, tindex, name=self.name)
        h.loc[h.index > tstart] = 1

        b = self.get_block(p[:-1], dt, tmin, tmax)
        npoints = h.index.size
        h = Series(data=fftconvolve(h, b, 'full')[:npoints],
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

    start: str
        String with a date to start the trend, will be transformed to an
        ordinal number internally. E.g. "2018-01-01"
    end: str
        String with a date to end the trend, will be transformed to an ordinal
        number internally. E.g. "2018-01-01"
    name: str, optional
        String with the name of the stressmodel

    """
    _name = "LinearTrend"

    def __init__(self, start, end, name="linear_trend"):
        StressModelBase.__init__(self, name=name, tmin=Timestamp.min,
                                 tmax=Timestamp.max)
        self.start = start
        self.end = end
        self.set_init_parameters()

    def set_init_parameters(self):
        start = Timestamp(self.start).toordinal()
        end = Timestamp(self.end).toordinal()
        tmin = Timestamp.min.toordinal()
        tmax = Timestamp.max.toordinal()

        self.parameters.loc[self.name + "_a"] = (
            0, -np.inf, np.inf, True, self.name)
        self.parameters.loc[self.name + "_tstart"] = (
            start, tmin, tmax, True, self.name)
        self.parameters.loc[self.name + "_tend"] = (
            end, tmin, tmax, True, self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1):
        tindex = date_range(tmin, tmax, freq=freq)

        if p[1] < tindex[0].toordinal():
            tmin = tindex[0]
        else:
            tmin = Timestamp.fromordinal(int(p[1]))

        if p[2] >= tindex[-1].toordinal():
            tmax = tindex[-1]
        else:
            tmax = Timestamp.fromordinal(int(p[2]))

        trend = tindex.to_series().diff() / Timedelta(1, "D")
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
    name: str, optional
        Name of the stressmodel
    initial: float, optional
        Initial estimate of the parameter value. E.g. The minimum of the
        observed series.

    """
    _name = "Constant"

    def __init__(self, name="constant", initial=0.0):
        StressModelBase.__init__(self, name=name, tmin=Timestamp.min,
                                 tmax=Timestamp.max)
        self.initial = initial
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters.loc[self.name + '_d'] = (
            self.initial, np.nan, np.nan, True, self.name)

    @staticmethod
    def simulate(p=None):
        return p


class WellModel(StressModelBase):
    """
    Convolution of one or more stresses with one response function.

    Parameters
    ----------
    stress: list
        list containing the stresses timeseries.
    rfunc: pastas.rfunc
        this model only works with the HantushWellModel response function.
    name: str
        Name of the stressmodel.
    distances: list or list-like
        list of distances to oseries, must be ordered the same as the
        stresses.
    up: bool, optional
        whether a positive stress has an increasing or decreasing effect on
        the model, by default False, in which case positive stress lowers
        e.g., the groundwater level.
    cutoff: float, optional
        percentage at which to cutoff the step response, by default 0.999.
    settings: str, list of dict, optional
        settings of the timeseries, by default "well".
    sort_wells: bool, optional
        sort wells from closest to furthest, by default True.

    Notes
    -----
    This class implements convolution of multiple series with a the same
    response function. This can be applied when dealing with multiple
    wells in a time series model. The distance from an influence to the
    location of the oseries has to be provided for each stress.

    Warnings
    --------
    This model only works with the HantushWellModel response function.

    """
    _name = "WellModel"

    def __init__(self, stress, rfunc, name, distances, up=False, cutoff=0.999,
                 settings="well", sort_wells=True):
        if not issubclass(rfunc, HantushWellModel):
            raise NotImplementedError("WellModel only supports the rfunc "
                                      "HantushWellModel fow now!")

        # sort wells by distance
        self.sort_wells = sort_wells
        if self.sort_wells:
            stress = [s for _, s in sorted(zip(distances, stress),
                                           key=lambda pair: pair[0])]
            if isinstance(settings, list):
                settings = [s for _, s in sorted(zip(distances, settings),
                                                 key=lambda pair: pair[0])]
            distances.sort()

        if settings is None or isinstance(settings, str):
            settings = len(stress) * [None]

        # convert stresses to TimeSeries if necessary
        stress = self.handle_stress(stress, settings)

        # Check if number of stresses and distances match
        if len(stress) != len(distances):
            msg = "The number of stresses does not match the number" \
                  "of distances provided."
            logger.error(msg)
            raise ValueError(msg)
        else:
            self.distances = distances

        meanstress = np.max([s.series.std() for s in stress])
        rfunc = rfunc(up=up, cutoff=cutoff, meanstress=meanstress)

        StressModelBase.__init__(self, name=name, tmin=Timestamp.min,
                                 tmax=Timestamp.max, rfunc=rfunc)

        self.stress = stress
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
        stresses = self.get_stress(tmin=tmin, tmax=tmax, freq=freq,
                                   istress=istress)
        distances = self.get_distances(istress=istress)
        h = Series(data=0, index=self.stress[0].series.index, name=self.name)
        for stress, r in zip(stresses, distances):
            npoints = stress.index.size
            p_with_r = np.concatenate([p, np.asarray([r])])
            b = self.get_block(p_with_r, dt, tmin, tmax)
            c = fftconvolve(stress, b, 'full')[:npoints]
            h = h.add(Series(c, index=stress.index, fastpath=True),
                      fill_value=0.0)
        if istress is not None:
            if self.stress[istress].name is not None:
                h.name = self.stress[istress].name
            else:
                h.name = self.name + "_" + str(istress)
        else:
            h.name = self.name
        return h

    def handle_stress(self, stress, settings):
        """Internal method to handle user provided stress in init.

        Parameters
        ----------
        stress: pandas.Series, pastas.TimeSeries, list or dict
            stress or collection of stresses
        settings: dict or iterable
            settings dictionary

        Returns
        -------
        stress: list
            return a list with the stresses transformed to pastas TimeSeries.

        """
        data = []

        if isinstance(stress, Series):
            data.append(TimeSeries(stress, settings))
        elif isinstance(stress, dict):
            for i, value in enumerate(stress.values()):
                data.append(TimeSeries(value, settings=settings[i]))
        elif isinstance(stress, list):
            for i, value in enumerate(stress):
                data.append(TimeSeries(value, settings=settings[i]))
        else:
            logger.error("Stress format is unknown. Provide a"
                         "Series, dict or list.")
        return data

    def get_stress(self, p=None, tmin=None, tmax=None, freq=None,
                   istress=None, **kwargs):
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax

        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)

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
        if distances.size > 1:
            p_with_r = np.concatenate([np.tile(p, (distances.size, 1)),
                                       distances[:, np.newaxis]], axis=1)
        else:
            p_with_r = np.r_[p, distances]
        return p_with_r

    def to_dict(self, series=True):
        """Internal method to export the WellModel object.

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
            "up": True if self.rfunc.up else False,
            "distances": self.distances,
            "cutoff": self.rfunc.cutoff,
            "stress": self.dump_stress(series),
            "sort_wells": self.sort_wells
        }
        return data


class FactorModel(StressModelBase):
    """Model that multiplies a stress by a single value.

    Parameters
    ----------
    stress: pandas.Series or pastas.timeseries.TimeSeries
        Stress which will be multiplied by a factor. The stress does not
        have to be equidistant.
    name: str, optional
        String with the name of the stressmodel.
    settings: dict or str, optional
        Dict or String that is forwarded to the TimeSeries object created
        from the stress.
    metadata: dict, optional
        Dictionary with metadata, forwarded to the TimeSeries object created
        from the stress.

    """
    _name = "FactorModel"

    def __init__(self, stress, name="factor", settings=None, metadata=None):
        if isinstance(stress, list):
            stress = stress[0]  # Temporary fix Raoul, 2017-10-24

        stress = TimeSeries(stress, settings=settings, metadata=metadata)

        tmin = stress.series_original.index.min()
        tmax = stress.series_original.index.max()

        StressModelBase.__init__(self, name=name, tmin=tmin, tmax=tmax)
        self.value = 1.  # Initial value
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
    prec: pandas.Series or pastas.timeseries.TimeSeries
        pandas.Series or pastas.timeseries object containing the
        precipitation series.
    evap: pandas.Series or pastas.timeseries.TimeSeries
        pandas.Series or pastas.timeseries object containing the
        evaporation series.
    rfunc: pastas.rfunc class, optional
        Response function used in the convolution with the stress. Default
        is Exponential.
    name: str, optional
        Name of the stress. Default is "recharge".
    recharge: pastas.recharge instance, optional
        String with the name of the recharge model. Options are: Linear (
        default), FlexModel and Berendrecht. These can be accessed through
        ps.rch.
    temp: pandas.Series or pastas.TimeSeries, optional
        pandas.Series or pastas.TimeSeries object containing the
        temperature series. It depends on the recharge model is this
        argument is required or not.
    cutoff: float, optional
        float between 0 and 1 to determine how long the response is (default)
        is 99.9% of the actual response time). Used to reduce computation
        times.
    settings: list of dicts or str, optional
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
    pastas.timeseries
    pastas.rch

    Notes
    -----
    This stress model computes the contribution of precipitation and
    potential evaporation in two steps. In the first step a recharge flux is
    computed by a model determined by the input argument `recharge`. In the
    second step this recharge flux is convoluted with a response function to
    obtain the contribution of recharge to the groundwater levels.

    Examples
    --------
    >>> sm = ps.RechargeModel(rain, evap, rfunc=ps.Exponential,
    >>>                       recharge=ps.rch.FlexModel(), name="rch")
    >>> ml.add_stressmodel(sm)

    Warning
    -------
    We recommend not to store a RechargeModel is a variable named `rm`. This
    name is already reserved in IPython to remove files and will cause
    problems later.

    """
    _name = "RechargeModel"

    def __init__(self, prec, evap, rfunc=Exponential, name="recharge",
                 recharge=Linear(), temp=None, cutoff=0.999,
                 settings=("prec", "evap"), metadata=(None, None)):
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

        # Store recharge object
        self.recharge = recharge

        # Store a temperature time series if needed or set to None
        if self.recharge.temp is True:
            if temp is None:
                msg = "Recharge module requires a temperature series. " \
                      "No temperature series were provided"
                raise TypeError(msg)
            elif len(settings) < 3 or len(metadata) < 3:
                msg = "Number of values for the settings and/or metadata is " \
                      "incorrect."
                raise TypeError(msg)
            else:
                self.temp = TimeSeries(temp, settings=settings[2],
                                       metadata=metadata[2])
        else:
            self.temp = None

        # Select indices from validated stress where both series are available.
        index = self.prec.series.index.intersection(self.evap.series.index)
        if index.empty:
            msg = ("The stresses that were provided have no overlapping"
                   "time indices. Please make sure the indices of the time"
                   "series overlap.")
            logger.error(msg)
            raise Exception(msg)

        # Calculate initial recharge estimation for initial rfunc parameters
        p = self.recharge.get_init_parameters().initial.values
        meanstress = self.get_stress(p=p, tmin=index.min(), tmax=index.max(),
                                     freq=self.prec.settings["freq"]).std()

        rfunc = rfunc(up=True, cutoff=cutoff, meanstress=meanstress)

        StressModelBase.__init__(self, name=name, tmin=index.min(),
                                 tmax=index.max(), rfunc=rfunc)

        self.stress = [self.prec, self.evap]
        if self.temp:
            self.stress.append(self.temp)
        self.freq = self.prec.settings["freq"]
        self.set_init_parameters()
        if isinstance(self.recharge, Linear):
            self.nsplit = 2
        else:
            self.nsplit = 1

    def set_init_parameters(self):
        """Internal method to set the initial parameters."""
        self.parameters = concat(
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
        ps.timeseries.TimeSeries.update_series

        """
        self.prec.update_series(**kwargs)
        self.evap.update_series(**kwargs)
        if self.temp is not None:
            self.temp.update_series(**kwargs)

        if "freq" in kwargs:
            self.freq = kwargs["freq"]

    def simulate(self, p=None, tmin=None, tmax=None, freq=None, dt=1.0,
                 istress=None):
        """Method to simulate the contribution of recharge to the head.

        Parameters
        ----------
        p: numpy.ndarray, optional
            parameter used for the simulation
        tmin: string, optional
        tmax: string, optional
        freq: string, optional
        dt: float, optional
            Time step to use in the recharge calculation.
        istress: int, optional
            This only works for the Linear model!

        Returns
        -------
        pandas.Series

        """
        if p is None:
            p = self.parameters.initial.values
        b = self.get_block(p[:-self.recharge.nparam], dt, tmin, tmax)
        stress = self.get_stress(p=p, tmin=tmin, tmax=tmax, freq=freq,
                                 istress=istress).values
        name = self.name

        if istress is not None:
            if istress is 1 and self.nsplit > 1:
                # only happen when Linear is used as the recharge model
                stress = stress * p[-1]
            if self.stress[istress].name is not None:
                name = "{} ({})".format(self.name, self.stress[istress].name)

        return Series(data=fftconvolve(stress, b, 'full')[:stress.size],
                      index=self.prec.series.index, name=name, fastpath=True)

    def get_stress(self, p=None, tmin=None, tmax=None, freq=None,
                   istress=None, **kwargs):
        """Method to obtain the recharge stress calculated by the recharge
        model.

        Parameters
        ----------
        p: array, optional
            array with the parameters values. Must be the length self.nparam.
        tmin: string, optional
        tmax: string, optional
        freq: string, optional
        istress: int, optional
            Return one of the stresses used for the recharge calculation.
            0 for precipitation, 1 for evaporation and 2 for temperature.
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

            return Series(data=stress, index=self.prec.series.index,
                          name="recharge", fastpath=True)
        elif istress == 0:
            return self.prec.series
        elif istress == 1:
            return self.evap.series
        else:
            return self.temp.series

    def get_water_balance(self, p=None, tmin=None, tmax=None, freq=None):
        """
        Internal method to obtain the water balance components.

        Parameters
        ----------
        p: array, optional
            array with the parameters values.
        tmin: string, optional
        tmax: string, optional
        freq: string, optional

        Returns
        -------
        wb: pandas.DataFrame
            Dataframe with the water balance components, both fluxes and
            states.

        Notes
        -----
        This method return a data frame with all water balance components,
        fluxes and states. All ingoing fluxes have a positive sign (e.g.,
        precipitation) and all outgoing fluxes have negative sign (e.g.,
        recharge).

        Warning
        -------
        This is an experimental method and may change in the future.

        Examples
        --------
        >>> sm = ps.RechargeModel(prec, evap, ps.Gamma, ps.rch.FlexModel(),
        >>>                       name="rch")
        >>> ml.add_stressmodel(sm)
        >>> ml.solve()
        >>> wb = sm.get_water_balance(ml.get_parameters("rch"))
        >>> wb.plot(subplots=True)

        """
        if p is None:
            p = self.parameters.initial.values

        prec = self.get_stress(tmin=tmin, tmax=tmax, freq=freq,
                               istress=0).values
        evap = self.get_stress(tmin=tmin, tmax=tmax, freq=freq,
                               istress=1).values

        df = self.recharge.get_water_balance(prec=prec, evap=evap, temp=None,
                                             p=p[-self.recharge.nparam:])
        df.index = self.prec.series.index
        return df

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


class TarsoModel(RechargeModel):
    """
    Stressmodel simulating the effect of recharge using the Tarso method.

    Parameters
    ----------
    oseries: pandas.Series or pastas.TimeSeries, optional
        A series of observations on which the model will be calibrated. It is
        used to determine the initial values of the drainage levels and the
        boundaries of the upper drainage level. Specify either oseries or dmin
        and dmax.
    dmin: float, optional
        The minimum drainage level. It is used to determine the initial values
        of the drainage levels and the lower boundary of the upper drainage
        level. Specify either oseries or dmin and dmax.
    dmax : float, optional
        The maximum drainage level. It is used to determine the initial values
        of the drainage levels and the upper boundary of the upper drainage
        level. Specify either oseries or dmin and dmax.
    rfunc: pastas.rfunc
        this model only works with the Exponential response function.

    See Also
    --------
    pastas.recharge

    Notes
    -----
    The Threshold autoregressive self-exciting open-loop (Tarso) model
    [knotters_1999]_ is nonlinear in structure because it incorporates two
    regimes which are separated by a threshold. This model method can be
    used to simulate a groundwater system where the groundwater head reaches
    the surface or drainage level in wet conditions. TarsoModel uses two
    drainage levels, with two exponential response functions. When the
    simulation reaches the second drainage level, the second response
    function becomes active. Because of its structure, TarsoModel cannot be
    combined with other stress models, a constant or a transform.
    TarsoModel inherits from RechargeModel. Only parameters specific to the
    child class are named above.

    References
    ----------
    .. [knotters_1999] Knotters, M. & De Gooijer, Jan G.. (1999). TARSO
       modeling of water table depths. Water Resources Research. 35.
       10.1029/1998WR900049.

    """
    _name = "TarsoModel"

    def __init__(self, prec, evap, oseries=None, dmin=None, dmax=None,
                 rfunc=Exponential, **kwargs):
        if oseries is not None:
            if dmin is not None or dmax is not None:
                msg = 'Please specify either oseries or dmin and dmax'
                raise (Exception(msg))
            o = TimeSeries(oseries).series
            dmin = o.min()
            dmax = o.max()
        elif dmin is None or dmax is None:
            msg = 'Please specify either oseries or dmin and dmax'
            raise (Exception(msg))
        if not issubclass(rfunc, Exponential):
            raise NotImplementedError("TarsoModel only supports rfunc "
                                      "Exponential!")
        self.dmin = dmin
        self.dmax = dmax
        super().__init__(prec=prec, evap=evap, rfunc=rfunc, **kwargs)

    def set_init_parameters(self):
        # parameters for the first drainage level
        p0 = self.rfunc.get_init_parameters(self.name)
        one = One(meanstress=self.dmin + 0.5 * (self.dmax - self.dmin))
        pd0 = one.get_init_parameters(self.name).squeeze()
        p0.loc['{}_d'.format(self.name)] = pd0
        p0.index = ['{}0'.format(x) for x in p0.index]

        # parameters for the second drainage level
        p1 = self.rfunc.get_init_parameters(self.name)
        initial = self.dmin + 0.75 * (self.dmax - self.dmin)
        pd1 = Series({'initial': initial, 'pmin': self.dmin, 'pmax': self.dmax,
                      'vary': True, 'name': self.name})
        p1.loc['{}_d'.format(self.name)] = pd1
        p1.index = ['{}1'.format(x) for x in p1.index]

        # parameters for the recharge-method
        pr = self.recharge.get_init_parameters(self.name)

        # combine all parameters
        self.parameters = concat([p0, p1, pr])

    def simulate(self, p=None, tmin=None, tmax=None, freq=None, dt=1):
        stress = self.get_stress(p=p, tmin=tmin, tmax=tmax, freq=freq)
        h = self.tarso(p[:-self.recharge.nparam], stress.values, dt)
        sim = Series(h, name=self.name, index=stress.index)
        return sim

    def to_dict(self, series=True):
        data = super().to_dict(series)
        data['dmin'] = self.dmin
        data['dmax'] = self.dmax
        return data

    @staticmethod
    def _check_stressmodel_compatibility(ml):
        """Internal method to check if no other stressmodels, a constants or a
        transform is used."""
        msg = "A TarsoModel cannot be combined with {}. Either remove the" \
              " TarsoModel or the {}."
        if len(ml.stressmodels) > 1:
            logger.warning(msg.format("other stressmodels", "stressmodels"))
        if ml.constant is not None:
            logger.warning(msg.format("a constant", "constant"))
        if ml.transform is not None:
            logger.warning(msg.format("a transform", "transform"))

    @staticmethod
    @njit
    def tarso(p, r, dt):
        """Calculates the head based on exponential decay of the previous
        timestep and recharge, using two thresholds."""
        A0, a0, d0, A1, a1, d1 = p

        # calculate physical meaning of these parameters
        S0 = a0 / A0
        c0 = A0

        S1 = a1 / A1
        c1 = A1

        # calculate effective parameters for the top level
        c_e = 1 / ((1 / c0) + (1 / c1))
        d_e = (c1 / (c0 + c1)) * d0 + (c0 / (c0 + c1)) * d1
        a_e = S1 * c_e

        h = np.full(len(r), np.NaN)
        for i in range(len(r)):
            if i == 0:
                h0 = (d0 + d1) / 2
                high = h0 > d1
                if high:
                    S, a, c, d = S1, a_e, c_e, d_e
                else:
                    S, a, c, d = S0, a0, c0, d0
            else:
                h0 = h[i - 1]
            exp_a = np.exp(-dt / a)
            h[i] = (h0 - d) * exp_a + r[i] * c * (1 - exp_a) + d
            newhigh = h[i] > d1
            if high != newhigh:
                # calculate time until d1 is reached
                dtdr = - S * c * np.log(
                    (d1 - d - r[i] * c) / (h0 - d - r[i] * c))
                if dtdr > dt:
                    raise (Exception())
                # change paraemeters
                high = newhigh
                if high:
                    S, a, c, d = S1, a_e, c_e, d_e
                else:
                    S, a, c, d = S0, a0, c0, d0
                # calculate new level after reaching d1
                exp_a = np.exp(-(dt - dtdr) / a)
                h[i] = (d1 - d) * exp_a + r[i] * c * (1 - exp_a) + d
        return h
