"""This module contains all the stress models available in Pastas.

Stress models are used to translate an input time series into contribution that
explains (part of) the output series.

Examples
--------

>>> sm = ps.StressModel(stress, rfunc=ps.Gamma(), name="sm1")
>>> ml.add_stressmodel(stressmodel=sm)

See Also
--------
pastas.model.Model.add_stressmodel
"""

from inspect import isclass
from logging import getLogger

# Type Hinting
from typing import List, Optional, Tuple, Union

import numpy as np
from packaging.version import parse as parse_version
from pandas import DataFrame, Series, Timedelta, Timestamp, concat, date_range
from pandas import __version__ as pd_version
from scipy.signal import fftconvolve

from pastas.typing import (
    ArrayLike,
    Model,
    Recharge,
    RFunc,
    StressSettingsDict,
    TimestampType,
)

from .decorators import njit, set_parameter
from .recharge import Linear
from .rfunc import Exponential, HantushWellModel, One
from .timeseries import TimeSeries
from .utils import validate_name

pandas_version = parse_version(pd_version)

logger = getLogger(__name__)

__all__ = [
    "StressModel",
    "Constant",
    "StepModel",
    "LinearTrend",
    "RechargeModel",
    "WellModel",
    "TarsoModel",
    "ChangeModel",
]


class StressModelBase:
    """StressModel Base class called by each StressModel object.

    Attributes
    ----------
    name: str
        Name of this stressmodel object. Used as prefix for the parameters.
    parameters: pandas.DataFrame
        The DataFrame containing the parameters.
    """

    _name = "StressModelBase"

    def __init__(
        self,
        name: str,
        tmin: TimestampType,
        tmax: TimestampType,
        rfunc: Optional[RFunc] = None,
        up: bool = True,
        gain_scale_factor: float = 1.0,
    ) -> None:
        self.name = validate_name(name)
        self.tmin = tmin
        self.tmax = tmax
        self.freq = None

        if rfunc is not None:
            if isclass(rfunc):
                raise TypeError(
                    "the rfunc argument must be an instance of response function, not "
                    "a class. Please provide an instance, e.g., ps.Exponential()"
                )
            rfunc.update_rfunc_settings(up=up, gain_scale_factor=gain_scale_factor)
        self.rfunc = rfunc

        self.parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )

        self.stress = []

    @property
    def nparam(self) -> Tuple[int]:
        return self.parameters.index.size

    def set_init_parameters(self) -> None:
        """Set the initial parameters (back) to their default values."""

    @set_parameter
    def _set_initial(self, name: str, value: float) -> None:
        """Internal method to set the initial parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "initial"] = value

    @set_parameter
    def _set_pmin(self, name: str, value: float) -> None:
        """Internal method to set the lower bound of the parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "pmin"] = value

    @set_parameter
    def _set_pmax(self, name: str, value: float) -> None:
        """Internal method to set the upper bound of the parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "pmax"] = value

    @set_parameter
    def _set_vary(self, name: str, value: float) -> None:
        """Internal method to set if the parameter is varied during optimization.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "vary"] = bool(value)

    @set_parameter
    def _set_dist(self, name: str, value: str) -> None:
        """Internal method to set distribution of prior of the parameter.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "dist"] = str(value)

    def update_stress(
        self,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
    ) -> None:
        """Method to update the settings of the all stresses in the stress model.

        Parameters
        ----------
        freq: str, optional
            String representing the desired frequency of the time series. Must be one
            of the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        tmin: str or pandas.Timestamp, optional
            String that can be converted to, or a Pandas Timestamp with the minimum
            time of the series.
        tmax: str or pandas.Timestamp, optional
            String that can be converted to, or a Pandas Timestamp with the maximum
            time of the series.

        Notes
        -----
        For the individual options for the different settings please refer to the
        docstring from the TimeSeries.update_series() method.

        See Also
        --------
        ps.timeseries.TimeSeries.update_series
        """
        for stress in self.stress:
            stress.update_series(freq=freq, tmin=tmin, tmax=tmax)

        if freq:
            self.freq = freq

    def get_stress(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        istress: Optional[int] = None,
        **kwargs,
    ) -> DataFrame:
        """Returns the stress(es) of the time series object as a pandas DataFrame.

        If the time series object has multiple stresses each column represents a stress.

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

    def to_dict(self, **kwargs):
        """Method to export the stress model object."""

    def get_nsplit(self) -> int:
        """Determine in how many time series the contribution can be split."""
        if hasattr(self, "nsplit"):
            return self.nsplit
        else:
            return len(self.stress)

    def _get_block(
        self, p: ArrayLike, dt: float, tmin: TimestampType, tmax: TimestampType
    ) -> ArrayLike:
        """Internal method to get the block-response function."""
        if tmin is not None and tmax is not None:
            day = Timedelta(1, "D")
            maxtmax = (Timestamp(tmax) - Timestamp(tmin)) / day
        else:
            maxtmax = None
        b = self.rfunc.block(p, dt, maxtmax=maxtmax)
        return b

    def get_settings(self) -> dict:
        """Method to obtain the settings of the stresses.

        Returns
        -------
        settings: dict

        Notes
        -----
        To update the settings of the time series, use the `update_stress` method.
        """
        if len(self.stress) == 0:
            settings = None
        else:
            settings = {stress.name: stress.settings for stress in self.stress}
        return settings

    def get_parameters(self, model=None) -> ArrayLike:
        """Get parameters and return as array.

        Parameters
        ----------
        model : pastas.Model, optional
            If provided, and the model is solved, return optimal model parameter-values.
            Otherwise, return initial parameter-values.
        istress : int, optional
            if provided, return specific parameter set, else return all parameters.

        Returns
        -------
        p : array_like
            An array of the parameters of the stressmodel.
        """
        if model is None:
            p = self.parameters.initial.values
        else:
            p = model.get_parameters(self.name)
        return p


class StressModel(StressModelBase):
    """Stress model convoluting a stress with a response function.

    Parameters
    ----------
    stress: pandas.Series
        pandas.Series with pandas.DatetimeIndex containing the stress.
    rfunc: pastas.rfunc instance
        An instance of the response function used in the convolution with the stress.
    name: str
        Name of the stress.
    up: bool or None, optional
        True if response function is positive (default), False if negative. None if
        you don't want to define if response is positive or negative.
    settings: dict or str, optional
        The settings of the stress. This can be a string referring to a predefined
        settings dictionary (defined in ps.rcParams["timeseries"]), or a dictionary with
        the settings to apply. For more information refer to Time series settings
        section below.
    metadata: dict, optional
        dictionary containing metadata about the stress. This is passed onto the
        TimeSeries object.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.

    Time series settings
    --------------------
    fill_nan : {"drop", "mean", "interpolate"} or float
        Method for filling NaNs.
           * `drop`: drop NaNs from time series
           * `mean`: fill NaNs with mean value of time series
           * `interpolate`: fill NaNs by interpolating between finite values
           * `float`: fill NaN with provided value, e.g. 0.0
    fill_before : {"mean", "bfill"} or float
        Method for extending time series into past.
           * `mean`: extend time series into past with mean value of time series
           * `bfill`: extend time series into past by back-filling first value
           * `float`: extend time series into past with provided value, e.g. 0.0
    fill_after : {"mean", "ffill"} or float
        Method for extending time series into future.
           * `mean`: extend time series into future with mean value of time series
           * `ffill`: extend time series into future by forward-filling last value
           * `float`: extend time series into future with provided value, e.g. 0.0
    sample_up : {"mean", "interpolate", "divide"} or float
        Method for up-sampling time series (increasing frequency, e.g. going from weekly
        to daily values).
           * `bfill` or `backfill`: fill up-sampled time steps by back-filling current
             values
           * `ffill` or `pad`: fill up-sampled time steps by forward-filling current
             values
           * `mean`: fill up-sampled time steps with mean of timeseries
           * `interpolate`: fill up-sampled time steps by interpolating between current
             values
           * `divide`: fill up-sampled steps with current value divided by length of
             current time steps (i.e. spread value over new time steps).
    sample_down : {"mean", "drop", "sum", "min", "max"}
        Method for down-sampling time series (decreasing frequency, e.g. going from
        daily to weekly values).
           * `mean`: resample time series by taking the mean
           * `drop`: resample the time series by taking the mean, dropping any
             NaN-values
           * `sum`: resample time series by summing values
           * `max`: resample time series with maximum value
           * `min`: resample time series with minimum value

    Examples
    --------
    >>> import pastas as ps
    >>> import pandas as pd
    >>> sm = ps.StressModel(stress=pd.Series(), rfunc=ps.Gamma(), name="Prec",
    >>>                     settings="prec")

    See Also
    --------
    pastas.rfunc
    pastas.timeseries.TimeSeries
    """

    _name = "StressModel"

    def __init__(
        self,
        stress: Series,
        rfunc: RFunc,
        name: str,
        up: bool = True,
        settings: Optional[Union[str, StressSettingsDict]] = None,
        metadata: Optional[dict] = None,
        gain_scale_factor: Optional[float] = None,
    ) -> None:
        stress = TimeSeries(stress, settings=settings, metadata=metadata)

        StressModelBase.__init__(
            self,
            name=name,
            tmin=stress.series.index.min(),
            tmax=stress.series.index.max(),
            rfunc=rfunc,
            up=up,
            gain_scale_factor=(
                stress.series.std() if gain_scale_factor is None else gain_scale_factor
            ),
        )

        self.gain_scale_factor = gain_scale_factor
        self.freq = stress.settings["freq"]
        self.stress.append(stress)
        self.set_init_parameters()

    def set_init_parameters(self) -> None:
        """Set the initial parameters (back) to their default values."""
        self.parameters = self.rfunc.get_init_parameters(self.name)

    def simulate(
        self,
        p: ArrayLike,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        dt: float = 1.0,
    ) -> Series:
        """Simulates the head contribution.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the model
            parameters.
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
        b = self._get_block(p, dt, tmin, tmax)
        stress = self.stress[0].series
        npoints = stress.index.size
        h = Series(
            data=fftconvolve(stress, b, "full")[:npoints],
            index=stress.index,
            name=self.name,
        )
        return h

    def to_dict(self, series: bool = True) -> dict:
        """Method to export the StressModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the StressModel
            object.

        Notes
        -----
        Settings and metadata are exported with the stress.
        """
        data = {
            "class": self._name,
            "rfunc": self.rfunc.to_dict(),
            "name": self.name,
            "up": self.rfunc.up,
            "stress": self.stress[0].to_dict(series=series),
            "gain_scale_factor": self.gain_scale_factor,
        }
        return data


class StepModel(StressModelBase):
    """Stressmodel that simulates a step trend.

    Parameters
    ----------
    tstart: str or Timestamp
        String with the start date of the step, e.g. '2018-01-01'. This value is
        fixed by default. Use ml.set_parameter("step_tstart", vary=True) to vary the
        start time of the step trend.
    name: str
        String with the name of the stressmodel.
    rfunc: pastas.rfunc instance
        Pastas response function used to simulate the effect of the step. Default is
        ps.rfunc.One(), an instant effect.
    up: bool, optional
        Force a direction of the step. Default is None.

    Notes
    -----
    The step trend is calculated as follows. First, a binary series is created,
    with zero values before tstart, and ones after the start. This series is
    convolved with the block response to simulate a step trend.
    """

    _name = "StepModel"

    def __init__(
        self,
        tstart: TimestampType,
        name: str,
        rfunc: Optional[RFunc] = None,
        up: bool = None,
    ) -> None:
        if rfunc is None:
            rfunc = One()
        StressModelBase.__init__(
            self,
            name=name,
            tmin=Timestamp.min,
            tmax=Timestamp.max,
            rfunc=rfunc,
            up=up,
        )
        self.tstart = Timestamp(tstart)
        self.set_init_parameters()

    def set_init_parameters(self) -> None:
        self.parameters = self.rfunc.get_init_parameters(self.name)
        tmin = Timestamp.min.toordinal()
        tmax = Timestamp.max.toordinal()
        tinit = self.tstart.toordinal()

        self.parameters.loc[self.name + "_tstart"] = (
            tinit,
            tmin,
            tmax,
            False,
            self.name,
            "uniform",
        )

    def simulate(
        self,
        p: ArrayLike,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        dt: float = 1.0,
    ) -> Series:
        tstart = Timestamp.fromordinal(int(p[-1]))
        tindex = date_range(tmin, tmax, freq=freq)
        h = Series(0, tindex, name=self.name)
        h.loc[h.index > tstart] = 1

        b = self._get_block(p[:-1], dt, tmin, tmax)
        npoints = h.index.size
        h = Series(
            data=fftconvolve(h, b, "full")[:npoints],
            index=h.index,
            name=self.name,
        )
        return h

    def to_dict(self, **kwargs) -> dict:
        """Method to export the StepModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct object.
        """
        data = {
            "class": self._name,
            "tstart": self.tstart,
            "name": self.name,
            "rfunc": self.rfunc.to_dict(),
            "up": self.rfunc.up,
        }
        return data


class LinearTrend(StressModelBase):
    """Stressmodel that simulates a linear trend.

    Parameters
    ----------
    start: str
        String with a date to start the trend (e.g., "2018-01-01"), will be
        transformed to an ordinal number internally.
    end: str
        String with a date to end the trend (e.g., "2018-01-01"), will be transformed
        to an ordinal number internally.
    name: str, optional
        String with the name of the stress model.

    Notes
    -----
    While possible, it is not recommended to vary the parameters for the start and
    end time of the linear trend. These parameters are usually hard or even impossible
    to estimate from the data.
    """

    _name = "LinearTrend"

    def __init__(
        self, start: TimestampType, end: TimestampType, name: str = "trend"
    ) -> None:
        StressModelBase.__init__(
            self, name=name, tmin=Timestamp.min, tmax=Timestamp.max
        )
        self.start = start
        self.end = end
        self.set_init_parameters()

    def set_init_parameters(self) -> None:
        """Set the initial parameters for the stress model."""
        start = Timestamp(self.start).toordinal()
        end = Timestamp(self.end).toordinal()
        tmin = Timestamp.min.toordinal()
        tmax = Timestamp.max.toordinal()

        self.parameters.loc[self.name + "_a"] = (
            0.0,
            -np.inf,
            np.inf,
            True,
            self.name,
            "uniform",
        )
        self.parameters.loc[self.name + "_tstart"] = (
            start,
            tmin,
            tmax,
            False,
            self.name,
            "uniform",
        )
        self.parameters.loc[self.name + "_tend"] = (
            end,
            tmin,
            tmax,
            False,
            self.name,
            "uniform",
        )

    def simulate(
        self,
        p: ArrayLike,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        dt: float = 1.0,
    ) -> Series:
        """Simulate the trend."""
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
        return trend.rename(self.name)

    def to_dict(self, **kwargs) -> dict:
        """Method to export a dictionary to reconstruct the stressmodel.

        Parameters
        ----------
        kwargs

        Returns
        -------
        data: dict
        """
        data = {
            "class": self._name,
            "start": self.start,
            "end": self.end,
            "name": self.name,
        }
        return data


class Constant(StressModelBase):
    """A constant value that is added to the time series model.

    Parameters
    ----------
    name: str, optional
        Name of the stressmodel.
    initial: float, optional
        Initial estimate of the parameter value. For example, the minimum of the
        observed series.
    """

    _name = "Constant"

    def __init__(self, name: str = "constant", initial: float = 0.0) -> None:
        StressModelBase.__init__(
            self, name=name, tmin=Timestamp.min, tmax=Timestamp.max
        )
        self.initial = initial
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters.loc[self.name + "_d"] = (
            self.initial,
            np.nan,
            np.nan,
            True,
            self.name,
            "uniform",
        )

    @staticmethod
    def simulate(p: Optional[float] = None) -> float:
        return p

    def to_dict(self, **kwargs):
        """Method to export the StressModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the StressModel
            object.
        """
        data = {
            "class": self._name,
            "name": self.name,
            "initial": self.initial,
        }
        return data


class WellModel(StressModelBase):
    """Convolution of one or more stresses with a single scaled response function.

    Parameters
    ----------
    stress: list
        list containing the stresses time series.
    name: str
        name of the stressmodel.
    distances: array_like
        array_like of distances between the stresses (wells) and the oseries
        (monitoring well), must be in the same order as the stresses. This
        distance is used to scale the HantushWellModel response function for
        each stress.
    rfunc: pastas.rfunc instance, optional
        this model only works with the HantushWellModel response function, default is
        None which will initialize a HantushWellModel response function.
    up: bool, optional
        whether a positive stress has an increasing or decreasing effect on the model,
        by default False, in which case positive stress lowers e.g., the groundwater
        level.
    settings: str, list of dict, optional
        The settings of the stress. By default this is "well". This can be a string
        referring to a predefined settings dictionary (defined in
        ps.rcParams["timeseries"]), or a dictionary with the settings to apply. For more
        information, refer to Time series settings section below.
    sort_wells: bool, optional
        sort wells from closest to furthest, by default True.

    Time series settings
    --------------------
    fill_nan : {"drop", "mean", "interpolate"} or float
        Method for filling NaNs.
           * `drop`: drop NaNs from time series
           * `mean`: fill NaNs with mean value of time series
           * `interpolate`: fill NaNs by interpolating between finite values
           * `float`: fill NaN with provided value, e.g. 0.0
    fill_before : {"mean", "bfill"} or float
        Method for extending time series into past.
           * `mean`: extend time series into past with mean value of time series
           * `bfill`: extend time series into past by back-filling first value
           * `float`: extend time series into past with provided value, e.g. 0.0
    fill_after : {"mean", "ffill"} or float
        Method for extending time series into future.
           * `mean`: extend time series into future with mean value of time series
           * `ffill`: extend time series into future by forward-filling last value
           * `float`: extend time series into future with provided value, e.g. 0.0
    sample_up : {"mean", "interpolate", "divide"} or float
        Method for up-sampling time series (increasing frequency, e.g. going from weekly
        to daily values).
           * `bfill` or `backfill`: fill up-sampled time steps by back-filling current
             values
           * `ffill` or `pad`: fill up-sampled time steps by forward-filling current
             values
           * `mean`: fill up-sampled time steps with mean of timeseries
           * `interpolate`: fill up-sampled time steps by interpolating between current
             values
           * `divide`: fill up-sampled steps with current value divided by length of
             current time steps (i.e. spread value over new time steps).
    sample_down : {"mean", "drop", "sum", "min", "max"}
        Method for down-sampling time series (decreasing frequency, e.g. going from
        daily to weekly values).
           * `mean`: resample time series by taking the mean
           * `drop`: resample the time series by taking the mean, dropping any
             NaN-values
           * `sum`: resample time series by summing values
           * `max`: resample time series with maximum value
           * `min`: resample time series with minimum value

    Notes
    -----
    This class implements convolution of multiple series with the same response
    function. This can be applied when dealing with multiple wells in a time series
    model. The distance(s) from the pumping well(s) to the monitoring well have to be
    provided for each stress. See :cite:t:`brakenhoff_application_2022` for more details on the methods for this model.

    Only works with the HantushWellModel response function.
    """

    _name = "WellModel"

    def __init__(
        self,
        stress: List[Series],
        name: str,
        distances: ArrayLike,
        rfunc: Optional[RFunc] = None,
        up: bool = False,
        settings: Union[str, StressSettingsDict] = "well",
        sort_wells: bool = True,
        metadata: Optional[list] = None,
    ) -> None:
        # check response function
        if rfunc is None:
            rfunc = HantushWellModel()
        elif not isinstance(rfunc, HantushWellModel):
            raise NotImplementedError(
                "WellModel only supports the rfunc HantushWellModel!"
            )

        # check if number of stresses and distances match
        if len(stress) != len(distances):
            msg = (
                "The number of stresses does not match the number of distances "
                "provided."
            )
            logger.error(msg)
            raise ValueError(msg)
        else:
            self.distances = Series(
                index=[s.squeeze().name for s in stress],
                data=distances,
                name="distances",
            )

        # parse settings input
        if settings is None or isinstance(settings, str) or isinstance(settings, dict):
            settings = len(stress) * [settings]

        # if metadata is passed as dict -> convert to list
        if metadata is not None and isinstance(metadata, dict):
            metadata = [metadata]

        # parse stresses input
        stress = self._handle_stress(stress, settings, metadata)

        # sort wells by distance
        self.sort_wells = sort_wells
        if self.sort_wells:
            stress = [
                s for _, s in sorted(zip(distances, stress), key=lambda pair: pair[0])
            ]
            self.distances.sort_values(inplace=True)

        # estimate gain_scale_factor w/ max of stresses stdev
        gain_scale_factor = np.max([s.series.std() for s in stress])

        tmin = np.min([s.series.index.min() for s in stress])
        tmax = np.max([s.series.index.max() for s in stress])

        StressModelBase.__init__(
            self,
            name=name,
            tmin=tmin,
            tmax=tmax,
            rfunc=rfunc,
            up=up,
            gain_scale_factor=gain_scale_factor,
        )

        self.rfunc.set_distances(self.distances.values)

        self.stress = stress
        self.freq = self.stress[0].settings["freq"]
        self.set_init_parameters()

    def set_init_parameters(self) -> None:
        self.parameters = self.rfunc.get_init_parameters(self.name)

    def simulate(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        dt: float = 1.0,
        istress: Optional[int] = None,
        **kwargs,
    ) -> Series:
        distances = self.get_distances(istress=istress)
        stress_df = self.get_stress(
            p=p, tmin=tmin, tmax=tmax, freq=freq, istress=istress, squeeze=False
        )
        h = Series(data=0, index=self.stress[0].series.index, name=self.name)
        for name, r in distances.items():
            stress = stress_df.loc[:, name]
            npoints = stress.index.size
            p_with_r = np.concatenate([p, np.array([r])])
            b = self._get_block(p_with_r, dt, tmin, tmax)
            c = fftconvolve(stress, b, "full")[:npoints]
            h = h.add(Series(c, index=stress.index), fill_value=0.0)
        if istress is not None:
            if isinstance(istress, list):
                h.name = self.name + "_" + "+".join(str(i) for i in istress)
            elif self.stress[istress].name is not None:
                h.name = self.stress[istress].name
            else:
                h.name = self.name + "_" + str(istress)
        else:
            h.name = self.name
        return h

    @staticmethod
    def _handle_stress(stress, settings, metadata):
        """Internal method to handle user provided stress in init.

        Parameters
        ----------
        stress: pandas.Series, list or dict
            stress or collection of stresses.
        settings: dict or iterable
            settings dictionary.
        metadata : dict or list of dict
            metadata dictionaries corresponding to stress

        Returns
        -------
        stress: list
            return a list with the stresses transformed to pastas TimeSeries.
        """
        data = []

        if isinstance(stress, Series):
            data.append(TimeSeries(stress, settings=settings, metadata=metadata))
        elif isinstance(stress, dict):
            for i, (name, value) in enumerate(stress.items()):
                if metadata is not None:
                    imeta = metadata[i]
                else:
                    imeta = None
                data.append(
                    TimeSeries(value, name=name, settings=settings[i], metadata=imeta)
                )
        elif isinstance(stress, list):
            for i, value in enumerate(stress):
                if metadata is not None:
                    imeta = metadata[i]
                else:
                    imeta = None
                data.append(TimeSeries(value, settings=settings[i], metadata=imeta))
        else:
            msg = "Cannot parse 'stress' input. Provide a Series, dict or list."
            logger.error(msg)
            raise TypeError(msg)

        return data

    def get_stress(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        istress: Optional[int] = None,
        squeeze: bool = True,
        **kwargs,
    ) -> DataFrame:
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax

        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)

        if istress is None:
            df = DataFrame.from_dict({s.name: s.series for s in self.stress})
            if squeeze:
                return df.squeeze()
            else:
                return df
        elif isinstance(istress, list):
            return DataFrame.from_dict({s.name: s.series for s in self.stress}).iloc[
                :, istress
            ]
        else:
            if squeeze:
                return self.stress[istress].series
            else:
                return self.stress[istress].series.to_frame()

    def get_distances(self, istress: Optional[int] = None) -> DataFrame:
        if istress is None:
            return self.distances
        elif isinstance(istress, list):
            return self.distances.iloc[istress]
        else:
            return self.distances.iloc[istress : istress + 1]

    def get_parameters(self, model=None, istress: Optional[int] = None) -> ArrayLike:
        """Get parameters including distance to observation point and return as array
        (dimensions = (nstresses, 4)).

        Parameters
        ----------
        model : pastas.Model, optional
            If provided, and the model is solved, return optimal model parameter-values.
            Otherwise, return initial parameter-values.
        istress : int, optional
            if provided, return specific parameter set, else return all parameters.

        Returns
        -------
        p : array_like
            parameters for each stress as row of array, if istress is used returns
            only one row.
        """
        if model is None:
            p = self.parameters.initial.values
        else:
            p = model.get_parameters(self.name)

        distances = self.get_distances(istress=istress).values
        if distances.size > 1:
            p_with_r = np.concatenate(
                [np.tile(p, (distances.size, 1)), distances[:, np.newaxis]], axis=1
            )
        else:
            p_with_r = np.r_[p, distances]
        return p_with_r

    def dump_stress(self, series: bool = True) -> list:
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
            stress.name = validate_name(stress.name, raise_error=True)
            data.append(stress.to_dict(series=series))

        return data

    def to_dict(self, series: bool = True) -> dict:
        """Method to export the WellModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the WellModel
            object.
        """
        data = {
            "class": self._name,
            "stress": self.dump_stress(series),
            "rfunc": self.rfunc.to_dict(),
            "name": self.name,
            "distances": self.distances.to_list(),
            "up": True if self.rfunc.up else False,
            "sort_wells": self.sort_wells,
        }
        return data

    def variance_gain(
        self, model: Model, istress: Optional[int] = None, r: Optional[ArrayLike] = None
    ) -> float:
        """Calculate variance of the gain for WellModel.

        Variance of the gain is calculated based on propagation of uncertainty using
        optimal parameter values and the estimated variances of A and b and the
        covariance between A and b.

        Parameters
        ----------
        model : pastas.Model
            optimized model
        istress : int or list of int, optional
            index of stress(es) for which to calculate variance of gain
        r : array_like, optional
            radial distance(s) at which to calculate variance of the gain,
            only considered if istress is None

        Returns
        -------
        var_gain : float
            variance of the gain calculated from model results for parameters A and b.

        See Also
        --------
        pastas.HantushWellModel.variance_gain
        """
        if model.solver is None:
            raise AttributeError("Model not optimized! Run solve() first!")
        if self.rfunc._name != "HantushWellModel":
            raise ValueError("Response function must be HantushWellModel!")
        if model.solver.pcov.isna().all(axis=None):
            model.logger.warning("Covariance matrix contains only NaNs!")

        # get parameters and (co)variances
        A = model.parameters.at[self.name + "_A", "optimal"]
        b = model.parameters.at[self.name + "_b", "optimal"]
        var_A = model.solver.pcov.at[self.name + "_A", self.name + "_A"]
        var_b = model.solver.pcov.at[self.name + "_b", self.name + "_b"]
        cov_Ab = model.solver.pcov.at[self.name + "_A", self.name + "_b"]

        if istress is None and r is None:
            r = np.asarray(self.distances)
        elif isinstance(istress, int) or isinstance(istress, list):
            if r is not None:
                logger.warning("kwarg 'r' is only used if istress is None!")
            r = self.distances.iloc[istress]
        elif istress is not None and r is None:
            raise ValueError("Parameter 'istress' must be None, list or int!")

        return self.rfunc.variance_gain(A, b, var_A, var_b, cov_Ab, r=r)


class RechargeModel(StressModelBase):
    """Stressmodel simulating the effect of groundwater recharge on the head.

    Parameters
    ----------
    prec: pandas.Series
        pandas.Series with pandas.DatetimeIndex containing the precipitation series.
        The precipitation series should be provided in mm/day when a nonlinear model is
        used.
    evap: pandas.Series
        pandas.Series with pandas.DatetimeIndex containing the potential evaporation
        series. The evaporation series should be provided in mm/day when a nonlinear
        model is used.
    rfunc: pastas.rfunc instance, optional
        Instance of the response function used in the convolution with the stress.
        Default is ps.Exponential().
    name: str, optional
        Name of the stress. Default is "recharge".
    recharge: pastas.recharge instance, optional
        Instance of a recharge model. Options are: Linear, FlexModel and Berendrecht.
        These can be accessed through ps.rch. Default is ps.rch.Linear().
    temp: pandas.Series, optional
        pandas.Series with pandas.DatetimeIndex containing the temperature series.
        It depends on the recharge model if this argument is required or not. The
        temperature series should be provided in degrees Celsius.
    settings: list of dicts or str, optional
        The settings of the precipitation, evaporation and optionally temperature time
        series, in this order. By default ("prec", "evap", "evap"). This can be a string
        referring to a predefined settings dict (defined in ps.rcParams["timeseries"]),
        or a dict with the settings to apply. For more information refer to Time Series
        Settings section below for more information.
    metadata: tuple of dicts or list of dicts, optional
        dictionary containing metadata about the stress. This is passed onto the
        TimeSeries object.

    Examples
    --------
    >>> sm = ps.RechargeModel(rain, evap, rfunc=ps.Exponential(),
    >>>                       recharge=ps.rch.FlexModel(), name="rch")
    >>> ml.add_stressmodel(sm)

    Time series settings
    --------------------
    fill_nan : {"drop", "mean", "interpolate"} or float
        Method for filling NaNs.
           * `drop`: drop NaNs from time series
           * `mean`: fill NaNs with mean value of time series
           * `interpolate`: fill NaNs by interpolating between finite values
           * `float`: fill NaN with provided value, e.g. 0.0
    fill_before : {"mean", "bfill"} or float
        Method for extending time series into past.
           * `mean`: extend time series into past with mean value of time series
           * `bfill`: extend time series into past by back-filling first value
           * `float`: extend time series into past with provided value, e.g. 0.0
    fill_after : {"mean", "ffill"} or float
        Method for extending time series into future.
           * `mean`: extend time series into future with mean value of time series
           * `ffill`: extend time series into future by forward-filling last value
           * `float`: extend time series into future with provided value, e.g. 0.0
    sample_up : {"mean", "interpolate", "divide"} or float
        Method for up-sampling time series (increasing frequency, e.g. going from weekly
        to daily values).
           * `bfill` or `backfill`: fill up-sampled time steps by back-filling current
             values
           * `ffill` or `pad`: fill up-sampled time steps by forward-filling current
             values
           * `mean`: fill up-sampled time steps with mean of timeseries
           * `interpolate`: fill up-sampled time steps by interpolating between current
             values
           * `divide`: fill up-sampled steps with current value divided by length of
             current time steps (i.e. spread value over new time steps).
    sample_down : {"mean", "drop", "sum", "min", "max"}
        Method for down-sampling time series (decreasing frequency, e.g. going from
        daily to weekly values).
           * `mean`: resample time series by taking the mean
           * `drop`: resample the time series by taking the mean, dropping any
             NaN-values
           * `sum`: resample time series by summing values
           * `max`: resample time series with maximum value
           * `min`: resample time series with minimum value

    Notes
    -----
    This stress model computes the contribution of precipitation and potential
    evaporation in two steps. In the first step a recharge flux is computed by a
    model determined by the input argument `recharge`. In the second step this
    recharge flux is convolved with a response function to obtain the contribution
    of recharge to the groundwater levels. If a nonlinear recharge model is used, the
    precipitation should be in mm/d.

    Warnings
    --------
    We recommend not to store a RechargeModel is a variable named `rm`. This name is
    already reserved in IPython to remove files and will cause problems later.

    Raises
    ------
    A warning if the the maximum annual precipitation is smaller than 12 and a
    nonlinear recharge model is applied. This is likely an indication that the units of
    the precipitation series are in m/d instead of mm/d. Please check the units of the
    precipitation series.

    See Also
    --------
    pastas.rfunc
    pastas.timeseries.TimeSeries
    pastas.recharge
    """

    _name = "RechargeModel"

    def __init__(
        self,
        prec: Series,
        evap: Series,
        rfunc: Optional[RFunc] = None,
        name: str = "recharge",
        recharge: Optional[Recharge] = None,
        temp: Optional[Series] = None,
        settings: Tuple[
            Union[str, StressSettingsDict],
            Union[str, StressSettingsDict],
            Union[str, StressSettingsDict],
        ] = (
            "prec",
            "evap",
            "evap",
        ),
        metadata: Optional[Tuple[dict, dict, dict]] = (None, None, None),
    ) -> None:
        if rfunc is None:
            rfunc = Exponential()

        if recharge is None:
            recharge = Linear()

        # Store the precipitation and evaporation time series
        self.prec = TimeSeries(prec, settings=settings[0], metadata=metadata[0])
        self.evap = TimeSeries(evap, settings=settings[1], metadata=metadata[1])

        # Store recharge object
        self.recharge = recharge

        # Store a temperature time series if provided/needed or set to None
        if self.recharge.snow is True and temp is None:
            msg = (
                "Recharge model requires a temperature series. No temperature series "
                "were provided."
            )
            raise TypeError(msg)
        if temp is not None:
            if len(settings) < 3 or len(metadata) < 3:
                msg = "Number of values for the settings and/or metadata is incorrect."
                raise TypeError(msg)
            else:
                self.temp = TimeSeries(temp, settings=settings[2], metadata=metadata[2])
        else:
            self.temp = None

        # Select indices from validated stress where both series are available.
        index = self.prec.series.index.intersection(self.evap.series.index)
        if index.empty:
            msg = (
                "The stresses that were provided have no overlapping time indices. "
                "Please make sure the indices of the time series overlap."
            )
            logger.error(msg)
            raise Exception(msg)

        # Calculate initial recharge estimation for initial rfunc parameters
        p = self.recharge.get_init_parameters().initial.values
        gain_scale_factor = self.get_stress(
            p=p, tmin=index.min(), tmax=index.max(), freq=self.prec.settings["freq"]
        ).std()

        StressModelBase.__init__(
            self,
            name=name,
            tmin=index.min(),
            tmax=index.max(),
            rfunc=rfunc,
            up=True,
            gain_scale_factor=gain_scale_factor,
        )

        self.stress = [self.prec, self.evap]
        if self.temp:
            self.stress.append(self.temp)
        self.freq = self.prec.settings["freq"]
        self.set_init_parameters()
        if isinstance(self.recharge, Linear):
            self.nsplit = 2
        else:
            self.nsplit = 1

            # Check if precipitation is likely in mm/d and not m/d. If the maximum
            # value of the annual sums is smaller than 12 (m/d), the highest annual
            # precipitation in the world, then the precipitation is very likely in m/d
            # and not in mm/d. In this case a warning is given for nonlinear models.

            freq_offset = "YE" if pandas_version >= parse_version("2.2.0") else "A"
            if self.prec.series.resample(freq_offset).sum().max() < 12:
                msg = (
                    "The maximum annual precipitation is smaller than 12 m/d. Please "
                    "double-check if the stresses are in mm/d and not in m/d."
                )
                logger.warning(msg)

    def set_init_parameters(self) -> None:
        """Internal method to set the initial parameters."""
        self.parameters = concat(
            [
                self.rfunc.get_init_parameters(self.name),
                self.recharge.get_init_parameters(self.name),
            ]
        )

    def update_stress(
        self,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
    ) -> None:
        """Method to update the settings of the all stresses in the stress model.

        Parameters
        ----------
        freq: str, optional
            String representing the desired frequency of the time series. Must be one
            of the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        tmin: str or pandas.Timestamp, optional
            String that can be converted to, or a Pandas Timestamp with the minimum
            time of the series.
        tmax: str or pandas.Timestamp, optional
            String that can be converted to, or a Pandas Timestamp with the maximum
            time of the series.

        Notes
        -----
        For the individual options for the different settings please refer to the
        docstring from the TimeSeries.update_series() method.

        See Also
        --------
        ps.timeseries.TimeSeries.update_series
        """
        self.prec.update_series(freq=freq, tmin=tmin, tmax=tmax)
        self.evap.update_series(freq=freq, tmin=tmin, tmax=tmax)
        if self.temp is not None:
            self.temp.update_series(freq=freq, tmin=tmin, tmax=tmax)

        if freq:
            self.freq = freq

    def simulate(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        dt: float = 1.0,
        istress: Optional[int] = None,
        **kwargs,
    ) -> Series:
        """Method to simulate the contribution of recharge to the head.

        Parameters
        ----------
        p: array_like, optional
            array_like object with the values as floats representing the model
            parameters.
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
        b = self._get_block(p[: self.rfunc.nparam], dt, tmin, tmax)
        stress = self.get_stress(
            p=p, tmin=tmin, tmax=tmax, freq=freq, istress=istress
        ).values
        name = self.name

        if istress is not None:
            if istress == 1 and self.nsplit > 1:
                # only happen when Linear is used as the recharge model
                stress = stress * p[-1]
            if self.stress[istress].name is not None:
                name = f"{self.name} ({self.stress[istress].name})"

        return Series(
            data=fftconvolve(stress, b, "full")[: stress.size],
            index=self.prec.series.index,
            name=name,
        )

    def get_stress(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        istress: Optional[int] = None,
        **kwargs,
    ) -> Series:
        """Method to obtain the recharge stress calculated by the model.

        Parameters
        ----------
        p: array_like, optional
            array_like object with the values as floats representing the model
            parameters.
        tmin: string, optional
        tmax: string, optional
        freq: string, optional
        istress: int, optional
            Return one of the stresses used for the recharge calculation. 0 for
            precipitation, 1 for evaporation and 2 for temperature.
        kwargs

        Returns
        -------
        stress: pandas.Series
            When no istress is selected, this return the estimated recharge flux that
            is convolved with a response function on the simulate method.
        """
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax

        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)

        if istress is None:
            prec = self.prec.series.values
            evap = self.evap.series.values
            temp = None
            if self.temp is not None:
                temp = self.temp.series.values
            if p is None:
                p = self.parameters.initial.values
            stress = self.recharge.simulate(
                prec=prec, evap=evap, p=p[-self.recharge.nparam :], **{"temp": temp}
            )
            return Series(
                data=stress,
                index=self.prec.series.index,
                name="recharge",
            )
        elif istress == 0:
            return self.prec.series
        elif istress == 1:
            return self.evap.series
        else:
            return self.temp.series

    def get_water_balance(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
    ) -> DataFrame:
        """Method to obtain the water balance components.

        Parameters
        ----------
        p: array_like, optional
            array_like object with the values as floats representing the model
            parameters.
        tmin: string, optional
        tmax: string, optional
        freq: string, optional

        Returns
        -------
        wb: pandas.DataFrame
            Dataframe with the water balance components, both fluxes and states.

        Notes
        -----
        This method return a data frame with all water balance components, fluxes and
        states. All ingoing fluxes have a positive sign (e.g., precipitation) and all
        outgoing fluxes have negative sign (e.g., recharge).

        Warnings
        --------
        This is an experimental method and may change in the future.

        Examples
        --------
        >>> sm = ps.RechargeModel(prec, evap, ps.Gamma(), ps.rch.FlexModel(),
        >>>                       name="rch")
        >>> ml.add_stressmodel(sm)
        >>> ml.solve()
        >>> wb = sm.get_water_balance(ml.get_parameters("rch"))
        >>> wb.plot(subplots=True)
        """
        if p is None:
            p = self.parameters.initial.values

        prec = self.get_stress(tmin=tmin, tmax=tmax, freq=freq, istress=0).values
        evap = self.get_stress(tmin=tmin, tmax=tmax, freq=freq, istress=1).values

        if self.temp is not None:
            temp = self.get_stress(tmin=tmin, tmax=tmax, freq=freq, istress=2).values
        else:
            temp = None
        df = self.recharge.get_water_balance(
            prec=prec, evap=evap, temp=temp, p=p[-self.recharge.nparam :]
        )
        df.index = self.prec.series.index
        return df

    def get_parameters(self, model=None, istress: Optional[int] = None) -> ArrayLike:
        """Get parameters and return as array.

        Parameters
        ----------
        model : pastas.Model, optional
            If provided, and the model is solved, return optimal model parameter-values.
            Otherwise, return initial parameter-values.
        istress : int, optional
            if provided, return specific parameter set, else return all parameters.

        Returns
        -------
        p : array_like
            An array of the parameters of the stressmodel.
        """
        if model is None:
            p = self.parameters.initial.values
        else:
            p = model.get_parameters(self.name)

        if istress is not None and isinstance(self.recharge, Linear):
            if istress == 0:
                p = p[:-1]
            elif istress == 1:
                p[0] *= p[-1]
                p = p[:-1]
        return p

    def to_dict(self, series: bool = True) -> dict:
        """Method to export the RechargeModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the object.

        Notes
        -----
        Settings and metadata are exported with the stress.
        """
        data = {
            "class": self._name,
            "prec": self.prec.to_dict(series=series),
            "evap": self.evap.to_dict(series=series),
            "rfunc": self.rfunc.to_dict(),
            "name": self.name,
            "recharge": self.recharge.to_dict(),
            "temp": self.temp.to_dict() if self.temp else None,
        }
        return data


class TarsoModel(RechargeModel):
    """Stressmodel simulating the effect of recharge using the Tarso method.

    Parameters
    ----------
    prec: pandas.Series
        pandas.Series with pandas.DatetimeIndex containing the precipitation series.
    evap: pandas.Series
        pandas.Series with pandas.DatetimeIndex containing the potential evaporation
        series.
    oseries: pandas.Series, optional
        A pandas.Series with pandas.DatetimeIndex of observations to which the model
        will be calibrated. It is used to determine the initial values of the
        drainage levels and the boundaries of the upper drainage level. Specify
        either oseries or dmin and dmax.
    dmin: float, optional
        The minimum drainage level. It is used to determine the initial values of the
        drainage levels and the lower boundary of the upper drainage level. Specify
        either oseries or dmin and dmax.
    dmax : float, optional
        The maximum drainage level. It is used to determine the initial values of the
        drainage levels and the upper boundary of the upper drainage level. Specify
        either oseries or dmin and dmax.
    rfunc: pastas.rfunc instance
        this model only works with the Exponential response function.

    See Also
    --------
    pastas.recharge

    Notes
    -----
    The Threshold autoregressive self-exciting open-loop (Tarso) model
    :cite:t:`knotters_tarso_1999` is nonlinear in structure because it incorporates
    two regimes which are separated by a threshold. This model method can be used to
    simulate a groundwater system where the groundwater head reaches the surface or
    drainage level in wet conditions. TarsoModel uses two drainage levels, with two
    exponential response functions. When the simulation reaches the second drainage
    level, the second response function becomes active. Because of its structure,
    TarsoModel cannot be combined with other stress models, a constant or a transform.
    TarsoModel inherits from RechargeModel. Only parameters specific to the child
    class are named above.
    """

    _name = "TarsoModel"

    def __init__(
        self,
        prec: Series,
        evap: Series,
        oseries: Optional[Series] = None,
        dmin: Optional[float] = None,
        dmax: Optional[float] = None,
        rfunc: Optional[RFunc] = None,
        **kwargs,
    ) -> None:
        if oseries is not None:
            if dmin is not None or dmax is not None:
                msg = "Please specify either oseries or dmin and dmax"
                raise (Exception(msg))
            dmin = oseries.min()
            dmax = oseries.max()
        elif dmin is None or dmax is None:
            msg = "Please specify either oseries or dmin and dmax"
            raise (Exception(msg))
        if rfunc is None:
            rfunc = Exponential()
        if not isinstance(rfunc, Exponential):
            raise NotImplementedError("TarsoModel only supports rfunc Exponential!")
        self.dmin = dmin
        self.dmax = dmax
        super().__init__(prec=prec, evap=evap, rfunc=rfunc, **kwargs)
        self.nsplit = 1

    def set_init_parameters(self) -> None:
        # parameters for the first drainage level
        p0 = self.rfunc.get_init_parameters(self.name)
        initial = self.dmin + 0.5 * (self.dmax - self.dmin)
        pd0 = Series(
            {
                "initial": initial,
                "pmin": np.nan,
                "pmax": np.nan,
                "vary": True,
                "name": self.name,
                "dist": "uniform",
            }
        )
        p0.loc[f"{self.name}_d"] = pd0
        p0.index = [f"{x}0" for x in p0.index]

        # parameters for the second drainage level
        p1 = self.rfunc.get_init_parameters(self.name)
        initial = self.dmin + 0.75 * (self.dmax - self.dmin)
        pd1 = Series(
            {
                "initial": initial,
                "pmin": self.dmin,
                "pmax": self.dmax,
                "vary": True,
                "name": self.name,
                "dist": "uniform",
            }
        )
        p1.loc[f"{self.name}_d"] = pd1
        p1.index = [f"{x}1" for x in p1.index]

        # parameters for the recharge-method
        pr = self.recharge.get_init_parameters(self.name)

        # combine all parameters
        self.parameters = concat([p0, p1, pr])

    def simulate(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq=None,
        dt: float = 1.0,
    ) -> Series:
        stress = self.get_stress(p=p, tmin=tmin, tmax=tmax, freq=freq)
        h = self.tarso(p[: -self.recharge.nparam], stress.values, dt)
        sim = Series(h, name=self.name, index=stress.index)
        return sim

    def to_dict(self, series: bool = True) -> dict:
        """Method to export the TarsoModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the object.

        Notes
        -----
        Settings and metadata are exported with the stress.
        """
        data = super().to_dict(series)
        data["dmin"] = self.dmin
        data["dmax"] = self.dmax
        return data

    @staticmethod
    def _check_stressmodel_compatibility(ml: Model) -> None:
        """Internal method to check if no other stressmodels, a constants or a transform
        is used."""
        msg = (
            "A TarsoModel cannot be combined with %s. Either remove the TarsoModel or "
            "the %s."
        )
        if len(ml.stressmodels) > 1:
            logger.warning(msg, "other stressmodels", "stressmodels")
        if ml.constant is not None:
            logger.warning(msg, "a constant", "constant")
        if ml.transform is not None:
            logger.warning(msg, "a transform", "transform")

    @staticmethod
    @njit
    def tarso(p: ArrayLike, r: ArrayLike, dt: float) -> ArrayLike:
        """Calculates the head based on exponential decay of the previous timestep and
        recharge, using two thresholds."""
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

        h = np.full(len(r), np.nan)
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
                dtdr = -S * c * np.log((d1 - d - r[i] * c) / (h0 - d - r[i] * c))
                if dtdr > dt:
                    raise (Exception())
                # change parameters
                high = newhigh
                if high:
                    S, a, c, d = S1, a_e, c_e, d_e
                else:
                    S, a, c, d = S0, a0, c0, d0
                # calculate new level after reaching d1
                exp_a = np.exp(-(dt - dtdr) / a)
                h[i] = (d1 - d) * exp_a + r[i] * c * (1 - exp_a) + d
        return h


class ChangeModel(StressModelBase):
    """Model where the response function changes from one to another over time.

    Parameters
    ----------
    stress: pandas.Series
        pandas Series object containing the stress.
    rfunc1: pastas.rfunc instance
        The instance of the response function used in the convolution with the stress.
    rfunc2: pastas.rfunc instance
        The instance of the response function used in the convolution with the stress.
    name: str
        name of the stress.
    tchange: str
        string with the approximate date of the change.
    up: bool or None, optional
        True if response function is positive (default), False if negative. None if
        you don't want to define if response is positive or negative.
    settings: dict or str, optional
        The settings of the stress. This can be a string referring to a predefined
        settings dict (defined in ps.rcParams["timeseries"]), or a dict with the
        settings to apply. For more information, refer to the docs of pastas.Timeseries
        for further information.
    metadata: dict, optional
        dictionary containing metadata about the stress. This is passed onto the
        TimeSeries object.

    Time series settings
    --------------------
    fill_nan : {"drop", "mean", "interpolate"} or float
        Method for filling NaNs.
           * `drop`: drop NaNs from time series
           * `mean`: fill NaNs with mean value of time series
           * `interpolate`: fill NaNs by interpolating between finite values
           * `float`: fill NaN with provided value, e.g. 0.0
    fill_before : {"mean", "bfill"} or float
        Method for extending time series into past.
           * `mean`: extend time series into past with mean value of time series
           * `bfill`: extend time series into past by back-filling first value
           * `float`: extend time series into past with provided value, e.g. 0.0
    fill_after : {"mean", "ffill"} or float
        Method for extending time series into future.
           * `mean`: extend time series into future with mean value of time series
           * `ffill`: extend time series into future by forward-filling last value
           * `float`: extend time series into future with provided value, e.g. 0.0
    sample_up : {"mean", "interpolate", "divide"} or float
        Method for up-sampling time series (increasing frequency, e.g. going from weekly
        to daily values).
           * `bfill` or `backfill`: fill up-sampled time steps by back-filling current
             values
           * `ffill` or `pad`: fill up-sampled time steps by forward-filling current
             values
           * `mean`: fill up-sampled time steps with mean of timeseries
           * `interpolate`: fill up-sampled time steps by interpolating between current
             values
           * `divide`: fill up-sampled steps with current value divided by length of
             current time steps (i.e. spread value over new time steps).
    sample_down : {"mean", "drop", "sum", "min", "max"}
        Method for down-sampling time series (decreasing frequency, e.g. going from
        daily to weekly values).
           * `mean`: resample time series by taking the mean
           * `drop`: resample the time series by taking the mean, dropping any
             NaN-values
           * `sum`: resample time series by summing values
           * `max`: resample time series with maximum value
           * `min`: resample time series with minimum value

    Notes
    -----
    This model is based on :cite:t:`obergfell_identification_2019`.
    """

    _name = "ChangeModel"

    def __init__(
        self,
        stress: Series,
        rfunc1: RFunc,
        rfunc2: RFunc,
        name: str,
        tchange: Union[str, TimestampType],
        up: bool = True,
        settings: Optional[Union[str, StressSettingsDict]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        stress = TimeSeries(stress, settings=settings, metadata=metadata)

        StressModelBase.__init__(
            self,
            name=name,
            rfunc=None,
            tmin=stress.series.index.min(),
            tmax=stress.series.index.max(),
        )

        rfunc1.update_rfunc_settings(up=up)
        self.rfunc1 = rfunc1

        rfunc2.update_rfunc_settings(up=up)
        self.rfunc2 = rfunc2
        self.tchange = Timestamp(tchange)

        self.freq = stress.settings["freq"]
        self.stress.append(stress)
        self.set_init_parameters()

    def set_init_parameters(self) -> None:
        """Internal method to set the initial parameters."""
        self.parameters = concat(
            [
                self.rfunc1.get_init_parameters("{}_1".format(self.name)),
                self.rfunc2.get_init_parameters("{}_2".format(self.name)),
            ]
        )

        tmin = Timestamp.min.toordinal()
        tmax = Timestamp.max.toordinal()
        tchange = self.tchange.toordinal()

        self.parameters.loc[self.name + "_beta"] = (
            0.0,
            -np.inf,
            np.inf,
            True,
            self.name,
            "uniform",
        )
        self.parameters.loc[self.name + "_tchange"] = (
            tchange,
            tmin,
            tmax,
            False,
            self.name,
            "uniform",
        )
        self.parameters.name = self.name

    def simulate(
        self,
        p: ArrayLike,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        dt: float = 1.0,
    ) -> Series:
        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)
        rfunc1 = self.rfunc1.block(p[: self.rfunc1.nparam])
        rfunc2 = self.rfunc2.block(
            p[self.rfunc1.nparam : self.rfunc1.nparam + self.rfunc2.nparam]
        )

        stress = self.stress[0].series
        npoints = stress.index.size
        t = np.linspace(0, 1, npoints)
        beta = p[-2]

        sigma = stress.index.get_loc(Timestamp.fromordinal(int(p[-1]))) / npoints
        omega = 1 / (np.exp(beta * (t - sigma)) + 1)

        h1 = Series(
            data=fftconvolve(stress, rfunc1, "full")[:npoints],
            index=stress.index,
            name="1",
        )
        h2 = Series(
            data=fftconvolve(stress, rfunc2, "full")[:npoints],
            index=stress.index,
            name="1",
        )
        h = omega * h1 + (1 - omega) * h2

        return h

    def to_dict(self, series: bool = True):
        """Method to export the ChangeModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the object.

        Notes
        -----
        Settings and metadata are exported with the stress.
        """
        data = {
            "stress": self.stress[0].to_dict(series=series),
            "rfunc1": self.rfunc1.to_dict(),
            "rfunc2": self.rfunc2.to_dict(),
            "name": self.name,
            "tchange": self.tchange,
            "up": self.rfunc1.up,
        }
        return data
