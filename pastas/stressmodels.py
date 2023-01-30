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

import inspect
from logging import getLogger
from importlib.metadata import version

# Type Hinting
from typing import List, Optional, Tuple, Union

import numpy as np
from pandas import DataFrame, Series, Timedelta, Timestamp, concat, date_range
from scipy.signal import fftconvolve

from pastas.typing import ArrayLike, Model, Recharge, RFunc, TimestampType

from .decorators import njit, set_parameter
from .recharge import Linear
from .rfunc import Exponential, HantushWellModel, One
from .timeseries import TimeSeries
from .utils import validate_name

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
            if inspect.isclass(rfunc):
                raise DeprecationWarning(
                    "Response functions should be added to a stress model as an "
                    "instance (e.g., ps.One()), and not as a class (e.g., ps.One). "
                    "Please provide an instance of the response function."
                )
            rfunc._update_rfunc_settings(up=up, gain_scale_factor=gain_scale_factor)
        self.rfunc = rfunc

        self.parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])

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
        self.parameters.loc[name, "initial"] = value

    @set_parameter
    def _set_pmin(self, name: str, value: float) -> None:
        """Internal method to set the lower bound of the parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "pmin"] = value

    @set_parameter
    def _set_pmax(self, name: str, value: float) -> None:
        """Internal method to set the upper bound of the parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "pmax"] = value

    @set_parameter
    def _set_vary(self, name: str, value: float) -> None:
        """Internal method to set if the parameter is varied during optimization.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "vary"] = bool(value)

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
    cutoff: float, optional
        This argument is deprecated since 0.23. Directly provide this to directly to
        the response function instance (e.g., ps.Gamma(cutoff=0.999).
    settings: dict or str, optional
        The settings of the stress. This can be a string referring to a predefined
        settings dict, or a dict with the settings to apply. Refer to the docstring
        of pastas.Timeseries for further information.
    metadata: dict, optional
        dictionary containing metadata about the stress. This is passed onto the
        TimeSeries object.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    meanstress: float, optional
        This argument is deprecated since 0.23. Use `gain_scale_factor` instead.

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
        cutoff=None,
        settings: Optional[Union[dict, str]] = None,
        metadata: Optional[dict] = None,
        gain_scale_factor: Optional[float] = None,
        meanstress: Optional[float] = None,
    ) -> None:
        raise_deprecations(meanstress=meanstress, cutoff=cutoff)

        if isinstance(stress, list):
            logger.warning(
                "providing a stress as list is deprecated and will results "
                "in an Error in Pastas 1.0."
            )
            stress = stress[0]  # TODO Remove in Pastas 1.0

        stress = TimeSeries(stress, settings=settings, metadata=metadata)

        StressModelBase.__init__(
            self,
            name=name,
            tmin=stress.series.index.min(),
            tmax=stress.series.index.max(),
            rfunc=rfunc,
            up=up,
            gain_scale_factor=stress.series.std()
            if gain_scale_factor is None
            else gain_scale_factor,
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
            fastpath=True,
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
    cutoff: float, optional
        This argument is deprecated since 0.23. Directly provide this to directly to
        the response function instance (e.g., ps.Gamma(cutoff=0.999).

    Notes
    -----
    The step trend is calculated as follows. First, a binary series is created,
    with zero values before tstart, and ones after the start. This series is
    convoluted with the block response to simulate a step trend.
    """

    _name = "StepModel"

    def __init__(
        self,
        tstart: TimestampType,
        name: str,
        rfunc: Optional[RFunc] = None,
        up: bool = True,
        cutoff=None,
    ) -> None:
        raise_deprecations(cutoff=cutoff)

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
            fastpath=True,
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

        self.parameters.loc[self.name + "_a"] = (0.0, -np.inf, np.inf, True, self.name)
        self.parameters.loc[self.name + "_tstart"] = (
            start,
            tmin,
            tmax,
            False,
            self.name,
        )
        self.parameters.loc[self.name + "_tend"] = (end, tmin, tmax, False, self.name)

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
    rfunc: pastas.rfunc instance
        this model only works with the HantushWellModel response function.
    name: str
        Name of the stressmodel.
    distances: array_like
        list of distances to oseries, must be ordered the same as the stresses.
    up: bool, optional
        whether a positive stress has an increasing or decreasing effect on the model,
        by default False, in which case positive stress lowers e.g., the groundwater
        level.
    cutoff: float, optional
        This argument is deprecated since 0.23. Directly provide this to directly to
        the response function instance (e.g., ps.Gamma(cutoff=0.999).
    settings: str, list of dict, optional
        settings of the time series, by default "well".
    sort_wells: bool, optional
        sort wells from closest to furthest, by default True.

    Notes
    -----
    This class implements convolution of multiple series with the same response
    function. This can be applied when dealing with multiple wells in a time series
    model. The distance(s) from the pumping well(s) to the monitoring well have to be
    provided for each stress.

    Warnings
    --------
    This model only works with the HantushWellModel response function.
    """

    _name = "WellModel"

    def __init__(
        self,
        stress: List[Series],
        rfunc: RFunc,
        name: str,
        distances: ArrayLike,
        up: bool = False,
        cutoff=None,
        settings: str = "well",
        sort_wells: bool = True,
        metadata: Optional[list] = None,
    ) -> None:
        raise_deprecations(cutoff=cutoff)
        if not isinstance(rfunc, HantushWellModel):
            raise NotImplementedError(
                "WellModel only supports the rfunc HantushWellModel!"
            )

        # sort wells by distance
        self.sort_wells = sort_wells
        if self.sort_wells:
            stress = [
                s for _, s in sorted(zip(distances, stress), key=lambda pair: pair[0])
            ]
            if isinstance(settings, list):
                settings = [
                    s
                    for _, s in sorted(
                        zip(distances, settings), key=lambda pair: pair[0]
                    )
                ]

            distances = np.sort(distances)

        if settings is None or isinstance(settings, str):
            settings = len(stress) * [settings]

        # convert stresses to TimeSeries if necessary
        stress = self._handle_stress(stress, settings)

        # Check if number of stresses and distances match
        if len(stress) != len(distances):
            msg = (
                "The number of stresses does not match the number of distances "
                "provided."
            )
            logger.error(msg)
            raise ValueError(msg)
        else:
            self.distances = Series(
                index=[s.name for s in stress], data=distances, name="distances"
            )

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
            h = h.add(Series(c, index=stress.index, fastpath=True), fill_value=0.0)
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
    def _handle_stress(stress, settings):
        """Internal method to handle user provided stress in init.

        Parameters
        ----------
        stress: pandas.Series, list or dict
            stress or collection of stresses.
        settings: dict or iterable
            settings dictionary.

        Returns
        -------
        stress: list
            return a list with the stresses transformed to pastas.TimeSeries.
        """
        data = []

        if isinstance(stress, Series):
            data.append(TimeSeries(stress, settings=settings))
        elif isinstance(stress, dict):
            for i, (name, value) in enumerate(stress.items()):
                data.append(TimeSeries(value, name=name, settings=settings[i]))
        elif isinstance(stress, list):
            for i, value in enumerate(stress):
                data.append(TimeSeries(value, settings=settings[i]))
        else:
            logger.error("Stress format is unknown. Provide a Series, dict or list.")
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
            if provided, return optimal model parameters, else return initial
            parameters.
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
        if model.fit is None:
            raise AttributeError("Model not optimized! Run solve() first!")
        if self.rfunc._name != "HantushWellModel":
            raise ValueError("Response function must be HantushWellModel!")
        if model.fit.pcov.isna().all(axis=None):
            model.logger.warning("Covariance matrix contains only NaNs!")

        # get parameters and (co)variances
        A = model.parameters.loc[self.name + "_A", "optimal"]
        b = model.parameters.loc[self.name + "_b", "optimal"]
        var_A = model.fit.pcov.loc[self.name + "_A", self.name + "_A"]
        var_b = model.fit.pcov.loc[self.name + "_b", self.name + "_b"]
        cov_Ab = model.fit.pcov.loc[self.name + "_A", self.name + "_b"]

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
    evap: pandas.Series
        pandas.Series with pandas.DatetimeIndex containing the potential evaporation
        series.
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
        It depends on the recharge model is this argument is required or not.
    cutoff: float, optional
        This argument is deprecated since 0.23. Directly provide this to directly to
        the response function instance (e.g., ps.Gamma(cutoff=0.999).
    settings: list of dicts or str, optional
        The settings of the precipitation and evaporation time series, in this order.
        This can be a string referring to a predefined settings dict, or a dict with
        the settings to apply. Refer to the docstring of pastas.Timeseries for
        further information. Default is ("prec", "evap").
    metadata: tuple of dicts or list of dicts, optional
        dictionary containing metadata about the stress. This is passed onto the
        TimeSeries object.

    See Also
    --------
    pastas.rfunc
    pastas.timeseries.TimeSeries
    pastas.recharge

    Notes
    -----
    This stress model computes the contribution of precipitation and potential
    evaporation in two steps. In the first step a recharge flux is computed by a
    model determined by the input argument `recharge`. In the second step this
    recharge flux is convoluted with a response function to obtain the contribution
    of recharge to the groundwater levels.

    Examples
    --------
    >>> sm = ps.RechargeModel(rain, evap, rfunc=ps.Exponential(),
    >>>                       recharge=ps.rch.FlexModel(), name="rch")
    >>> ml.add_stressmodel(sm)

    Warnings
    --------
    We recommend not to store a RechargeModel is a variable named `rm`. This name is
    already reserved in IPython to remove files and will cause problems later.
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
        cutoff=None,
        settings: Tuple[Union[str, dict], Union[str, dict], Union[str, dict]] = (
            "prec",
            "evap",
            "evap",
        ),
        metadata: Optional[Tuple[dict, dict, dict]] = (None, None, None),
    ) -> None:
        raise_deprecations(cutoff=cutoff)
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
                "were provided"
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
            fastpath=True,
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
            is convoluted with a response function on the simulate method.
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
                fastpath=True,
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
        rfunc: Optional[RFunc] = Exponential(),
        **kwargs,
    ) -> None:
        if oseries is not None:
            if dmin is not None or dmax is not None:
                msg = "Please specify either oseries or dmin and dmax"
                raise (Exception(msg))
            o = TimeSeries(oseries).series
            dmin = o.min()
            dmax = o.max()
        elif dmin is None or dmax is None:
            msg = "Please specify either oseries or dmin and dmax"
            raise (Exception(msg))
        if rfunc is None:
            rfunc = Exponential()
        if not isinstance(rfunc, Exponential):
            raise NotImplementedError("TarsoModel only supports rfunc Exponential!")
        self.oseries = oseries
        self.dmin = dmin
        self.dmax = dmax
        super().__init__(prec=prec, evap=evap, rfunc=rfunc, **kwargs)

    def set_init_parameters(self) -> None:
        # parameters for the first drainage level
        p0 = self.rfunc.get_init_parameters(self.name)
        one = One()
        gain_scale_factor = self.dmin + 0.5 * (self.dmax - self.dmin)
        one._update_rfunc_settings(gain_scale_factor=gain_scale_factor)
        pd0 = one.get_init_parameters(self.name).squeeze()
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
        data["oseries"] = self.oseries
        return data

    @staticmethod
    def _check_stressmodel_compatibility(ml: Model) -> None:
        """Internal method to check if no other stressmodels, a constants or a
        transform is used."""
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
        """Calculates the head based on exponential decay of the previous timestep
        and recharge, using two thresholds."""
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
    cutoff: float, optional
        This argument is deprecated since 0.23. Directly provide this to directly to
        the response function instance (e.g., ps.Gamma(cutoff=0.999).
    settings: dict or str, optional
        the settings of the stress. This can be a string referring to a predefined
        settings dict, or a dict with the settings to apply. Refer to the docstring
        of pastas.Timeseries for further information.
    metadata: dict, optional
        dictionary containing metadata about the stress. This is passed onto the
        TimeSeries object.

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
        cutoff=None,
        settings: Optional[Union[dict, str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        raise_deprecations(cutoff=cutoff)

        if isinstance(stress, list):
            logger.warning(
                "providing a stress as list is deprecated and will results "
                "in an Error in Pastas 1.0."
            )
            stress = stress[0]  # TODO Remove in Pastas 1.0

        stress = TimeSeries(stress, settings=settings, metadata=metadata)

        StressModelBase.__init__(
            self,
            name=name,
            rfunc=None,
            tmin=stress.series.index.min(),
            tmax=stress.series.index.max(),
        )
        if inspect.isclass(rfunc1) or inspect.isclass(rfunc2):
            raise DeprecationWarning(
                "Response functions should be provided as an instance (e.g., ps.One()),"
                " not as a class (e.g., ps.One). Please provide an instance."
            )

        rfunc1._update_rfunc_settings(up=up)
        self.rfunc1 = rfunc1

        rfunc2._update_rfunc_settings(up=up)
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
        )
        self.parameters.loc[self.name + "_tchange"] = (
            tchange,
            tmin,
            tmax,
            False,
            self.name,
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
            fastpath=True,
        )
        h2 = Series(
            data=fftconvolve(stress, rfunc2, "full")[:npoints],
            index=stress.index,
            name="1",
            fastpath=True,
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


def StressModel2(*args, **kwargs):
    msg = (
        "StressModel2 is removed in Pastas 0.23 and replaced by the RechargeModel "
        "stress model. Use ps.RechargeModel(prec, evap, recharge=ps.rch.Linear) for "
        "the same stress model in future codes."
    )
    logger.error(msg)


def raise_deprecations(meanstress=None, cutoff=None):
    if meanstress is not None:
        raise DeprecationWarning(
            "The argument `meanstress` is deprecated since Pastas 0.23. Please "
            "use the new argument `gain_scale_factor` instead."
        )
    if cutoff is not None:
        raise DeprecationWarning(
            "The argument `cutoff` is deprecated since Pastas 0.23. Please "
            "apply this directly to the response function (e.g., ps.Gamma(cutoff=0.99))"
        )
