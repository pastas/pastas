from logging import getLogger

# Type Hinting
from typing import Optional, Union

import pandas as pd
from pandas import Series
from pandas.tseries.frequencies import to_offset

from pastas.typing import Axes

from .rcparams import rcParams
from .timeseries_utils import (
    _get_dt,
    _get_time_offset,
)
from .utils import validate_name

logger = getLogger(__name__)


class TimeSeries:
    """Class that deals with all user-provided time series.

    Parameters
    ----------
    series: pandas.Series
        pandas.Series with pandas.DatetimeIndex.
    name: str, optional
        String with the name of the time series, if None is provided, pastas will try
        to derive the name from the series.
    settings: str or dict, optional
        String with the name of one of the predefined settings (obtained through
        ps.TimeSeries._predefined_settings.) or a dictionary with the settings to be
        applied. This does not have to include all the settings arguments.
    metadata: dict, optional
        Dictionary with metadata of the time series.

    Returns
    -------
    series: pastas.TimeSeries
        Returns a pastas.TimeSeries object.

    Examples
    --------
    To obtain the predefined TimeSeries settings, you can run the following line of
    code:

    >>> ps.rcParams["timeseries"]

    See Also
    --------
    pastas.timeseries.TimeSeries.update_series
        For the individual options for the different settings.
    """

    _predefined_settings = rcParams["timeseries"]

    def __init__(
        self,
        series: Series,
        name: Optional[str] = None,
        settings: Optional[Union[str, dict]] = None,
        metadata: Optional[dict] = None,
        **kwargs,  # TODO remove in Pastas 1.0
    ) -> None:
        # First, deal with all deprecated features and raise errors
        check_deprecated_input(series, settings, **kwargs)  # TODO remove in Pastas 1.0

        # Make sure we have a Pandas Series and not a 1D-DataFrame
        if isinstance(series, pd.DataFrame):
            if len(series.columns) == 1:
                series = series.iloc[:, 0]
                logger.info(
                    "1D-DataFrame was provided, automatically transformed to "
                    "pandas.Series."
                )

        # Make sure we have a workable Pandas Series, depends on type of time series
        if settings == "oseries":
            validate_oseries(series)
        else:
            if settings is not None and not isinstance(settings, str):
                if settings["fill_nan"] == "drop":
                    raise UserWarning(
                        "The fill_nan setting 'drop' for a stress is not allowed "
                        "because the stress time series  need to be equidistant. "
                        "Please change this."
                    )
            validate_stress(series)

        # Store a copy of the original series
        self._series_original = series.copy()  # copy of the original series
        self._series = None  #
        self.freq_original = pd.infer_freq(self._series_original.index)
        self.settings = {
            "freq": None,
            "sample_up": None,
            "sample_down": None,
            "fill_nan": "interpolate",
            "fill_before": None,
            "fill_after": None,
            "tmin": series.index.min(),
            "tmax": series.index.max(),
            "time_offset": pd.Timedelta(0),
        }
        self.metadata = {"x": 0.0, "y": 0.0, "z": 0.0, "projection": None}

        # Use user provided name or set from series
        if name is None:
            name = series.name
        self.name = validate_name(name)
        self._series_original.name = validate_name(name)

        if metadata is not None:
            self.metadata.update(metadata)

        # Update the settings with user-provided values, if any.
        if settings:
            if isinstance(settings, str):
                if settings in self._predefined_settings.keys():
                    settings = self._predefined_settings[settings]
                else:
                    error = (
                        f"Settings shortcut code '{settings}' is not in the "
                        f"predefined settings options. Please choose from"
                        f" {self._predefined_settings.keys()}"
                    )
                    raise KeyError(error)
            self._update_settings(**settings)

        self.update_series(force_update=True, **self.settings)

    def __repr__(self) -> str:
        """Prints a simple string representation of the time series."""
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name}, "
            f"freq={self.settings['freq']}, "
            f"freq_original={self.freq_original}, "
            f"tmin={self.settings['tmin']}, "
            f"tmax={self.settings['tmax']})"
        )

    @property
    def series_original(self) -> Series:
        return self._series_original

    @series_original.setter
    def series_original(self, series: Series) -> None:
        """Sets a new freq_original for the TimeSeries."""
        validate_stress(series)
        self._series_original = series.copy()
        self.freq_original = pd.infer_freq(self._series_original.index)
        self.settings["tmin"] = series.index.min()  # reset tmin
        self.settings["tmax"] = series.index.max()  # reset tmax
        self.update_series(force_update=True, **self.settings)

    @property
    def series(self) -> Series:
        return self._series

    @series.setter
    def series(self, value):
        raise AttributeError(
            "You cannot set series by yourself, as it is calculated from "
            "series_original. Please set series_original to update the series."
        )

    @property
    def series_validated(self):
        raise DeprecationWarning(
            "TimeSeries objects no longer have a validated time series. Use the "
            "_series_original instead."
        )

    def update_series(self, force_update: bool = False, **kwargs) -> None:
        """Method to update the series with new options.

        Parameters
        ----------
        force_update: bool, optional
            argument that is used to force an update, even when no changes are found.
            Internally used by the __init__ method. Default is False.
        freq: str, optional
            String representing the desired frequency of the time series. Must be one
            of the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        sample_up: str or float, optional
            String with the method to use when the frequency is increased (e.g.,
            Weekly to daily). Possible values are: "backfill", "bfill", "pad",
            "ffill", "mean", "interpolate", "divide" or a float value to fill the gaps.
        sample_down: str, optional
            String with the method to use when the frequency decreases (e.g., from
            daily to weekly values). Possible values are: "mean", "drop", "sum",
            "min", "max".
        fill_nan: str or float, optional
            Method to use when there ar nan-values in the time series.
            Possible values are: "mean", "drop", "interpolate" (default) or a
            float value.
        fill_before: str or float, optional
            Method used to extend a time series before any measurements are
            available. possible values are: "mean" or a float value.
        fill_after: str or float, optional
            Method used to extend a time series after any measurements are available.
            Possible values are: "mean" or a float value.
        tmin: str or pandas.Timestamp, optional
            String that can be converted to, or a Pandas Timestamp with the minimum
            time of the series.
        tmax: str or pandas.Timestamp, optional
            String that can be converted to, or a Pandas Timestamp with the maximum
            time of the series.

        Notes
        -----
        The method will validate if any of the settings is changed to determine if
        the series need to be updated.
        """
        if self._update_settings(**kwargs) or force_update:
            tmin = self.settings["tmin"]
            freq = self.settings["freq"]
            if tmin is not None and freq is not None:
                self.settings["time_offset"] = _get_time_offset(tmin, freq)

            # Get the original series to start with
            series = self._series_original.copy(deep=True)
            series = self._fill_nan(series)

            # Update the series with the new settings
            series = self._change_frequency(series)
            series = self._fill_before(series)
            series = self._fill_after(series)
            series.name = self._series_original.name

            self._series = series

    def multiply(self, other: float) -> None:
        raise DeprecationWarning(
            "TimeSeries objects no longer have the multiply method. Provide a new "
            "_series_original that is multiplied instead."
        )

    def _update_settings(self, **kwargs) -> bool:
        """Internal method that check if an update is actually necessary.

        Returns
        -------
        update: bool
            True if settings are changed and series need to be updated.
        """
        update = False
        for key, value in kwargs.items():
            if key in ["tmin", "tmax"]:
                if value is None:
                    pass
                else:
                    value = pd.Timestamp(value)
            if (value != self.settings[key]) and (value is not None):
                self.settings[key] = value
                update = True
        return update

    def _change_frequency(self, series: Series) -> Series:
        """Method to change the frequency of the time series."""
        freq = self.settings["freq"]

        # 1. If no freq string is present or is provided (e.g. Oseries)
        if not freq:
            return series
        # 2. If new frequency is required (only up or down sampling allowed)
        else:
            dt_new = _get_dt(freq)
            dt_org = _get_dt(self.freq_original)

            # If new frequency is lower than its original
            if dt_new < dt_org:
                series = self._sample_up(series)
            # If new frequency is higher than its original
            elif dt_new > dt_org:
                series = self._sample_down(series)

        # Drop nan-values at the beginning and end of the time series
        series = series.loc[series.first_valid_index() : series.last_valid_index()]

        return series

    def _sample_up(self, series: Series) -> Series:
        """Resample the time series when the frequency increases (e.g. from weekly to
        daily values)."""
        method = self.settings["sample_up"]
        freq = self.settings["freq"]

        if method in ["backfill", "bfill", "pad", "ffill"]:
            series = series.asfreq(freq, method=method)
        elif method is None:
            pass
        else:
            if method == "mean":
                series = series.asfreq(freq).fillna(series.mean())
            elif method == "interpolate":
                series = series.asfreq(freq).interpolate(method="time")
            elif method == "divide":
                dt = series.index.to_series().diff() / to_offset(freq).delta
                series = series / dt
                series = series.asfreq(freq, method="bfill")
            elif isinstance(method, float):
                series = series.asfreq(freq).fillna(method)
            else:
                logger.warning(
                    "Time Series %s: User-defined option for sample_up %s is not "
                    "supported",
                    self.name,
                    method,
                )

        logger.info("Time Series %s were sampled up using %s.", self.name, method)

        return series

    def _sample_down(self, series: Series) -> Series:
        """Resample the time series when the frequency decreases (e.g. from daily to
        weekly values).

        Notes
        -----
        make sure the labels are still at the end of each period, and data at the
        right-side of the bucket is included (see
        http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.resample.html)
        """
        method = self.settings["sample_down"]
        freq = self.settings["freq"]

        # when a multiple freq is used (like '7D') make sure the first record
        # has a rounded index
        # TODO: check if we can replace this with origin with pandas 1.1.0
        start_time = series.index[0].ceil(freq) + self.settings["time_offset"]
        series = series.loc[start_time:]

        # TODO: replace by adding offset to resample method with pandas 1.1.0
        # Shift time series back by offset so resample can take it into account
        if self.settings["time_offset"] > pd.Timedelta(0):
            series = series.shift(-1, freq=self.settings["time_offset"])

        # Provide some standard pandas arguments for all options
        kwargs = {"label": "right", "closed": "right"}

        if method == "mean":
            series = series.resample(freq, **kwargs).mean()
        elif method == "drop":
            series = series.resample(freq, **kwargs).mean().dropna()
        elif method == "sum":
            series = series.resample(freq, **kwargs).sum()
        elif method == "min":
            series = series.resample(freq, **kwargs).min()
        elif method == "max":
            series = series.resample(freq, **kwargs).max()
        else:
            logger.warning(
                "Time Series %s: User-defined option for sample down %s is not "
                "supported",
                self.name,
                method,
            )

        # TODO: replace by adding offset to resample method with pandas 1.1.0
        if self.settings["time_offset"] > pd.Timedelta(0):
            # The offset is removed by the resample-method, so we add it again
            series = series.shift(1, freq=self.settings["time_offset"])

        logger.info(
            "Time Series %s was sampled down to freq %s with method " "%s.",
            self.name,
            freq,
            method,
        )

        return series

    def _fill_nan(self, series: Series) -> Series:
        """Fill up the nan-values when present."""

        method = self.settings["fill_nan"]

        n = series.isnull().values.sum()
        if n == 0:
            pass
        elif method == "drop":
            series = series.dropna()
        elif method == "mean":
            series = series.fillna(series.mean())
        elif method == "interpolate":
            series = series.interpolate(method="time")
        elif isinstance(method, float):
            series = series.fillna(method)
        else:
            logger.warning(
                "Time Series %s: User-defined option for fill_nan %s is not supported.",
                self.name,
                method,
            )

        if n > 0:
            logger.info(
                "Time Series %s: %s nan-value(s) was/were found and filled with: %s.",
                self.name,
                n,
                method,
            )

        return series

    def _fill_before(self, series: Series) -> Series:
        """Method to add a period in front of the available time series."""
        freq = self.settings["freq"]
        method = self.settings["fill_before"]
        tmin = self.settings["tmin"]

        if tmin is None or method is None:
            pass
        elif pd.Timestamp(tmin) >= series.index.min():
            series = series.loc[pd.Timestamp(tmin) :]
        else:
            index_extend = pd.date_range(
                start=pd.Timestamp(tmin), end=series.index.min(), freq=freq
            )
            series = series.reindex(series.index.union(index_extend[:-1]))

            if method == "mean":
                mean_value = series.mean()
                series = series.fillna(mean_value)  # Default option
                logger.info(
                    "Time Series %s was extended in the past to %s with the mean "
                    "value (%.2g) of the time series.",
                    self.name,
                    series.index.min(),
                    mean_value,
                )
            elif isinstance(method, float):
                series = series.fillna(method)
                logger.info(
                    "Time Series %s was extended in the past to %s by adding %s "
                    "values.",
                    self.name,
                    series.index.min(),
                    method,
                )
            else:
                logger.info(
                    "Time Series %s: User-defined option for fill_before %s is not "
                    "supported.",
                    self.name,
                    method,
                )

        return series

    def _fill_after(self, series: Series) -> Series:
        """Method to add a period in front of the available time series."""
        freq = self.settings["freq"]
        method = self.settings["fill_after"]
        tmax = self.settings["tmax"]

        if tmax is None or method is None:
            pass
        elif pd.Timestamp(tmax) <= series.index.max():
            series = series.loc[: pd.Timestamp(tmax)]
        else:
            index_extend = pd.date_range(
                start=series.index.max(), end=pd.Timestamp(tmax), freq=freq
            )
            series = series.reindex(series.index.union(index_extend))

            if method == "mean":
                mean_value = series.mean()
                series = series.fillna(mean_value)  # Default option
                logger.info(
                    "Time Series %s was extended in the future to %s with the mean "
                    "value (%.2g) of the time series.",
                    self.name,
                    series.index.max(),
                    mean_value,
                )
            elif isinstance(method, float):
                series = series.fillna(method)
                logger.info(
                    "Time Series %s was extended in the future to %s by adding %s "
                    "values.",
                    self.name,
                    series.index.max(),
                    method,
                )
            else:
                logger.info(
                    "Time Series %s: User-defined option for fill_after %s is not "
                    "supported",
                    self.name,
                    method,
                )

        return series

    def to_dict(self, series: Optional[bool] = True) -> dict:
        """Method to export the Time Series to a json format.

        Parameters
        ----------
        series: bool, optional
            True to export the original time series, False to only export the
            TimeSeries object"s name.

        Returns
        -------
        data: dict
            dictionary with the necessary information to recreate the TimeSeries
            object completely.
        """
        data = {}

        if series is True or series == "original":
            data["series"] = self.series_original
        elif series == "modified":
            data["series"] = self

        data["name"] = self.name
        data["settings"] = self.settings
        data["metadata"] = self.metadata

        return data

    def plot(self, original: bool = False, **kwargs) -> Axes:
        raise DeprecationWarning(
            "The plot method is deprecated since 0.23 and will be removed in Pastas "
            "1.0. Use the series.plot function from Pandas instead."
        )


def check_deprecated_input(series, settings, **kwargs):
    """Method to check input data for Pastas version 0.23 and raise errors.

    Parameters
    ----------
    series: pandas.Series
    settings: dict

    """
    if "freq_original" in kwargs.keys():
        raise DeprecationWarning(
            "Freq_original is no longer supported. Please provide an equidistant "
            "time series."
        )

    if isinstance(series, TimeSeries):
        raise DeprecationWarning(
            "TimeSeries are no longer allowed as input for to create new "
            "TimeSeries objects. Please use the original pandas.Series object "
            "and provide the settings and name."
        )

    if isinstance(settings, dict):
        for key in settings.keys():
            if key in ["norm"]:
                raise DeprecationWarning(
                    "Key %s is no longer supported. Please remove this keyword from "
                    "the settings dictionary."
                )


def validate_stress(series: Series):
    """Method to validate user-provided stress input time series.

    Parameters
    ----------
    series: pandas.Series
        Pandas.Series object containing the series time series.

    Notes
    -----
    The Series are validated for the following cases:

    0. Make sure the series is a Pandas.Series
    1. Make sure the values are floats
    2. Make sure the index is a DatetimeIndex
    3. Make sure the indices are datetime64
    4. Make sure the index is monotonically increasing
    5. Make sure there are no duplicate indices
    6. Make sure the time series has no nan-values
    7. Make sure the time series has equidistant time steps

    If any of these checks are not passed the method will throw an error that needs
    to be fixed by the user.

    Examples
    --------

    >>> ps.validate_stress(series)

    """
    _validate_series(series, equidistant=True)


def validate_oseries(series: Series):
    """Method to validate user-provided oseries input time series.

    Parameters
    ----------
    series: pandas.Series
        Pandas.Series object containing the series time series.

    Notes
    -----
    The Series are validated for the following cases:

    0. Make sure the series is a Pandas.Series
    1. Make sure the values are floats
    2. Make sure the index is a DatetimeIndex
    3. Make sure the indices are datetime64
    4. Make sure the index is monotonically increasing
    5. Make sure there are no duplicate indices
    6. Make sure the time series has no nan-values

    If any of these checks are not passed the method will throw an error that needs
    to be fixed by the user.

    Examples
    --------

    >>> ps.validate_oseries(series)

    """
    _validate_series(series, equidistant=False)


def _validate_series(series: Series, equidistant: bool = True):
    """Internal method to validate user-provided input time series.

    Parameters
    ----------
    series: pandas.Series
        Pandas.Series object containing the series time series.
    equidistant: bool, optional
        Whether the time series should have equidistant time step or not.

    Notes
    -----
    If any of these checks are not passed the method will throw an error that needs
    to be fixed by the user.

    """
    # Because we are friendly and allow 1D DataFrames
    if isinstance(series, pd.DataFrame):
        if len(series.columns) == 1:
            series = series.iloc[:, 0]

    # 0. Make sure it is a Series and not something else (e.g., DataFrame)
    if not isinstance(series, pd.Series):
        msg = f"Expected a Pandas Series, got {type(series)}"
        logger.error(msg)
        raise ValueError(msg)

    name = series.name  # Only Series have a name, DateFrame do not

    # 1. Make sure the values are floats
    if not pd.api.types.is_float_dtype(series):
        msg = f"Values of time series {name} are not dtype=float."
        logger.error(msg)
        raise ValueError(msg)

    # 2. Make sure the index is a DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        msg = f"Index of series {name} is not a pandas.DatetimeIndex."
        logger.error(msg)
        raise ValueError(msg)

    # 3. Make sure the indices are datetime64
    if not pd.api.types.is_datetime64_dtype(series.index):
        msg = f"Indices os series {name} are not datetime64."
        logger.error(msg)
        raise ValueError(msg)

    # 4. Make sure the index is monotonically increasing
    if not series.index.is_monotonic_increasing:
        msg = (
            f"The time-indices of series {name} are not monotonically increasing. Try "
            f"to use `series.sort_index()` to fix it."
        )
        logger.error(msg)
        raise ValueError(msg)

    # 5. Make sure there are no duplicate indices
    if not series.index.is_unique:
        msg = (
            f"duplicate time-indexes were found in the time series {name}. Make sure "
            f"there are no duplicate indices. For example by "
            f"`grouped = series.groupby(level=0); series = grouped.mean()`"
        )
        logger.error(msg)
        raise ValueError(msg)

    # 6. Make sure the time series has no nan-values
    if series.hasnans:
        msg = (
            "The time series %s has nan-values. Pastas will use the fill_nan "
            "settings to fill up the nan-values."
        )
        logger.warning(msg, name)

    # 7. Make sure the time series has equidistant time steps
    if equidistant:
        if not pd.infer_freq(series.index):
            msg = (
                f"The frequency of the index of time series {name} could not be "
                f"inferred. Please provide a time series with a regular time step."
            )
            logger.error(msg)
            raise ValueError(msg)
