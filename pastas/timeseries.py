from logging import getLogger

import pandas as pd
from pandas.tseries.frequencies import to_offset

from .utils import _get_stress_dt, _get_dt, _get_time_offset, \
    timestep_weighted_resample

logger = getLogger(__name__)


class TimeSeries:
    """Class that deals with all user-provided time series.

    Parameters
    ----------
    series: pandas.Series or pastas.timeseries.TimeSeries
        Pandas Series with time indices and values or a Pastas.TimeSeries
        instance. If the latter is provided, a new TimeSeries.
    name: str, optional
        String with the name of the time series, if None is provided,
        pastas will try to derive the name from the series.
    settings: str or dict, optional
        String with the name of one of the predefined settings (obtained
        through ps.TimeSeries._predefined_settings.) or a dictionary with the
        settings to be applied. This does not have to include all the
        settings arguments.
    metadata: dict, optional
        Dictionary with metadata of the time series.
    freq_original: str, optional
        By providing a frequency string here, a frequency can be forced on the
        time series if it can not be inferred with pd.infer_freq.
    **kwargs: optional
        Any keyword arguments that are provided but are not listed will be
        passed as additional settings.

    Returns
    -------
    series: pastas.timeseries.TimeSeries
        Returns a pastas.TimeSeries object.

    Examples
    --------
    To obtain the predefined TimeSeries settings, you can run the following
    line of code:

    >>> ps.TimeSeries._predefined_settings

    See Also
    --------
    pastas.timeseries.TimeSeries.update_series
        For the individual options for the different settings.

    """
    _predefined_settings = {
        "oseries": {"fill_nan": "drop", "sample_down": "drop"},
        "prec": {"sample_up": "bfill", "sample_down": "mean",
                 "fill_nan": 0.0, "fill_before": "mean", "fill_after": "mean"},
        "evap": {"sample_up": "bfill", "sample_down": "mean",
                 "fill_before": "mean", "fill_after": "mean",
                 "fill_nan": "interpolate"},
        "well": {"sample_up": "bfill", "sample_down": "mean",
                 "fill_nan": 0.0, "fill_before": 0.0, "fill_after": 0.0},
        "waterlevel": {"sample_up": "interpolate", "sample_down": "mean",
                       "fill_before": "mean", "fill_after": "mean",
                       "fill_nan": "interpolate"},
        "level": {"sample_up": "interpolate", "sample_down": "mean",
                  "fill_before": "mean", "fill_after": "mean",
                  "fill_nan": "interpolate"},
        "flux": {"sample_up": "bfill", "sample_down": "mean",
                 "fill_before": "mean", "fill_after": "mean",
                 "fill_nan": 0.0},
        "quantity": {"sample_up": "divide", "sample_down": "sum",
                     "fill_before": "mean", "fill_after": "mean",
                     "fill_nan": 0.0},
    }

    def __init__(self, series, name=None, settings=None, metadata=None,
                 freq_original=None, **kwargs):
        if isinstance(series, TimeSeries):
            # Copy all the series
            self._series_original = series.series_original.copy()
            self._series_validated = series.series_validated.copy()
            self._series = series.series.copy()
            # Copy all the properties
            self.freq_original = series.freq_original
            self.settings = series.settings.copy()
            self.metadata = series.metadata.copy()

            validate = False
            update = False

            if settings is None:
                settings = self.settings.copy()
        else:
            # Make sure we have a Pandas Series and not a 1D-DataFrame
            if isinstance(series, pd.DataFrame):
                if len(series.columns) == 1:
                    series = series.iloc[:, 0]
            elif not isinstance(series, pd.Series):
                msg = f"Expected a Pandas Series, got {type(series)}"
                raise TypeError(msg)

            validate = True
            update = True
            # Store a copy of the original series
            self._series_original = series.copy()

            self.freq_original = freq_original
            self.settings = {
                "freq": None,
                "sample_up": None,
                "sample_down": None,
                "fill_nan": "interpolate",
                "fill_before": None,
                "fill_after": None,
                "tmin": None,
                "tmax": None,
                "norm": None,
                "time_offset": pd.Timedelta(0)
            }
            self.metadata = {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "projection": None
            }

        # Use user provided name or set from series
        if name is None:
            name = series.name
        self.name = name
        self._series_original.name = name

        if metadata is not None:
            self.metadata.update(metadata)

        # Update the settings with user-provided values, if any.
        if settings:
            if isinstance(settings, str):
                if settings in self._predefined_settings.keys():
                    settings = self._predefined_settings[settings]
                else:
                    error = f"Settings shortcut code '{settings}' is not in " \
                            f"the predefined settings options. Please " \
                            f"choose from {self._predefined_settings.keys()}"
                    raise KeyError(error)
            if self._update_settings(**settings):
                update = True
        if kwargs:
            if self._update_settings(**kwargs):
                update = True

        # Create a validated series for computations and update
        if validate:
            self._series_validated = self._validate_series(
                self._series_original)
        if update:
            self.update_series(force_update=True, **self.settings)

    def __repr__(self):
        """Prints a simple string representation of the time series.
        """
        return f"{self.__class__.__name__}" \
               f"(name={self.name}, " \
               f"freq={self.settings['freq']}, " \
               f"freq_original={self.freq_original}, " \
               f"tmin={self.settings['tmin']}, " \
               f"tmax={self.settings['tmax']})"

    @property
    def series_original(self):
        return self._series_original

    @series_original.setter
    def series_original(self, series):
        """Sets a new freq_original for the TimeSeries"""
        if not isinstance(series, pd.Series):
            raise TypeError("Expected a Pandas Series, got {}".format(
                type(series)))
        else:
            self._series_original = series
            # make sure that tmin and tmax and freq_original are set in validate_series
            self.settings["tmin"] = None
            self.settings["tmax"] = None
            freq_original = self.freq_original  # remember what it was
            self.freq_original = None
            self._series_validated = self._validate_series(
                self._series_original)
            if self.freq_original is None:
                self.freq_original = freq_original
            self.update_series(force_update=True, **self.settings)

    @property
    def series(self):
        return self._series

    @series.setter
    def series(self, value):
        raise AttributeError("You cannot set series by yourself, as it is "
                             "calculated from series_original. Please set "
                             "series_original to update the series.")

    @property
    def series_validated(self):
        return self._series_validated

    @series_validated.setter
    def series_validated(self, value):
        raise AttributeError("You cannot set series_validated by yourself, as"
                             " it is calculated from series_original. Please"
                             " set series_original to update the series.")

    def update_series(self, force_update=False, **kwargs):
        """Method to update the series with new options.

        Parameters
        ----------
        force_update: bool, optional
            argument that is used to force an update, even when no changes
            are found. Internally used by the __init__ method. Default is
            False.
        freq: str, optional
            String representing the desired frequency of the time series. Must
            be one of the following: (D, h, m, s, ms, us, ns) or a multiple of
            that e.g. "7D".
        sample_up: str or float, optional
            String with the method to use when the frequency is increased (
            e.g. Weekly to daily). Possible values are: "backfill", "bfill",
            "pad", "ffill", "mean", "interpolate", "divide" or a float value
            to fill the gaps.
        sample_down: str, optional
            String with the method to use when the frequency decreases
            (e.g. from daily to weekly values). Possible values are: "mean",
            "drop", "sum", "min", "max".
        fill_nan: str or float, optional
            Method to use when there ar nan-values in the time series.
            Possible values are: "mean", "drop", "interpolate" (default) or a
            float value.
        fill_before: str or float, optional
            Method used to extend a time series before any measurements are
            available. possible values are: "mean" or a float value.
        fill_after: str or float, optional
            Method used to extend a time series after any measurements are
            available. Possible values are: "mean" or a float value.
        tmin: str or pandas.TimeStamp, optional
            String that can be converted to, or a Pandas TimeStamp with the
            minimum time of the series.
        tmax: str or pandas.TimeStamp, optional
            String that can be converted to, or a Pandas TimeStamp with the
            maximum time of the series.
        norm: str or float, optional
            String with the method to normalize the time series with.
            Possible values are: "mean" or "median", "min", "max" or a float
            value.

        Notes
        -----
        The method will validate if any of the settings is changed to
        determine if the series need to be updated.

        """
        if self._update_settings(**kwargs) or force_update:
            tmin = self.settings['tmin']
            freq = self.settings['freq']
            if tmin is not None and freq is not None:
                self.settings['time_offset'] = _get_time_offset(tmin, freq)

            # Get the validated series to start with
            series = self._series_validated.copy(deep=True)

            # Update the series with the new settings
            series = self._change_frequency(series)
            series = self._fill_before(series)
            series = self._fill_after(series)
            series = self._normalize(series)
            series.name = self._series_original.name

            self._series = series

    def multiply(self, other):
        """Method to multiply the original time series.

        Parameters
        ----------
        other: float or pandas.Series

        """
        self._series = self.series.multiply(other)
        self._series_original = self.series_original.multiply(other)
        self.update_series(force_update=True)

    def _validate_series(self, series):
        """ Validate user provided time series.

        Parameters
        ----------
        series: pandas.Series
            Pandas.series object containing the series time series.

        Returns
        -------
        series: pandas.Series
            The validated series as pd.Series

        Notes
        -----
        The Series are validated for the following cases:

        1. Make sure the values are floats
        2. Make sure the index is a datetimeindex
        3. Make sure the index is increasing (also works for irregular dt)
        4. Drop nan-values at the beginning and end of the time series
        5. Find the frequency of the time series
        6. Handle duplicate indices, average if they exist
        7. drop nan-values (info message is provided by _fill_nan method)

        """

        # 1. Make sure the values are floats
        if not pd.api.types.is_float_dtype(series):
            series = series.astype(float)
            logger.info(f"Time series {self.name} updated to dtype float.")

        # 2. Make sure the index is a datetimeindex
        if not pd.api.types.is_datetime64_dtype(series.index):
            series.index = pd.to_datetime(series.index)
            logger.info(f"Time series index for {self.name} updated to "
                        f"dtype datetime64.")

        # 3. Make sure the index is increasing (also works for irregular dt)
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()
            logger.info(f"Time series index for {self.name} sorted to have "
                        f"time increasing.")

        # 4. Drop nan-values at the beginning and end of the time series
        if series.first_valid_index() != series.index[0]:
            series = series.loc[series.first_valid_index():].copy(deep=True)
            logger.info(f"Nan-values were removed at the start of the "
                        f"time series {self.name}.")

        if series.last_valid_index() != series.index[-1]:
            series = series.loc[:series.last_valid_index()].copy(deep=True)
            logger.info(f"Nan-values were removed at the end of the "
                        f"time series {self.name}.")

        # 5. Find the frequency of the time series
        if self.freq_original:
            msg = f"User provided frequency for time series {self.name}: " \
                  f"freq={self.freq_original}"
        elif pd.infer_freq(series.index):
            self.freq_original = pd.infer_freq(series.index)
            msg = f"Inferred frequency for time series {self.name}: " \
                  f"freq={self.freq_original}"
        elif self.settings["fill_nan"] != "drop":
            msg = f"Cannot determine frequency of series {self.name}: " \
                  f"freq=None. Resample settings are ignored and " \
                  f"timestep_weighted_resample is used."
        else:
            msg = f"Cannot determine frequency of series {self.name}: " \
                  f"freq=None. The time series is irregular."

        logger.info(msg)  # Always report a message for the frequency

        # 6. Handle duplicate indices
        if not series.index.is_unique:
            msg = f"duplicate time-indexes were found in the Time Series " \
                  f"{self.name}. Values were averaged."
            logger.warning(msg)
            grouped = series.groupby(level=0)
            series = grouped.mean()

        # 7. drop or fill up nan-values (info message is provided by
        # _fill_nan method)
        series = self._fill_nan(series)

        if self.settings["tmin"] is None:
            self.settings["tmin"] = series.index.min()
        if self.settings["tmax"] is None:
            self.settings["tmax"] = series.index.max()

        series.index.name = ""

        return series

    def _update_settings(self, **kwargs):
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
            if value != self.settings[key]:
                self.settings[key] = value
                update = True
        return update

    def _change_frequency(self, series):
        """Method to change the frequency of the time series.

        """
        freq = self.settings["freq"]

        # 1. If no freq string is present or is provided (e.g. Oseries)
        if not freq:
            return series
        # 2. If original frequency could not be determined
        elif not self.freq_original:
            series = self._sample_weighted(series)
        else:
            dt_new = _get_dt(freq)
            dt_org = _get_stress_dt(self.freq_original)
            # 3. If new and original frequency are not a multiple of each other
            eps = 1e-10
            if not ((dt_new % dt_org) < eps or (dt_org % dt_new) < eps):
                series = self._sample_weighted(series)
            # 4. If new frequency is lower than its original
            elif dt_new < dt_org:
                series = self._sample_up(series)
            # 5. If new frequency is higher than its original
            elif dt_new > dt_org:
                series = self._sample_down(series)

        # Drop nan-values at the beginning and end of the time series
        series = series.loc[
                 series.first_valid_index():series.last_valid_index()]

        return series

    def _sample_up(self, series):
        """Resample the time series when the frequency increases (e.g. from
        weekly to daily values).

        """
        method = self.settings["sample_up"]
        freq = self.settings["freq"]

        if method in ["backfill", "bfill", "pad", "ffill"]:
            series = series.asfreq(freq, method=method)
        elif method is None:
            pass
        else:
            if method == "mean":
                series = series.asfreq(freq)
                series.fillna(series.mean(), inplace=True)
            elif method == "interpolate":
                series = series.asfreq(freq)
                series.interpolate(method="time", inplace=True)
            elif method == "divide":
                dt = series.index.to_series().diff() / to_offset(freq).delta
                series = series / dt
                series = series.asfreq(freq, method="bfill")
            elif isinstance(method, float):
                series = series.asfreq(freq)
                series.fillna(method, inplace=True)
            else:
                msg = f"Time Series {self.name}: User-defined option for " \
                      f"sample_up {method} is not supported"
                logger.warning(msg)

        msg = f"Time Series {self.name} were sampled up using {method}."
        logger.info(msg)

        return series

    def _sample_down(self, series):
        """Resample the time series when the frequency decreases (e.g. from
        daily to weekly values).

        Notes
        -----
        make sure the labels are still at the end of each period, and
        data at the right side of the bucket is included (see
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
        if self.settings['time_offset'] > pd.Timedelta(0):
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
            msg = f"Time Series {self.name}: User-defined option for " \
                  f"sample_down {method} is not supported"
            logger.warning(msg)

        # TODO: replace by adding offset to resample method with pandas 1.1.0
        if self.settings['time_offset'] > pd.Timedelta(0):
            # The offset is removed by the resample-method, so we add it again
            series = series.shift(1, freq=self.settings["time_offset"])

        logger.info(f"Time Series {self.name} was sampled down to freq "
                    f"{freq} with method {method}.")

        return series

    def _sample_weighted(self, series):
        freq = self.settings["freq"]
        time_offset = self.settings['time_offset']
        tindex = pd.date_range(series.index[0].ceil(freq) + time_offset,
                               series.index[-1], freq=freq)
        series = timestep_weighted_resample(series, tindex)
        msg = f"Time Series {self.name} was sampled down to freq {freq} " \
              f"with method timestep_weighted_resample."
        logger.info(msg)
        return series

    def _fill_nan(self, series):
        """Fill up the nan-values when present and a constant frequency is
        required.

        """

        method = self.settings["fill_nan"]
        freq = self.freq_original

        if freq:
            series = series.asfreq(freq)
            n = series.isnull().values.sum()
            if n is 0:
                pass
            elif method == "drop":
                series.dropna(inplace=True)
            elif method == "mean":
                series.fillna(series.mean(), inplace=True)
            elif method == "interpolate":
                series.interpolate(method="time", inplace=True)
            elif isinstance(method, float):
                series.fillna(method, inplace=True)
            else:
                msg = f"Time Series {self.name}: User-defined option for " \
                      f"fill_nan {method} is not supported."
                logger.warning(msg)
        else:
            method = "drop"
            n = series.isnull().values.sum()
            series.dropna(inplace=True)
        if n > 0:
            logger.info(f"Time Series {self.name}: {n} nan-value(s) was/were "
                        f"found and filled with: {method}.")

        return series

    def _fill_before(self, series):
        """Method to add a period in front of the available time series.

        """
        freq = self.settings["freq"]
        method = self.settings["fill_before"]
        tmin = self.settings["tmin"]

        if tmin is None or method is None:
            pass
        elif pd.Timestamp(tmin) >= series.index.min():
            series = series.loc[pd.Timestamp(tmin):]
        else:
            index_extend = pd.date_range(start=pd.Timestamp(tmin),
                                         end=series.index.min(), freq=freq)
            series = series.reindex(series.index.union(index_extend[:-1]))

            if method == "mean":
                series.fillna(series.mean(), inplace=True)  # Default option
                msg = f"Time Series {self.name} was extended to " \
                      f"{series.index.min()} with the mean value of the " \
                      f"time series."
            elif isinstance(method, float):
                series.fillna(method, inplace=True)
                msg = f"Time Series {self.name} was extended to" \
                      f" {series.index.min()} by adding {method} values."
            else:
                msg = f"Time Series {self.name}: User-defined option for " \
                      f"fill_before {method} is not supported."
            logger.info(msg)

        return series

    def _fill_after(self, series):
        """Method to add a period in front of the available time series.

        """
        freq = self.settings["freq"]
        method = self.settings["fill_after"]
        tmax = self.settings["tmax"]

        if tmax is None or method is None:
            pass
        elif pd.Timestamp(tmax) <= series.index.max():
            series = series.loc[:pd.Timestamp(tmax)]
        else:
            index_extend = pd.date_range(start=series.index.max(),
                                         end=pd.Timestamp(tmax), freq=freq)
            series = series.reindex(series.index.union(index_extend))

            if method == "mean":
                series.fillna(series.mean(), inplace=True)  # Default option
                msg = f"Time Series {self.name} was extended to " \
                      f"{series.index.max()} with the mean value of the " \
                      f"time series."
            elif isinstance(method, float):
                series.fillna(method, inplace=True)
                msg = f"Time Series {self.name} was extended to " \
                      f"{series.index.max()} by adding {method} values."
            else:
                msg = f"Time Series {self.name}: User-defined option for " \
                      f"fill_after {method} is not supported"
            logger.info(msg)

        return series

    def _normalize(self, series):
        """Method to normalize the time series.

        """
        method = self.settings["norm"]
        msg = f"Time series {self.name} is normalized with the {method}."

        if method is None:
            msg = None
        elif method == "mean":
            series = series.subtract(series.mean())
        elif method == "median":
            series = series.subtract(series.median())
        elif method == "min":
            series = series.subtract(series.min())
        elif method == "max":
            series = series.subtract(series.max())
        elif isinstance(method, float):
            series = series.subtract(method)
        else:
            msg = f"Time Series {self.name}: Selected method {method} to " \
                  f"normalize the time series is not supported"
        if msg:
            logger.info(msg)

        return series

    def to_dict(self, series=True):
        """Method to export the Time Series to a json format.

        Parameters
        ----------
        series: bool, optional
            True to export the original time series, False to only export
            the TimeSeries object"s name.

        Returns
        -------
        data: dict
            dictionary with the necessary information to recreate the
            TimeSeries object completely.

        """
        data = {}

        if series is True or series == "original":
            data["series"] = self.series_original
        elif series == "modified":
            data["series"] = self

        data["name"] = self.name
        data["settings"] = self.settings
        data["metadata"] = self.metadata
        data["freq_original"] = self.freq_original

        return data

    def plot(self, original=False, **kwargs):
        """Method to plot the TimeSeries object. Plots the edited series by
        default.

        Parameters
        ----------
        original: bool, optional
            Also plot the original series.

        Returns
        -------
        matplotlib.Axes

        """

        if original:
            ax = self.series_original.plot()
        else:
            ax = self.series.plot(**kwargs)
        return ax
