"""
This file contains a class that holds the TimeSeries class. This class is used
to "manage" the time series within PASTAS. It has methods to change a time
series in frequency and extend the time series, without losing the original
data.

August 2017, R.A. Collenteur

"""
from __future__ import print_function, division

from warnings import warn

import pandas as pd

from pastas.utils import get_dt


class TimeSeries(pd.Series):
    _type_options = {
        "oseries": {"freq": None, "sample_up": None, "sample_down": None,
                    "fill_nan": "drop", "fill_before": None, "fill_after":
                        None},
        "prec": {"freq": "D", "sample_up": "mean", "sample_down": "sum",
                 "fill_nan": 0.0, "fill_before": None, "fill_after": None},
        "evap": {"freq": "D", "sample_up": "interpolate", "sample_down": "sum",
                 "fill_nan": "interpolate", "fill_before": None, "fill_after":
                     None},
        "well": {"freq": "D", "sample_up": None, "sample_down": "sum",
                 "fill_nan": None, "fill_before": None, "fill_after": None},
        "none": {"freq": "D", "sample_up": None, "sample_down": None,
                 "fill_nan": None, "fill_before": None, "fill_after": None},
    }

    def __init__(self, stress, name=None, type=None, **kwargs):
        """Class that supports or user-provided time series within PASTAS.

        Parameters
        ----------
        stress: pandas.Series
            original stress series, which will be stored
        name: str
            string with the name for this stress
        type: str
            string with the type of the stress, to autocomplete the
            following keywords. The user can choose from: oseries, evap,
            prec, well.
        freq: str
            String containing the desired frequency. The required string format
             is found at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        sample_up: optional: str or float
            Methods or float number to fill nan-values. Default values is
            'mean'. Currently supported options are: 'interpolate', float,
            and 'mean'. Interpolation is performed with a standard linear
            interpolation.
        sample_down: str or float
            method

        fill_before
        fill_after

        """
        pd.Series.__init__(self)

        # Store a copy of the original stress
        self.stress_original = stress.copy()
        self.freq_original = None
        self.name = name

        # Options when creating the series
        self.type = type
        if type in self._type_options.keys():
            self.options = self._type_options[type]
        else:
            self.options = self._type_options["none"]

        # Update the options with user-provided values, if any.
        if kwargs:
            self.options.update(kwargs)

        # Create a validated stress for computations
        self.stress = self.validate_stress(stress)

        # Finally, update the TimeSeries so it is ready for simulation
        self.update_stress(**self.options)

    def validate_stress(self, stress):
        """ This method performs some PASTAS specific tests for the TimeSeries.

        Parameters
        ----------
        stress: pd.Series
            Pandas series object containing the stress time series.

        Returns
        -------
        stress: pandas.Series
            The validated stress as pd.Series

        Notes
        -----
        The Series are validated for the following cases:

            1. Series is an actual pandas Series;
            2. Nan-values from begin and end are removed;
            3. Nan-values between observations are removed;
            4. Indices are in Timestamps (standard throughout PASTAS);
            5. Duplicate indices are removed (by averaging).

        """

        # 1. Check if stress is a Pandas Series
        assert isinstance(stress, pd.Series), 'Expected a Pandas Series, ' \
                                              'got %s' % type(stress)

        # 2. Drop nan-values at the beginning and end of the time series
        stress = stress.loc[stress.first_valid_index():stress.last_valid_index(
        )].copy(deep=True)

        # 4. Make sure the indices are Timestamps
        stress.index = pd.to_datetime(stress.index)

        # 3. Find the frequency of the original series
        freq = pd.infer_freq(stress.index)

        if freq:
            self.freq_original = freq
            if not self.options["freq"]:
                self.options["freq"] = freq
            print('Inferred frequency from time series %s: freq=%s ' % (
                self.name, freq))
        else:
            self.freq_original = self.options["freq"]
            if self.options["fill_nan"] and self.options["fill_nan"] != \
                    "drop":
                warn("User-provided frequency is applied when validating the "
                     "Time Series %s. Make sure the provided frequency is "
                     "close to the real frequency of the original series." %
                     (self.name))

        # 3. drop nan-values
        if stress.hasnans:
            stress = self.fill_nan(stress)

        # 5. Handle duplicate indices
        if not stress.index.is_unique:
            print('duplicate time-indexes were found in the Time Series %s. '
                  'Values were averaged.' % (self.name))
            grouped = stress.groupby(level=0)
            stress = grouped.mean()

        return stress

    def update_stress(self, **kwargs):
        """Method to update the stress with new options, but most likely
        only a change in the frequency before solving a PASTAS model.

        Parameters
        ----------
        kwargs: dict
            dictionary with the keyword arguments that are updated. Possible
            arguments are: "freq", "sample_up", "sample_down",
                 "fill_before" and "fill_after".

        """

        # Get the validated stress to start with
        stress = self.stress

        if kwargs:
            # Update the options with any provided arguments
            self.options.update(kwargs)

            # Update the stress with the new settings
            stress = self.change_frequency(stress)

            if "start" in kwargs:
                stress = self.fill_before(stress)
            if "end" in kwargs:
                stress = self.fill_after(stress)

            self._update_inplace(stress)

    def change_frequency(self, stress):
        """Method to change the frequency of the time series.

        Parameters
        ----------
        freq: str
            String containing the desired frequency. The required string format
             is found at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

        Returns
        -------

        """

        freq = self.options["freq"]

        # 1. If no freq string is present or is provided (e.g. Oseries)
        if not self.freq_original and not freq:
            pass

        # 2. If new frequency is lower than its original.
        elif get_dt(freq) < get_dt(self.freq_original):
            stress = self.sample_up(stress)

        # 3. If new frequency is higher than its original, downsample.
        elif get_dt(freq) > get_dt(self.freq_original):
            stress = self.sample_down(stress)

        # 4. If new frequency is equal to its original.
        elif get_dt(freq) == get_dt(self.freq_original):
            stress = self.sample_up(stress)
        else:
            stress = self.stress

        # Drop nan-values at the beginning and end of the time series
        stress = stress.loc[
                 stress.first_valid_index():stress.last_valid_index()]

        return stress

    def sample_up(self, stress):
        """Resample the time series when the frequency increases (e.g. from
        weekly to daily values).

        Parameters
        ----------
        stress

        Returns
        -------

        """
        method = self.options["sample_up"]
        freq = self.options["freq"]

        if method in ['backfill', 'bfill', 'pad', 'ffill']:
            stress = stress.asfreq(freq, method=method)
        else:
            stress = stress.asfreq(freq)
            if method == 'mean':
                stress.fillna(stress.mean(), inplace=True)  # Default option
            elif method == 'interpolate':
                stress.interpolate(method='time', inplace=True)
            elif type(method) == float:
                stress.fillna(method, inplace=True)
            else:
                warn('User-defined option for sample_up %s is not '
                     'supported' % method)

        print('%i nan-value(s) was/were found and filled with: %s'
              % (stress.isnull().values.sum(), method))

        return stress

    def sample_down(self, stress):
        """Resample the time series when the frequency decreases (e.g. from
        daily to weekly values).

        Returns
        -------

        Notes
        -----

        # make sure the labels are still at the end of each period, and
        # data at the right side of the bucket is included (see
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.resample.html)

        """
        method = self.options["sample_down"]
        freq = self.options["freq"]

        # Provide some standard pandas arguments for all options
        kwargs = {"label": 'right', "closed": 'right'}

        if method == "mean":
            stress = stress.resample(freq, **kwargs).mean()
        elif method == "drop":
            stress = stress.resample(freq, **kwargs).dropna()
        elif method == "sum":
            stress = stress.resample(freq, **kwargs).sum()
        elif method == "min":
            stress = stress.resample(freq, **kwargs).min()
        elif method == "max":
            stress = stress.resample(freq, **kwargs).max()
        else:
            warn('User-defined option for sample_down %s is not '
                 'supported' % method)

        print("Time Series %s were sampled down to freq %s with method %s" %
              (self.name, freq, method))

        return stress

    def fill_nan(self, stress):
        """Fill up the nan-values when present and a constant frequency is
        required.

        Parameters
        ----------
        stress: pandas.Series
            stress series with nan-values

        Returns
        -------
        stress: pandas.Series
            stress series with the nan-values filled up.

        """

        method = self.options["fill_nan"]
        freq = self.freq_original

        if freq:
            stress = stress.asfreq(freq)

            if method == "drop":
                stress.dropna(inplace=True)
            elif method == 'mean':
                stress.fillna(stress.mean(), inplace=True)  # Default option
            elif method == 'interpolate':
                stress.interpolate(method='time', inplace=True)
            elif type(method) == float:
                stress.fillna(method, inplace=True)
            else:
                warn('User-defined option for sample_up %s is not '
                     'supported' % method)
        else:
            stress.dropna(inplace=True)

        print('%i nan-value(s) was/were found and filled with: %s'
              % (stress.isnull().values.sum(), method))

        return stress

    def fill_before(self, stress):
        """Method to add a period in front of the available time series

        Parameters
        ----------
        method
        period

        Returns
        -------

        """

        # method = self.options["fill_before"]
        freq = self.options["freq"]

        # If no index is provided, use the fill_before_period
        if not index:
            period = self.options["fill_before_period"]
            index_extend = pd.date_range(end=self.index.min() - 1,
                                         periods=period, freq=freq)
            index = self.index.union(index_extend)

        self.index.reindex(index)

        return stress

    def fill_after(self, stress):
        return stress

    def normalize_series(self):
        """

        Returns
        -------

        """
        return None

    def change_index(self, index):
        # reindex the original
        stress = self.reindex(index)

        # add data before and after this period, if needed
        if stress.isnull().values.any():
            if isinstance(stress, pd.Series):
                fillna_value = pd.Series(index=stress.index)
            else:
                fillna_value = pd.DataFrame(index=index,
                                            columns=self.stress.columns)
            if isinstance(self.fill_before, list) and len(
                    self.fill_before) > 1:
                for ind, column in enumerate(fillna_value.columns):
                    fillna_value.loc[:self.stress.first_valid_index(),
                    column] = self.fill_before[ind]
            else:
                fillna_value[
                :self.stress.first_valid_index()] = self.fill_before
            if isinstance(self.fill_after, list) and len(self.fill_after):
                for ind, column in enumerate(fillna_value.columns):
                    fillna_value.loc[self.stress.last_valid_index():, column] = \
                        self.fill_after[ind]
            else:
                fillna_value[self.stress.last_valid_index():] = self.fill_after
            stress = stress.fillna(fillna_value)

        # save result
        self._update_inplace(stress)

    def export(self):
        """Method to export the Time Series to a json format.

        Returns
        -------

        """
        data = dict()
        data["stress"] = self.stress_original.to_json(date_format='iso')
        data["options"] = self.options
        data["name"] = self.name
        data["type"] = self.type

        return data
