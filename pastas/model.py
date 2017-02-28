from __future__ import print_function, division

import datetime
from collections import OrderedDict
from warnings import warn

import numpy as np
import pandas as pd
from scipy import interpolate

from .checks import check_oseries
from .objfunc import residuals as objf_residuals
from .solver import LmfitSolve
from .stats import Statistics
from .tseries import Constant
from .plots import Plotting


class Model:
    def __init__(self, oseries, xy=(0, 0), name="TSA_Model", metadata=None,
                 warmup=0, fillnan='drop', constant=True):
        """Initiates a time series model.

        Parameters
        ----------
        oseries: pd.Series
            pandas Series object containing the dependent time series. The
            observation can be non-equidistant.
        xy: Optional[tuple]
            XY location of the oseries in lat-lon format.
        metadata: Optional[dict]
            Dictionary containing metadata of the model.
        warmup: Optional[float]
            Number of days used for warmup
        fillnan: Optional[str or float]
            Methods or float number to fill nan-values. Default values is
            'drop'. Currently supported options are: 'interpolate', float,
            'mean' and, 'drop'. Interpolation is performed with a standard
            linear interpolation.
        constant: Boolean
            Add a constant to the model (Default=True).

        """
        self.oseries = check_oseries(oseries, fillnan)
        self.oseries_calib = None

        # Min and max time of the model
        self.tmin = None
        self.tmax = None
        self.odelt = self.oseries.index.to_series().diff() / \
                     np.timedelta64(1, 'D')

        self.warmup = warmup
        self.freq = None
        self.time_offset = pd.to_timedelta(0)

        self.parameters = pd.DataFrame(
            columns=['initial', 'name', 'optimal', 'pmin', 'pmax', 'vary'])
        self.nparam = 0

        self.tseriesdict = OrderedDict()
        self.noisemodel = None

        if constant:
            self.add_constant()
        else:
            self.constant = None

        # Metadata
        self.xy = xy
        self.metadata = metadata
        self.name = name

        self.fit = None
        self.report = "Model has not been solved yet. "

        # Load other modules
        self.stats = Statistics(self)
        self.plots = Plotting(self)

    def add_tseries(self, tseries):
        """Adds a time series component to the model.

        Parameters
        ----------
        name: str
            string with the name of the tseries object.

        Notes
        -----
        To obtain a list of the tseries names type:
        >>> ml.tseriesdict.keys()

        """
        if tseries.name in self.tseriesdict.keys():
            warn('The name for the series you are trying to add '
                 'already exists for this model. Select another '
                 'name.')
        else:
            self.tseriesdict[tseries.name] = tseries
            self.parameters = self.get_init_parameters()
            self.nparam += tseries.nparam

            # Call these methods to set tmin, tmax and freq and enable
            # simulation.
            self.set_freq_offset()
            self.tmin, self.tmax = self.get_tmin_tmax()

    def add_noisemodel(self, noisemodel):
        """Adds a noise model to the time series Model.

        """
        self.noisemodel = noisemodel
        self.parameters = self.get_init_parameters()
        self.nparam += noisemodel.nparam

    def add_constant(self):
        """Adds a Constant to the time series Model.

        """
        self.constant = Constant(value=self.oseries.mean(), name='constant')
        self.parameters = self.get_init_parameters()
        self.nparam += self.constant.nparam

    def del_tseries(self, name):
        """ Save deletion of a tseries from the tseriesdict.

        Parameters
        ----------
        name: str
            string with the name of the tseries object.

        Notes
        -----
        To obtain a list of the tseries names type:
        >>> ml.tseriesdict.keys()

        """
        if name not in self.tseriesdict.keys():
            warn(message='The tseries name you provided is not in the '
                         'tseriesdict. Please select from the following list: '
                         '%s' % self.tseriesdict.keys())
        else:
            self.nparam -= self.tseriesdict[name].nparam
            self.parameters = self.parameters.ix[self.parameters.name != name]
            self.tseriesdict.pop(name)

    def del_constant(self):
        """ Save deletion of the constant from a Model.

        """
        if self.constant is None:
            warn("No constant is present in this model.")
        else:
            self.nparam -= self.constant.nparam
            self.parameters = self.parameters.ix[self.parameters.name !=
                                                 'constant']
            self.constant = None

    def del_noisemodel(self):
        """Save deletion of the noisemodel from the Model.

        """
        if self.noisemodel is None:
            warn("No noisemodel is present in this model.")
        else:
            self.nparam -= self.noisemodel.nparam
            self.parameters = self.parameters.ix[self.parameters.name !=
                                                 self.noisemodel.name]
            self.noisemodel = None

    def simulate(self, parameters=None, tmin=None, tmax=None, freq=None):
        """Simulate the time series model.

        Parameters
        ----------
        parameters: Optional[list]
            Array of the parameters used in the time series model.
        tmin: Optional[str]
        tmax: Optional[str]
        freq: Optional[str]
            frequency at which the time series are simulated.

        Returns
        -------
        h: pd.Series
            Pandas Series object containing the simulated time series

        Notes
        -----
        This method can be used without any parameters. When the model is
        solved, the optimal parameters values are used and if not,
        the initial parameter values are used. This allows the user to
        obtain an idea of how the simulation looks with only the initial
        parameters and no calibration.

        """
        # Default option when tmin and tmax and freq are not provided.
        if freq is None:
            freq = self.freq

        tmin, tmax = self.get_tmin_tmax(tmin, tmax, freq, use_oseries=False)

        tmin = pd.to_datetime(tmin) - pd.DateOffset(days=self.warmup)
        sim_index = pd.date_range(tmin, tmax, freq=freq)
        dt = self.get_dt(freq)

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        h = np.zeros(len(sim_index))
        istart = 0  # Track parameters index to pass to ts object
        for ts in self.tseriesdict.values():
            c = ts.simulate(parameters[istart: istart + ts.nparam], sim_index,
                            dt)
            h += c.values # no need to match on index, all tseries are on sim_index
            istart += ts.nparam

        if self.constant:
            h += self.constant.simulate(parameters[istart])

        # convert to Pandas Series
        h = pd.Series(h, index=sim_index)

        return h.loc[tmin:]

    def residuals(self, parameters=None, tmin=None, tmax=None, freq=None,
                  h_observed=None, sample_method='nearest'):
        """Calculate residual series

        Parameters
        ----------
        parameters : None, optional
            Array of parameter values
        tmin : None, optional
            Description
        tmax : None, optional
            Description
        freq : None, optional
            Frequency at which the time series are simulated.
        h_observed : None, optional
            Pandas series containing the observed values.
        sample_method : str, optional
            Sample method used for matching simulation to observations
            before calculating residuals.

        Returns
        -------
        TYPE
            Description
        """
        if freq is None:
            freq = self.freq

        tmin, tmax = self.get_tmin_tmax(tmin, tmax, freq, use_oseries=True)

        # simulate model
        simulated = self.simulate(parameters, tmin, tmax, freq)

        if h_observed is None:
            h_observed = self.oseries.loc[tmin:tmax]
            self.oseries_calib = h_observed
        h_simulated = (simulated
                       .reindex_like(h_observed, method=sample_method))
        res = h_observed - h_simulated

        return res.loc[tmin:].dropna()

    def innovations(self, parameters=None, tmin=None, tmax=None, freq=None,
                    h_observed=None):
        """Method to simulate the innovations when a noisemodel is present.

        Parameters
        ----------
        parameters: Optional[list]
            Array of the parameters used in the time series model.
        tmin: Optional[str]
        tmax: Optional[str]
        freq: Optional[str]
            frequency at which the time series are simulated.
        h_observed: Optional[pd.Series]
            Pandas series containing the observed values.

        Returns
        -------
        v: pd.Series
            Pandas series of the innovations.

        Notes
        -----
        The innovations are the time series that result when applying a noise
        model.

        """
        if self.noisemodel is None:
            warn("Innovations can not be calculated as there is no noisemodel")
            return None

        tmin, tmax = self.get_tmin_tmax(tmin, tmax, freq, use_oseries=True)

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        # Calculate the residuals
        res = self.residuals(parameters, tmin, tmax, freq, h_observed)

        # Calculate the innovations
        v = self.noisemodel.simulate(res, self.odelt[res.index],
                                     parameters[-self.noisemodel.nparam:],
                                     res.index)
        return v.loc[tmin:]

    def observations(self, tmin=None, tmax=None):
        """Method that returns the observations series.

        """
        tmin, tmax = self.get_tmin_tmax(tmin, tmax, use_oseries=True)

        return self.oseries.loc[tmin: tmax]

    def initialize(self, initial=True, noise=True):
        """Initialize the model before solving.

        Parameters
        ----------
        initial: Boolean
            Use initial values from parameter dataframe if True. If false, the
            optimal values are used.
        noise: Boolean
            Add the parameters for the noisemodel to the parameters
            Dataframe or not.

        """
        # Store optimized values in case they are needed
        if not initial:
            optimal = self.parameters.optimal

        # make sure calibration data is renewed
        self.oseries_calib = None

        # Set initial parameters
        self.parameters = self.get_init_parameters(noise=noise)
        self.nparam = len(self.parameters)

        # Set initial parameters to optimal parameters
        if not initial:
            self.parameters.initial = optimal

    def solve(self, tmin=None, tmax=None,
              solver=LmfitSolve, objfunc=objf_residuals,
              report=True, noise=True, initial=True):
        """
        Methods to solve the time series model.

        Parameters
        ----------
        tmin: Optional[str]
            String with a start date for the simulation period (E.g. '1980')
        tmax: Optional[str]
            String with an end date for the simulation period (E.g. '2010')
        solver: Optional[solver class]
            Class used to solve the model. Default is lmfit (LmfitSolve)
        report: Boolean
            Print a report to the screen after optimization finished.
        noise: Boolean
            Use the noise model (True) or not (False).
        initialize: Boolean
            Reset initial parameters.

        """
        if noise and (self.noisemodel is None):
            warn(message='Warning, solution with noise model while noise model'
                         'is not defined. No noise model is used.')

        # Check frequency of tseries
        self.set_freq_offset()

        # Check series with tmin, tmax
        self.tmin, self.tmax = self.get_tmin_tmax(tmin, tmax)

        # Initialize parameters
        self.initialize(initial=initial, noise=noise)

        # Solve model
        solver = solver(self.parameters)

        objfunc_kwargs= {'tmin': self.tmin, 'tmax': self.tmax,
            'noise': noise, 'freq': self.freq}

        self.fit = solver.solve(objfunc, self, **objfunc_kwargs)

        self.parameters.optimal = fit.optimal_params
        self.report = fit.report
        if report: print(self.report)

    def get_tmin_tmax(self, tmin=None, tmax=None, freq=None, use_oseries=True):
        """Method that checks and returns valid values for tmin and tmax.

        Parameters
        ----------
        tmin: str
            string with a year or date that can be turned into a pandas
            Timestamp (e.g. pd.tslib.Timestamp(tmin)).
        tmax: str
            string with a year or date that can be turned into a pandas
            Timestamp (e.g. pd.tslib.Timestamp(tmax)).
        freq: str

        use_oseries: Boolean
            boolean to check the tmin and tmax against the oseries.

        Returns
        -------
        tmin, tmax: pd.Timestamp
            returns a pandas timestamp for tmin and tmax.

        Notes
        -----
        The tmin and tmax are checked and returned according to the
        following rules:

        A. If no value for tmin/tmax is provided:
            1. if use_oseries is false, tmin is set to minimum of the tseries.
            2. if use_series is true tmin is set to minimum of the oseries.

        B. If a values for tmin/tmax is provided:
            1. A pandas timestamp is made from the string
            2. if use_oseries is True, tmin is checked against oseries.
            3. tmin is checked against the tseries.

        C. In all cases an offset for the tmin and tmax is added.

        A detailed description of dealing with tmin and tmax and timesteps
        in general can be found in the developers section of the docs.

        """
        # Get tmin and tmax from the tseries
        ts_tmin = pd.Timestamp.max
        ts_tmax = pd.Timestamp.min
        for tseries in self.tseriesdict.values():
            if tseries.tmin < ts_tmin:
                ts_tmin = tseries.tmin
            if tseries.tmax > ts_tmax:
                ts_tmax = tseries.tmax

        # Set tmin properly
        if not tmin and not use_oseries:
            tmin = ts_tmin
        elif not tmin:
            tmin = self.oseries.index.min()
        else:
            tmin = pd.tslib.Timestamp(tmin)
            # Check if tmin > oseries.tmin (Needs to be True)
            if tmin < self.oseries.index.min() and use_oseries:
                warn("Specified tmin is before the first observation. tmin"
                     " automatically set to %s" % self.oseries.index.min())
                tmin = self.oseries.index.min()
            # Check if tmin > tseries.tmin (Needs to be True)
            if tmin < ts_tmin:
                warn("Specified tmin is before any of the tseries tmin. tmin"
                     " automatically set to tseries tmin %s" % ts_tmin)
                tmin = ts_tmin

        # Set tmax properly
        if not tmax and not use_oseries:
            tmax = ts_tmax
        elif not tmax:
            tmax = self.oseries.index.max()
        else:
            tmax = pd.tslib.Timestamp(tmax)
            # Check if tmax < oseries.tmax (Needs to be True)
            if tmax > self.oseries.index.max() and use_oseries:
                warn("Specified tmax is after the last observation. tmax"
                     " automatically set to %s" % self.oseries.index.max())
                tmax = self.oseries.index.max()
            # Check if tmax < tseries.tmax (Needs to be True)
            if tmax > ts_tmax:
                warn("Specified tmax is after any of the tseries tmax. tmax"
                     " automatically set to tseries tmax %s" % ts_tmax)
                tmax = ts_tmax

        # adjust tmin and tmax so that the time-offset is equal to the tseries.
        if not freq:
            freq = self.freq

        offset = pd.tseries.frequencies.to_offset(freq)
        tmin = offset.rollforward(tmin).normalize()
        tmax = offset.rollback(tmax).normalize()

        assert tmax > tmin, \
            'Error: Specified tmax not larger than specified tmin'
        assert self.oseries[tmin: tmax].size > 0, \
            'Error: no observations between tmin and tmax'

        return tmin, tmax

    def set_freq_offset(self):
        """

        Notes
        -----
        Methods to check if the frequency is:

        1. The frequency should be the same for all tseries
        2. tseries timestamps should match (e.g. similar hours)
        3. freq of the tseries is lower than the max tdelta of the oseries

        """

        # calculate frequency and time-difference with default frequency
        freqs = set()
        time_offsets = set()

        for tseries in self.tseriesdict.values():
            if not tseries.stress.empty:
                freqs.add(tseries.freq)
                # calculate the offset from the default frequency
                time_offset = pd.tseries.frequencies.to_offset(tseries.freq)
                time_offsets.add(time_offset)

        # 1. The frequency should be the same for all tseries
        assert len(freqs) == 1, 'The frequency of the tseries is not the ' \
                                'same for all stresses.'
        self.freq = next(iter(freqs))

        # 2. tseries timestamps should match (e.g. similar hours')
        assert len(
            time_offsets) == 1, 'The time-differences with the default frequency is' \
                                ' not the same for all stresses.'
        self.time_offset = next(iter(time_offsets))

    def get_init_parameters(self, noise=True):
        """Method to get all initial parameters from the individual objects.

        Parameters
        ----------
        noise: Boolean
            Add the parameters for the noisemodel to the parameters
            Dataframe or not.

        Returns
        -------
        parameters: pd.DataFrame
            Pandas Dataframe with the parameters.

        """
        parameters = pd.DataFrame(columns=['initial', 'pmin', 'pmax',
                                           'vary', 'optimal', 'name'])
        for ts in self.tseriesdict.values():
            parameters = parameters.append(ts.parameters)
        if self.constant:
            parameters = parameters.append(self.constant.parameters)
        if self.noisemodel and noise:
            parameters = parameters.append(self.noisemodel.parameters)

        return parameters

    def get_parameters(self, name=None):
        """Helper method to obtain the parameters needed for calculation if
        none are provided. This method is used by the simulation, residuals
        and the innovations methods.

        Returns
        -------
        p: list
            Array of the parameters used in the time series model.

        """
        if name:
            p = self.parameters[self.parameters.name == name]
        else:
            p = self.parameters

        if p.optimal.hasnans:
            warn("Model is not optimized yet, initial parameters are used.")
            parameters = p.initial
        else:
            parameters = p.optimal

        return parameters.values

    def get_dt(self, freq):
        """Summary

        Parameters
        ----------
        freq : str
            Pandas frequency string

        Returns
        -------
        float
            Frequency time offset in decimal days
        """
        offset = pd.tseries.frequencies.to_offset(freq)

        to_days = {
            'years': 365.25,
            'months': 30.5,
            'weekday': 7.,
            'days': 1.,
            'hours': 1. / 24.,
            'minutes': 1. / 24. / 60.,
            'seconds': 1. / 24. / 60. / 60.,
            'milliseconds': 1. / 24. / 60. / 60. * 1e-3,
            }
        if not len(offset.kwds):
            return 1.
        else:
            # day of the week is not a multiplier, set to 1.
            offset.kwds['weekday'] = 1.
            return sum(n * to_days[u] for u, n in offset.kwds.items())

    def get_contribution(self, name):
        if name not in self.tseriesdict.keys():
            warn("Name not in tseriesdict, available names are: %s"
                 % self.tseriesdict.keys())
            return None
        else:
            p = self.get_parameters(name)
            dt = self.get_dt(self.freq)
            return self.tseriesdict[name].simulate(p, dt=dt)

    def get_block_response(self, name):
        if name not in self.tseriesdict.keys():
            warn("Name not in tseriesdict, available names are: %s"
                 % self.tseriesdict.keys())
            return None
        else:
            p = self.get_parameters(name)
            dt = self.get_dt(self.freq)
            return self.tseriesdict[name].rfunc.block(p, dt)

    def get_step_response(self, name):
        if name not in self.tseriesdict.keys():
            warn("Name not in tseriesdict, available names are: %s"
                 % self.tseriesdict.keys())
            return None
        else:
            p = self.get_parameters(name)
            dt = self.get_dt(self.freq)
            return self.tseriesdict[name].rfunc.step(p, dt)

    def get_stress(self, name):
        try:
            p = self.parameters.loc[
                self.parameters.name == name, 'optimal'].values
            return self.tseriesdict[name].get_stress(p)
        except KeyError:
            print("Name not in tseriesdict, available names are: %s"
                  % self.tseriesdict.keys())

    def plot(self, tmin=None, tmax=None, oseries=True, simulate=True):
        """

        Parameters
        ----------
        oseries: Boolean
            True to plot the observed time series.

        Returns
        -------
        Plot of the simulated and optionally the observed time series

        """
        self.plots.plot(tmin, tmax, oseries, simulate)
