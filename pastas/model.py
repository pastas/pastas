from __future__ import print_function, division

import datetime
from collections import OrderedDict
from warnings import warn

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from scipy import interpolate

from .checks import check_oseries
from .solver import LmfitSolve
from .stats import Statistics
from .tseries import Constant


class Model:
    def __init__(self, oseries, xy=(0, 0), metadata=None,
                 warmup=1500, fillnan='drop', constant=True):
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
        # Min and max time of observation series
        self.otmin = self.oseries.index[0]
        self.otmax = self.oseries.index[-1]
        self.odelt = self.oseries.index.to_series().diff() / \
                     np.timedelta64(1, 'D')

        self.warmup = warmup
        self.freq = None
        self.time_offset = None

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

        # Load other modules
        self.stats = Statistics(self)

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
            self.parameters = self.parameters.append(tseries.parameters)
            self.nparam += tseries.nparam

            # Call these methods to set tmin, tmax and freq and enable
            # simulation.
            self.set_freq_offset()
            self.set_tmin_tmax()

    def add_noisemodel(self, noisemodel):
        """Adds a noise model to the time series Model.

        """
        self.noisemodel = noisemodel
        self.parameters = self.parameters.append(self.noisemodel.parameters)
        self.nparam += noisemodel.nparam

    def add_constant(self):
        """Adds a Constant to the time series Model.

        """
        self.constant = Constant(value=self.oseries.mean(), name='constant')
        self.parameters = self.parameters.append(self.constant.parameters)
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
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax
        if freq is None:
            freq = self.freq

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        # adjust tmin and tmax so that the time-offset (determined in
        # check_frequency) is equal to that of the model (is also done in set_tmin_tmax)
        tmin = tmin - self.get_time_offset(tmin, freq) + self.time_offset
        tmax = tmax - self.get_time_offset(tmax, freq) + self.time_offset

        sim_index = pd.date_range(
            pd.to_datetime(tmin) - pd.DateOffset(days=self.warmup), tmax,
            freq=freq)
        dt = self.get_dt(freq)

        h = pd.Series(data=0, index=sim_index)
        istart = 0  # Track parameters index to pass to ts object
        for ts in self.tseriesdict.values():
            h += ts.simulate(parameters[istart: istart + ts.nparam], sim_index,
                             dt)
            istart += ts.nparam
        if self.constant:
            h += self.constant.simulate(parameters[istart])

        return h[tmin:]

    def residuals(self, parameters=None, tmin=None, tmax=None, freq=None,
                  h_observed=None):
        """

        Parameters
        ----------
        parameters: Optional[list]
            Array of the parameters used in the time series model.
        tmin: Optional[str]
        tmax: Optional[str]
        freq: Optional[str]
            frequency at which the time series are simulated.
        noise: Optional[Boolean]
            Boolean to indicate the use of the noisemodel (if provided) or not.
        h_observed: Optional[pd.Series]
            Pandas series containing the observed values.

        Returns
        -------
        res: pd.Series
            Pandas series with the simulated

        """
        if tmin is None:
            tmin = self.oseries.index.min()
        if tmax is None:
            tmax = self.oseries.index.max()
        if freq is None:
            freq = self.freq

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        # simulate model
        simulation = self.simulate(parameters, tmin, tmax, freq)

        if h_observed is None:
            h_observed = self.oseries[tmin: tmax]
            # sample measurements, so that frequency is not higher than model
            h_observed = self.sample(h_observed, simulation.index)
            # store this variable in the model, so that it can be used in the next iteration of the solver
            self.oseries_calib = h_observed

        obs_index = h_observed.index  # times used for calibration

        # h_observed - h_simulated
        if len(obs_index.difference(simulation.index)) == 0:
            # all of the observation indexes are in the simulation
            h_simulated = simulation[obs_index]
        else:
            # interpolate simulation to measurement-times
            h_simulated = np.interp(h_observed.index.asi8,
                                    simulation.index.asi8, simulation)
        res = h_observed - h_simulated

        if np.isnan(sum(res ** 2)):
            print('nan problem in residuals')  # quick and dirty check
        return res[tmin:]

    def innovations(self, parameters=None, tmin=None, tmax=None, freq=None,
                    h_observed=None):
        """Method to simulate the innovations.

        Parameters
        ----------
        parameters
        tmin
        tmax
        freq

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

        if tmin is None:
            tmin = self.oseries.index.min()
        if tmax is None:
            tmax = self.oseries.index.max()
        if freq is None:
            freq = self.freq

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        # Calculate the residuals
        res = self.residuals(parameters, tmin, tmax, freq, h_observed)

        # Calculate the innovations
        v = self.noisemodel.simulate(res, self.odelt[res.index],
                                     parameters[-self.noisemodel.nparam:],
                                     res.index)

        return v[tmin:]

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

    def solve(self, tmin=None, tmax=None, solver=LmfitSolve, report=True,
              noise=True, initial=True):
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
            Use the nose model (True) or not (False).
        initialize: Boolean
            Reset initial parameters.

        """
        if noise and (self.noisemodel is None):
            warn(message='Warning, solution with noise model while noise model'
                         'is not defined. No noise model is used')

        # Check frequency of tseries
        self.set_freq_offset()

        # Check series with tmin, tmax
        self.set_tmin_tmax(tmin, tmax)

        # Initialize parameters
        self.initialize(initial=initial, noise=noise)

        # Solve model
        fit = solver(self, tmin=self.tmin, tmax=self.tmax, noise=noise,
                     freq=self.freq)

        self.parameters.optimal = fit.optimal_params
        self.report = fit.report
        if report: print(self.report)

    def set_tmin_tmax(self, tmin=None, tmax=None):
        """
        Function to check if the dependent and independent time series match.

        - tmin and tmax are in oseries.index for optimization.
        - all the stresses are available for simulation between tmin and tmax.
        -

        Parameters
        ----------
        tmin
        tmax

        Returns
        -------

        """

        # Store tmax and tmin. If none is provided, use oseries to set them.
        if tmin is None:
            tmin = self.oseries.index.min()
        else:
            tmin = pd.tslib.Timestamp(tmin)
            if tmin < self.oseries.index[0]:
                warn('Specified tmin is before first observation ' + str(
                    self.oseries.index[0]))
            assert tmin <= self.oseries.index[
                -1], 'Error: Specified tmin is after last observation'
        if tmax is None:
            tmax = self.oseries.index.max()
        else:
            tmax = pd.tslib.Timestamp(tmax)
            if tmax > self.oseries.index[-1]:
                warn('Specified tmax is after last observation ' + str(
                    self.oseries.index[-1]))
            assert tmax >= self.oseries.index[
                0], 'Error: Specified tmax is before first observation'

        # adjust tmin and tmax, so that all the tseries cover the period
        for tseries in self.tseriesdict.values():
            if tseries.tmin > tmin:
                tmin = tseries.tmin
            if tseries.tmax < tmax:
                tmax = tseries.tmax

        # adjust tmin and tmax so that the time-offset (determined in
        # check_frequency) is equal to that of the tseries
        tmin = tmin - self.get_time_offset(tmin, self.freq) + self.time_offset
        tmax = tmax - self.get_time_offset(tmax, self.freq) + self.time_offset

        assert tmax > tmin, 'Error: Specified tmax not larger than specified tmin'
        assert len(self.oseries[
                   tmin: tmax]) > 0, 'Error: no observations between tmin and tmax'

        self.tmin = tmin
        self.tmax = tmax

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
                time_offset = self.get_time_offset(tseries.stress.index[0],
                                                   tseries.freq)
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
        """Method to reset all initial parameters to those provided in the
        individual objects.

        noise: Boolean
            Add the parameters for the noisemodel to the parameters
            Dataframe or not.

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

    def get_parameters(self):
        """Helper method to obtain the parameters needed for calculation if
        none are provided. This method is used by the simulation, residuals
        and the innovations methods.

        Returns
        -------
        p: list
            Array of the parameters used in the time series model.

        """
        if self.parameters.optimal.hasnans:
            parameters = self.parameters.initial.values
        else:
            parameters = self.parameters.optimal.values

        return parameters

    def get_dt(self, freq):
        options = {'W': 7,  # weekly frequency
                   'D': 1,  # calendar day frequency
                   'H': 1 / 24,  # hourly frequency
                   'T': 1 / 24 / 60,  # minutely frequency
                   'min': 1 / 24 / 60,  # minutely frequency
                   'S': 1 / 24 / 3600,  # secondly frequency
                   'L': 1 / 24 / 3600000,  # milliseconds
                   'ms': 1 / 24 / 3600000,  # milliseconds
                   }
        # Get the frequency string and multiplier
        num, freq = self.get_freqstr(freq)
        dt = num * options[freq]
        return dt

    def get_time_offset(self, t, freq):
        if isinstance(t, pd.Series):
            # Take the first timestep. The rest of index has the same offset,
            # as the frequency is constant.
            t = t.index[0]

        # define the function blocks
        def calc_week_offset(t):
            return datetime.timedelta(days=t.weekday(), hours=t.hour,
                                      minutes=t.minute, seconds=t.second)

        def calc_day_offset(t):
            return datetime.timedelta(hours=t.hour, minutes=t.minute,
                                      seconds=t.second)

        def calc_hour_offset(t):
            return datetime.timedelta(minutes=t.minute, seconds=t.second)

        def calc_minute_offset(t):
            return datetime.timedelta(seconds=t.second)

        def calc_second_offset(t):
            return datetime.timedelta(microseconds=t.microsecond)

        def calc_millisecond_offset(t):
            # t has no millisecond attribute, so use microsecond and use the remainder after division by 1000
            return datetime.timedelta(microseconds=t.microsecond % 1000.0)

        # map the inputs to the function blocks
        # see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
        options = {'W': calc_week_offset,  # weekly frequency
                   'D': calc_day_offset,  # calendar day frequency
                   'H': calc_hour_offset,  # hourly frequency
                   'T': calc_minute_offset,  # minutely frequency
                   'min': calc_minute_offset,  # minutely frequency
                   'S': calc_second_offset,  # secondly frequency
                   'L': calc_millisecond_offset,  # milliseconds
                   'ms': calc_millisecond_offset,  # milliseconds
                   }
        # Get the frequency string and multiplier
        num, freq = self.get_freqstr(freq)
        offset = num * options[freq](t)
        return offset

    def get_freqstr(self, freqstr):
        """Method to untangle the frequency string.

        """
        # remove the day from the week
        freqstr = freqstr.split("-", 1)[0]

        # Find a number by which the frequency is multiplied
        num = ''
        freq = ''
        for s in freqstr:
            if s.isdigit():
                num = num.__add__(s)
            else:
                freq = freq.__add__(s)
        if num:
            num = int(num)
        else:
            num = 1

        return num, freq

    def get_contribution(self, name):
        try:
            p = self.parameters.loc[
                self.parameters.name == name, 'optimal'].values
            return self.tseriesdict[name].simulate(p)
        except KeyError:
            print("Name not in tseriesdict, available names are: %s"
                  % self.tseriesdict.keys())

    def get_block_response(self, name):
        try:
            p = self.parameters.loc[
                self.parameters.name == name, 'optimal'].values
            dt = self.get_dt(self.freq)
            return self.tseriesdict[name].rfunc.block(p, dt)
        except KeyError:
            print("Name not in tseriesdict, available names are: %s"
                  % self.tseriesdict.keys())

    def get_stress(self, name):
        try:
            p = self.parameters.loc[
                self.parameters.name == name, 'optimal'].values
            return self.tseriesdict[name].get_stress(p)
        except KeyError:
            print("Name not in tseriesdict, available names are: %s"
                  % self.tseriesdict.keys())

    def sample(self, series, tindex):
        # Sample the series so that the frequency is not higher that tindex
        # Find the index closest to the tindex, and then return a selection of series
        f = interpolate.interp1d(series.index.asi8,
                                 np.arange(0, len(series.index)),
                                 kind='nearest', bounds_error=False,
                                 fill_value='extrapolate')
        ind = np.unique(f(tindex.asi8).astype(int))
        return series[ind]

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
        plt.figure()
        if oseries:
            self.oseries.plot(linestyle='', marker='.', color='k',
                              markersize=3)
        if simulate:
            if tmin is None:
                tmin = self.otmin
            if tmax is None:
                tmax = self.otmax
            h = self.simulate(tmin=tmin, tmax=tmax)
            h.plot()

        plt.show()

    def plot_results(self, tmin=None, tmax=None, savefig=False):
        """

        Parameters
        ----------
        tmin/tmax: str
            start and end time for plotting
        savefig: Optional[Boolean]
            True to save the figure, False is default. Figure is saved in the
            current working directory when running your python scripts.

        Returns
        -------

        """
        plt.figure('Model Results', facecolor='white')
        gs = plt.GridSpec(3, 4, wspace=0.4, hspace=0.4)

        # Plot the Groundwater levels
        h = self.simulate(tmin=tmin, tmax=tmax)
        ax1 = plt.subplot(gs[:2, :-1])
        self.oseries.plot(linestyle='', marker='.', color='k', markersize=3,
                          label='observed head', ax=ax1)
        h.plot(label='modeled head', ax=ax1)
        ax1.grid(which='both')
        ax1.minorticks_off()
        plt.legend(loc=(0, 1), ncol=3, frameon=False)
        plt.ylabel('Head [m]')

        # Plot the residuals and innovations
        residuals = self.residuals(tmin=tmin, tmax=tmax)
        ax2 = plt.subplot(gs[2, :-1], sharex=ax1)
        residuals.plot(color='k', label='residuals')
        if self.noisemodel is not None:
            innovations = self.residuals(tmin=tmin, tmax=tmax)
            innovations.plot(label='innovations')
        ax2.grid(which='both')
        ax2.minorticks_off()
        plt.legend(loc=(0, 1), ncol=3, frameon=False)
        plt.ylabel('Error [m]')
        plt.xlabel('Time [Years]')

        # Plot the block response function
        ax3 = plt.subplot(gs[0, -1])
        for name, ts in self.tseriesdict.items():
            dt = self.get_dt(self.freq)
            if "rfunc" in dir(ts):
                br = self.get_block_response(name)
                t = np.arange(0, len(br) * dt, dt)
                plt.plot(t, br)
        ax3.set_xticks(ax3.get_xticks()[::2])
        ax3.set_yticks(ax3.get_yticks()[::2])
        ax3.grid(which='both')
        plt.title('Block Response', loc='left')

        # Plot the Model Parameters (Experimental)
        # ax4 = plt.subplot(gs[1:2, -1])
        # ax4.xaxis.set_visible(False)
        # ax4.yaxis.set_visible(False)
        # text = np.vstack((self.parameters['optimal'].keys(), [round(float(i), 4) for i in
        #                                                       self.parameters['optimal'].values])).T
        # colLabels = ("Parameter", "Value")
        # ytable = ax4.table(cellText=text, colLabels=colLabels, loc='center')
        # ytable.scale(1, 1.1)

        # Table of the numerical diagnostic statistics.
        ax5 = plt.subplot(gs[2, -1])
        ax5.xaxis.set_visible(False)
        ax5.yaxis.set_visible(False)
        plt.text(0.05, 0.8, 'AIC: %.2f' % self.stats.aic())
        plt.text(0.05, 0.6, 'BIC: %.2f' % self.stats.aic())
        plt.title('Statistics', loc='left')
        plt.show()
        if savefig:
            plt.savefig('pastas.eps', bbox_inches='tight')

    def plot_decomposition(self, tmin=None, tmax=None):
        """Plot the decomposition of a time-series in the different stresses.

        """

        # Default option when not tmin and tmax is provided
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax
        assert (tmin is not None) and (
            tmax is not None), 'model needs to be solved first'

        # determine the simulation
        hsim = self.simulate(tmin=tmin, tmax=tmax)
        tindex = hsim.index
        h = [hsim]

        # determine the influence of the different stresses
        parameters = self.parameters.optimal.values
        istart = 0  # Track parameters index to pass to ts object
        for ts in self.tseriesdict.values():
            dt = self.get_dt(self.freq)
            h.append(
                ts.simulate(parameters[istart: istart + ts.nparam], tindex,
                            dt))
            istart += ts.nparam

        # open the figure
        if False:
            f, axarr = plt.subplots(1 + len(self.tseriesdict), sharex=True)
        else:
            # let the height of the axes be determined by the values
            # height_ratios = [1]*(len(self.tseriesdict)+1)
            height_ratios = [max([hsim.max(), self.oseries.max()]) - min(
                [hsim.min(), self.oseries.min()])]
            for ht in h[1:]:
                height_ratios.append(ht.max() - ht.min())
            f, axarr = plt.subplots(1 + len(self.tseriesdict), sharex=True,
                                    gridspec_kw={
                                        'height_ratios': height_ratios})

        # plot simulation and observations in top graph
        plt.axes(axarr[0])
        self.oseries.plot(linestyle='', marker='.', color='k', markersize=3,
                          ax=axarr[0], label='observations')
        hsim.plot(ax=axarr[0], label='simulation')
        axarr[0].autoscale(enable=True, axis='y', tight=True)
        axarr[0].grid(which='both')
        axarr[0].minorticks_off()

        # add a legend
        axarr[0].legend(loc=(0, 1), ncol=3, frameon=False)

        # determine the ytick-spacing of the top graph
        yticks, ylabels = plt.yticks()
        if len(yticks) > 2:
            base = yticks[1] - yticks[0]
        else:
            base = None

        # plot the influence of the stresses
        iax = 1
        for ts in self.tseriesdict.values():
            plt.axes(axarr[iax])
            plt.plot(h[iax].index, h[iax].values)
            if base is not None:
                # set the ytick-spacing equal to the top graph
                axarr[iax].yaxis.set_major_locator(
                    plticker.MultipleLocator(base=base))

            axarr[iax].set_title(ts.name)
            axarr[iax].autoscale(enable=True, axis='y', tight=True)
            axarr[iax].grid(which='both')
            axarr[iax].minorticks_off()
            iax += 1

        # show the figure
        plt.tight_layout()
        plt.show()

    def get_simulation(self, tmin=None, tmax=None):
        """Method that returns the simulated series.

        """
        series = self.simulate(tmin=tmin, tmax=tmax)
        return series

    def get_innovations(self, tmin=None, tmax=None):
        """Method that returns the innovation series.

        """
        v = self.innovations(tmin=tmin, tmax=tmax)
        return v

    def get_residuals(self, tmin=None, tmax=None):
        """Method that returns the residual series

        """
        return self.residuals(tmin, tmax)

    def get_observations(self, tmin=None, tmax=None):
        """Method that returns the observations series.

        """
        if tmin is None:
            tmin = self.oseries.index.min()
        if tmax is None:
            tmax = self.oseries.index.max()
        tindex = self.oseries[tmin: tmax].index
        return self.oseries[tindex]
