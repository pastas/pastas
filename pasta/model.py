from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import numpy as np
import pandas as pd
import datetime
from warnings import warn

from .checks import check_oseries
from .solver import LmfitSolve
from .tseries import Constant
from .stats import Statistics


class Model:
    def __init__(self, oseries, xy=(0, 0), metadata=None, freq=None,
                 warmup=1500, fillnan='drop'):
        """
        Initiates a time series model.

        Parameters
        ----------
        oseries: pd.Series
            pandas Series object containing the dependent time series. The
            observation can be non-equidistant.
        xy: Optional[tuple]
            XY location of the oseries in lat-lon format.
        metadata: Optional[dict]
            Dictionary containing metadata of the model.
        freq: Optional[str]
            String containing the desired frequency. By default freq=None and the
            observations are used as they are. The required string format is found
            at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
            -aliases
        warmup: Optional[float]
            Number of days used for warmup
        fillnan: Optional[str or float]
            Methods or float number to fill nan-values. Default values is
            'drop'. Currently supported options are: 'interpolate', float,
            'mean' and, 'drop'. Interpolation is performed with a standard linear
            interpolation.

        """
        self.oseries = check_oseries(oseries, freq, fillnan)
        self.warmup = warmup
        self.xy = xy
        self.metadata = metadata
        self.odelt = self.oseries.index.to_series().diff() / \
                     np.timedelta64(1, 'D')
        self.freq = None
        self.time_offset = None

        # Independent of the time unit
        self.tseriesdict = OrderedDict()
        self.noisemodel = None
        self.noiseparameters = None
        self.nparam = 0
        self.tmin = None
        self.tmax = None
        # Min and max time of observation series
        self.otmin = self.oseries.index[0]
        self.otmax = self.oseries.index[-1]
        self.stats = Statistics(self)

    def add_tseries(self, tseries):
        """
        adds a time series model component to the Model.

        """
        self.tseriesdict[tseries.name] = tseries

    def add_noisemodel(self, noisemodel):
        """
        Adds a noise model to the time series Model.

        """
        self.noisemodel = noisemodel

    def simulate(self, parameters=None, tmin=None, tmax=None, freq='D'):
        """

        Parameters
        ----------
        t: Optional[pd.series.index]
            Time indices to use for the simulation of the time series model.
        p: Optional[array]
            Array of the parameters used in the time series model.
        noise:

        Returns
        -------
        Pandas Series object containing the simulated time series.

        """

        # Default option when not tmin and tmax is provided
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax
        assert (tmin is not None) and (
            tmax is not None), 'model needs to be solved first'

        tindex = pd.date_range(
            pd.to_datetime(tmin) - pd.DateOffset(days=self.warmup), tmax,
            freq=freq)
        dt = self.get_dt(freq)

        if parameters is None:
            parameters = self.parameters.optimal.values
        h = pd.Series(data=0, index=tindex)
        istart = 0  # Track parameters index to pass to ts object
        for ts in self.tseriesdict.values():
            h += ts.simulate(parameters[istart: istart + ts.nparam], tindex, dt)
            istart += ts.nparam
        return h[tmin:]

    def residuals(self, parameters=None, tmin=None, tmax=None, freq='D', noise=True):
        """
        Method to calculate the residuals.

        """
        if tmin is None:
            tmin = self.oseries.index.min()
        if tmax is None:
            tmax = self.oseries.index.max()
        tindex = self.oseries[tmin: tmax].index  # times used for calibration

        if parameters is None:
            parameters = self.parameters.optimal.values

        # h_observed - h_simulated
        h_observed = self.oseries[tindex]
        simulation = self.simulate(parameters, tmin, tmax, freq)
        if len(tindex.difference(simulation.index)) == 0:
            # all of the observation indexes are in the simulation
            h_simulated = simulation[tindex]
        else:
            # interpolate simulation to measurement-times
            h_simulated = np.interp(h_observed.index.asi8,
                                    simulation.index.asi8, simulation)
        r = h_observed - h_simulated
        if noise and (self.noisemodel is not None):
            r = self.noisemodel.simulate(r, self.odelt[tindex],
                                         parameters[-self.noisemodel.nparam:],
                                         tindex)
        if np.isnan(sum(r ** 2)):
            print('nan problem in residuals')  # quick and dirty check
        return r[tmin:]

    def initialize(self, initial=True, noise=True):
        if not initial:
            optimal = self.parameters.optimal
        self.nparam = sum(ts.nparam for ts in self.tseriesdict.values())
        if self.noisemodel is not None:
            self.nparam += self.noisemodel.nparam
        self.parameters = pd.DataFrame(columns=['initial', 'pmin', 'pmax',
                                                'vary', 'optimal', 'name'])
        for ts in self.tseriesdict.values():
            self.parameters = self.parameters.append(ts.parameters)
        if self.noisemodel and noise:
            self.parameters = self.parameters.append(
                self.noisemodel.parameters)
        if not initial:
            self.parameters.initial = optimal

    def solve(self, tmin=None, tmax=None, solver=LmfitSolve, report=True,
              noise=True, initial=True, solve=True):
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
            Reset initial parameteres.

        """
        if noise and (self.noisemodel is None):
            print('Warning, solution with noise model while noise model is not '
                  'defined. No noise model is used')

        # Check frequency of tseries
        self.check_frequency()

        # Check series with tmin, tmax
        self.set_tmin_tmax(tmin, tmax)

        # Initialize parameters
        self.initialize(initial=initial, noise=noise)

        # Solve model
        fit = solver(self, tmin=self.tmin, tmax=self.tmax, noise=noise, freq=self.freq)

        self.parameters.optimal = fit.optimal_params
        self.report = fit.report
        if report: print(self.report)

        # self.stats = Statistics(self)

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

        # adjust tmin and tmax so that the time-offset (determined in check_frequency) is equal to that of the tseries
        tmin = tmin - self.get_time_offset(tmin, self.freq) + self.time_offset
        tmax = tmax - self.get_time_offset(tmax, self.freq) + self.time_offset

        assert tmax > tmin, 'Error: Specified tmax not larger than specified tmin'
        assert len(self.oseries[
                   tmin: tmax]) > 0, 'Error: no observations between tmin and tmax'

        self.tmin = tmin
        self.tmax = tmax

    def check_frequency(self):
        """
        Methods to check if the frequency is:

        1. The frequency should be the same for all tseries
        2. tseries timestamps should match (e.g. similar hours)
        3. freq of the tseries is lower than the max tdelta of the oseries

        """

        # calculate frequency and time-difference with default frequency
        freqs = set()
        time_offsets = set()

        for tseries in self.tseriesdict.values():
            if isinstance(tseries, Constant):
                pass
            else:
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

    def get_dt(self, freq):
        options = {'W': 7,  # weekly frequency
                   'D': 1,  # calendar day frequency
                   'H': 1/24,  # hourly frequency
                   'T': 1/24/60,  # minutely frequency
                   'min': 1/24/60,  # minutely frequency
                   'S': 1/24/3600,  # secondly frequency
                   'L': 1/24/3600000,  # milliseconds
                   'ms': 1/24/3600000,  # milliseconds
                   }
        # remove the day from the week
        freq = freq.split("-", 1)[0]
        return options[freq]

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
        # remove the day from the week
        freq = freq.split("-", 1)[0]
        return options[freq](t)

    def get_contribution(self, name):
        try:
            p = self.parameters.loc[
                self.parameters.name == name, 'optimal'].values
            return self.tseriesdict[name].simulate(p)
        except KeyError:
            print("Name not in tseriesdict, available names are: %s"
                  % self.tseriesdict.keys())

    def get_response_function(self, name):
        try:
            p = self.parameters.loc[
                self.parameters.name == name, 'optimal'].values
            return self.tseriesdict[name].rfunc.block(p)
        except KeyError:
            print("Name not in tseriesdict, available names are: %s"
                  % self.tseriesdict.keys())

    def get_stress(self, name):
        try:
            p = self.parameters.loc[
                self.parameters.name == name, 'optimal'].values
            return self.tseriesdict[name].__getstress__(p)
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
        plt.figure()
        if simulate:
            if tmin is None:
                tmin = self.otmin
            if tmax is None:
                tmax = self.otmax
            h = self.simulate(tmin=tmin, tmax=tmax)
            h.plot()
        if oseries:
            self.oseries.plot(linestyle='', marker='.', color='k',
                              markersize=3)
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
        gs = plt.GridSpec(3, 4, wspace=0.2)

        # Plot the Groundwater levels
        h = self.simulate(tmin=tmin, tmax=tmax)
        ax1 = plt.subplot(gs[:2, :-1])
        h.plot(label='modeled head')
        self.oseries.plot(linestyle='', marker='.', color='k', markersize=3,
                          label='observed head')
        # ax1.xaxis.set_visible(False)
        plt.legend(loc=(0, 1), ncol=3, frameon=False, handlelength=3)
        plt.ylabel('Head [m]')

        # Plot the residuals and innovations
        residuals = self.residuals(tmin=tmin, tmax=tmax)
        ax2 = plt.subplot(gs[2, :-1])  # , sharex=ax1)
        residuals.plot(color='k', label='residuals')
        # Ruben Calje commented next three lines on 31-10-2016:
        # if self.noisemodel is not None:
        # innovations = self.noisemodel.simulate(residuals, self.odelt)
        # innovations.plot(label='innovations')
        plt.legend(loc=(0, 1), ncol=3, frameon=False, handlelength=3)
        plt.ylabel('Error [m]')
        plt.xlabel('Time [Years]')

        # Plot the Impulse Response Function
        ax3 = plt.subplot(gs[0, -1])
        n = 0
        for ts in self.tseriesdict.values():
            p = self.parameters[n:n + ts.nparam]
            n += ts.nparam
            if "rfunc" in dir(ts):
                plt.plot(ts.rfunc.block(p.optimal))
        ax3.set_xticks(ax3.get_xticks()[::2])
        ax3.set_yticks(ax3.get_yticks()[::2])
        plt.title('Block Response')

        # Plot the Model Parameters (Experimental)
        ax4 = plt.subplot(gs[1:2, -1])
        ax4.xaxis.set_visible(False)
        ax4.yaxis.set_visible(False)
        text = np.vstack((self.parameters.keys(), [round(float(i), 4) for i in
                                                   self.parameters.optimal.values])).T
        colLabels = ("Parameter", "Value")
        ytable = ax4.table(cellText=text, colLabels=colLabels, loc='center')
        ytable.scale(1, 1.1)

        # Table of the numerical diagnostic statistics.
        ax5 = plt.subplot(gs[2, -1])
        ax5.xaxis.set_visible(False)
        ax5.yaxis.set_visible(False)
        # Ruben Calje commented next two lines on 31-10-2016:
        # plt.text(0.05, 0.8, 'AIC: %.2f' % self.fit.aic)
        # plt.text(0.05, 0.6, 'BIC: %.2f' % self.fit.bic)
        plt.show()
        if savefig:
            plt.savefig('.eps' % (self.name), bbox_inches='tight')

    def plot_decomposition(self, tmin=None, tmax=None, freq=None):
        """

        Plot the decomposition of a time-series in the different stresses

        """

        # Default option when not tmin and tmax is provided
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax
        assert (tmin is not None) and (
            tmax is not None), 'model needs to be solved first'
        if freq is None:
            freq = self.freq

        tindex = pd.date_range(tmin, tmax, freq=freq)

        # determine the simulation
        hsim = self.simulate(tmin=tmin, tmax=tmax, freq=freq)
        h = [hsim]

        # determine the influence of the different stresses
        parameters = self.parameters.optimal.values
        istart = 0  # Track parameters index to pass to ts object
        for ts in self.tseriesdict.values():
            dt = self.get_dt(freq)
            h.append(ts.simulate(parameters[istart: istart + ts.nparam], tindex, dt))
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
            f, axarr = plt.subplots(1 + len(self.tseriesdict), sharex=True, gridspec_kw={
                'height_ratios': height_ratios})

        # plot simulation and observations in top graph
        plt.axes(axarr[0])
        self.oseries.plot(linestyle='', marker='.', color='k', markersize=3,
                          ax=axarr[0], label='observations')
        hsim.plot(ax=axarr[0], label='simulation')
        axarr[0].set_title('Observations and simulation')
        axarr[0].autoscale(enable=True, axis='y', tight=True)
        axarr[0].grid(which='both')
        axarr[0].minorticks_off()

        # add a legend
        handles, labels = axarr[0].get_legend_handles_labels()
        leg = axarr[0].legend(handles, labels, loc=2)
        leg.get_frame().set_alpha(0.5)

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
            if isinstance(ts, Constant):
                xlim = axarr[iax].get_xlim()
                axarr[iax].plot(xlim, [h[iax], h[iax]])
                axarr[iax].yaxis.set_ticks(h[iax])
            else:
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
