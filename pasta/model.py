import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tseries import Constant
from checks import check_oseries
from stats import Statistics
from solver import LmfitSolve


class Model:
    def __init__(self, oseries, xy=(0, 0), metadata=None, freq=None,
                 fillnan='drop'):
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
        fillnan: Optional[str or float]
            Methods or float number to fill nan-values. Default values is
            'drop'. Currently supported options are: 'interpolate', float,
            'mean' and, 'drop'. Interpolation is performed with a standard linear
            interpolation.

        """
        self.oseries = check_oseries(oseries, freq, fillnan)
        self.xy = xy
        self.metadata = metadata
        self.odelt = self.oseries.index.to_series().diff() / np.timedelta64(1, 'D')
        # delt converted to days
        self.tserieslist = []
        self.noisemodel = None
        self.noiseparameters = None
        self.tmin = None
        self.tmax = None

    def addtseries(self, tseries):
        """
        adds a time series model component to the Model.

        """
        self.tserieslist.append(tseries)

    def addnoisemodel(self, noisemodel):
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
        assert (tmin is not None) and (tmax is not None), 'model needs to be solved first'

        tindex = pd.date_range(tmin, tmax, freq=freq)

        if parameters is None:
            parameters = self.optimal_params
        h = pd.Series(data=0, index=tindex)
        istart = 0  # Track parameters index to pass to ts object
        for ts in self.tserieslist:
            h += ts.simulate(tindex, parameters[istart: istart + ts.nparam])
            istart += ts.nparam
        return h

    def residuals(self, parameters=None, tmin=None, tmax=None, noise=True):
        """
        Method to calculate the residuals.

        """
        if tmin is None:
            tmin = self.oseries.index.min()
        if tmax is None:
            tmax = self.oseries.index.max()
        tindex = self.oseries[tmin: tmax].index  # times used for calibration

        if parameters is None:
            parameters = self.optimal_params

        # h_observed - h_simulated
        r = self.oseries[tindex] - self.simulate(parameters, tmin, tmax)[tindex]
        if noise and (self.noisemodel is not None):
            r = self.noisemodel.simulate(r, self.odelt[tindex], tindex,
                                         parameters[-self.noisemodel.nparam:])
        if np.isnan(sum(r ** 2)):
            print 'nan problem in residuals'  # quick and dirty check
        return r
    
    def sse(self, parameters=None, tmin=None, tmax=None, noise=True):
        res = self.residuals(parameters, tmin=tmin, tmax=tmax, noise=noise)
        return sum(res ** 2)

    def initialize(self, default_parameters=True):
        self.nparam = sum(ts.nparam for ts in self.tserieslist)
        if self.noisemodel is not None:
            self.nparam += self.noisemodel.nparam
        self.parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
        for ts in self.tserieslist:
            self.parameters = self.parameters.append(ts.parameters)
        if self.noisemodel:
            self.parameters = self.parameters.append(self.noisemodel.parameters)
        if not default_parameters:
            self.parameters.value[:] = self.optimal_params

    def solve(self, tmin=None, tmax=None, solver=LmfitSolve, report=True,
              noise=True, default_parameters=True, solve=True):
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
            print 'Warning, solution with noise model while noise model is not ' \
                  'defined. No noise model is used'

        # Check series with tmin, tmax
        tmin, tmax = self.check_series(tmin, tmax)

        # Initialize parameters
        self.initialize(default_parameters=default_parameters)

        # Solve model
        fit = solver(self, tmin=tmin, tmax=tmax, noise=noise)
         
        self.optimal_params = fit.optimal_params

        #TODO: this should be a pd.Series and should be given back to the ts objects
        #self.optimal_params = pd.Series(fit.optimal_params, index=self.parameters.index)
        #for ts in self.tserieslist:
        #    for k in ts.parameters.index:
        #        ts.parameters.loc[k].value = self.paramdict[k]
        #if self.noisemodel is not None:
        #    for k in self.noisemodel.parameters.index:
        #        self.noisemodel.parameters.loc[k].value = self.paramdict[k]
    
        
        self.report = fit.report
        if report: print self.report

        # self.stats = Statistics(self)

    def check_series(self, tmin=None, tmax=None):
        """
        Function to check if the dependent and independent time series match.

        - tmin and tmax are in oseries.index for optimization.
        - at least one stress is available for simulation between tmin and tmax.
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
        if tmax is None:
            tmax = self.oseries.index.max()
        else:
            tmax = pd.tslib.Timestamp(tmax)

        # Check tmin and tmax compared to oseries and raise warning.
        if tmin not in self.oseries.index:
            print 'Warning, given tmin is outside of the oseries. First valid ' \
                  'index is %s' % self.oseries.first_valid_index()
        if tmax not in self.oseries.index:
            print 'Warning, given tmax is outside of the oseries. Last valid ' \
                  'index is %s' % self.oseries.last_valid_index()

        # Get maximum simulation period.
        tstmin = self.tserieslist[0].tmin
        tstmax = self.tserieslist[0].tmax

        for ts in self.tserieslist:
            if isinstance(ts, Constant):  # Check if it is not a constant tseries.
                pass
            else:
                if ts.tmin < tstmin:
                    tstmin = ts.tmin
                if ts.tmax > tstmax:
                    tstmax = ts.tmax

        self.tmin = tmin
        self.tmax = tmax

        # Check if chosen period is within or outside the maximum period.
        if tstmin > tmin:
            tmin = tstmin
        if tstmax < tmax:
            tmax = tstmax

        return tmin, tmax

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
            h = self.simulate(tmin=tmin, tmax=tmax)
            h.plot()
        if oseries:
            self.oseries.plot(linestyle='', marker='.', color='k', markersize=3)
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
        if self.noisemodel is not None:
            innovations = self.noisemodel.simulate(residuals, self.odelt)
            innovations.plot(label='innovations')
        plt.legend(loc=(0, 1), ncol=3, frameon=False, handlelength=3)
        plt.ylabel('Error [m]')
        plt.xlabel('Time [Years]')

        # Plot the Impulse Response Function
        ax3 = plt.subplot(gs[0, -1])
        n = 0
        for ts in self.tserieslist:
            p = self.parameters[n:n + ts.nparam]
            n += ts.nparam
            if "rfunc" in dir(ts):
                plt.plot(ts.rfunc.block(p))
        ax3.set_xticks(ax3.get_xticks()[::2])
        ax3.set_yticks(ax3.get_yticks()[::2])
        plt.title('Block Response')

        # Plot the Model Parameters (Experimental)
        ax4 = plt.subplot(gs[1:2, -1])
        ax4.xaxis.set_visible(False)
        ax4.yaxis.set_visible(False)
        text = np.vstack((self.paramdict.keys(), [round(float(i), 4) for i in
                                                  self.paramdict.values()])).T
        colLabels = ("Parameter", "Value")
        ytable = ax4.table(cellText=text, colLabels=colLabels, loc='center')
        ytable.scale(1, 1.1)

        # Table of the numerical diagnostic statistics.
        ax5 = plt.subplot(gs[2, -1])
        ax5.xaxis.set_visible(False)
        ax5.yaxis.set_visible(False)
        plt.text(0.05, 0.8, 'AIC: %.2f' % self.fit.aic)
        plt.text(0.05, 0.6, 'BIC: %.2f' % self.fit.bic)
        plt.show()
        if savefig:
            plt.savefig('.eps' % (self.name), bbox_inches='tight')
