import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from checks import check_oseries


class Model:
    def __init__(self, oseries, xy=(0, 0), metadata=None, freq=None,
                 fillnan='drop'):
        self.oseries = check_oseries(oseries, freq, fillnan)
        self.xy = xy
        self.metadata = metadata
        self.odelt = self.oseries.index.to_series().diff() / np.timedelta64(1, 'D')
        # delt converted to days
        self.tserieslist = []
        self.noisemodel = None

    def addtseries(self, tseries):
        self.tserieslist.append(tseries)

    def addnoisemodel(self, noisemodel):
        self.noisemodel = noisemodel

    def simulate(self, t=None, p=None, noise=False):
        if t is None:
            t = self.oseries.index
        if p is None:
            p = self.parameters
        h = pd.Series(data=0, index=t)
        istart = 0
        for ts in self.tserieslist:
            h += ts.simulate(t, p[istart: istart + ts.nparam])
            istart += ts.nparam
        return h

    def residuals(self, parameters, tmin=None, tmax=None, solvemethod='lmfit',
                  noise=False):
        if tmin is None:
            tmin = self.oseries.index.min()
        if tmax is None:
            tmax = self.oseries.index.max()
        tindex = self.oseries[tmin: tmax].index  # times used for calibration
        if solvemethod == 'lmfit':  # probably needs to be a function call
            p = np.array([p.value for p in parameters.values()])
        if isinstance(parameters, np.ndarray):
            p = parameters
        # h_observed - h_simulated
        r = self.oseries[tindex] - self.simulate(tindex, p)
        if noise and (self.noisemodel is not None):
            r = self.noisemodel.simulate(r, self.odelt[tindex], tindex, p[-1])
        if np.isnan(sum(r ** 2)):
            print 'nan problem in residuals'  # quick and dirty check
        return r

    def solve(self, tmin=None, tmax=None, solvemethod='lmfit', report=True,
              noise=True):
        if noise and (self.noisemodel is None):
            print 'Warning, solution with noise model while noise model is not ' \
                  'defined. No noise model is used'
        self.solvemethod = solvemethod
        self.nparam = sum(ts.nparam for ts in self.tserieslist)
        if self.solvemethod == 'lmfit':
            parameters = lmfit.Parameters()
            for ts in self.tserieslist:
                for k in ts.parameters.index:
                    p = ts.parameters.loc[k]
                    pvalues = np.where(np.isnan(p.values), None,
                                       p.values)  # needed because lmfit doesn't
                    # take nan as input
                    parameters.add(k, value=pvalues[0], min=pvalues[1],
                                   max=pvalues[2], vary=pvalues[3])
            if self.noisemodel is not None:
                for k in self.noisemodel.parameters.index:
                    p = self.noisemodel.parameters.loc[k]
                    pvalues = np.where(np.isnan(p.values), None,
                                       p.values)  # needed because lmfit doesn't
                    # take nan as input
                    parameters.add(k, value=pvalues[0], min=pvalues[1],
                                   max=pvalues[2], vary=pvalues[3])
            self.lmfit_params = parameters
            self.fit = lmfit.minimize(fcn=self.residuals, params=parameters,
                                      ftol=1e-3, epsfcn=1e-4,
                                      args=(tmin, tmax, self.solvemethod, noise))
            if report: print lmfit.fit_report(self.fit)
            self.parameters = np.array([p.value for p in self.fit.params.values()])
            self.paramdict = self.fit.params.valuesdict()
            # Return parameters to tseries
            for ts in self.tserieslist:
                for k in ts.parameters.index:
                    ts.parameters.loc[k].value = self.paramdict[k]
            if self.noisemodel is not None:
                for k in self.noisemodel.parameters.index:
                    self.noisemodel.parameters.loc[k].value = self.paramdict[k]

    def plot(self, oseries=True):
        h = self.simulate()
        plt.figure()
        h.plot()
        if oseries:
            self.oseries.plot(style='ro')
        plt.show()

    def plot_results(self, savefig=False):
        plt.figure('Model Results', facecolor='white', figsize=(16, 12))
        gs = plt.GridSpec(3, 4, wspace=0.2)
        # Plot the Groundwater levels
        h = self.simulate()
        ax1 = plt.subplot(gs[:2, :-1])
        h.plot(label='modeled head')
        self.oseries.plot(linestyle='', marker='.', color='k', markersize=3,
                          label='observed head')
        ax1.xaxis.set_visible(False)
        plt.legend(loc=(0, 1), ncol=3, frameon=False, handlelength=3)
        plt.ylabel('Head [m]')
        # Plot the residuals and innovations
        residuals = self.oseries - h
        ax2 = plt.subplot(gs[2, :-1], sharex=ax1)
        residuals.plot(color='k', label='residuals')
        if self.noisemodel is not None:
            innovations = pd.Series(self.noisemodel.simulate(residuals,
                                                             self.odelt,
                                                             p=self.parameters[-1]),
                                    index=residuals.index)
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
