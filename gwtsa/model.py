import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lmfit

class Model:
    def __init__(self, oseries):
        self.oseries = oseries
        self.odelt = self.oseries.index.to_series().diff() / np.timedelta64(1,'D')  # delt converted to days
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
    def residuals(self, parameters, tmin=None, tmax=None, solvemethod='lmfit', noise=False):
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
            r = self.noisemodel.simulate(r, self.odelt, tindex, p[-1])
        if sum(r**2) is np.nan:
            print 'nan problem in residuals'  # quick and dirty check
        return r
    def solve(self, tmin=None, tmax=None, solvemethod='lmfit', report=True, noise=True):
        if noise and (self.noisemodel is None):
            print 'Warning, solution with noise model while noise model is not defined. No noise model is used'
        self.solvemethod = solvemethod
        self.nparam = sum(ts.nparam for ts in self.tserieslist)
        if self.solvemethod == 'lmfit':
            parameters = lmfit.Parameters()
            for ts in self.tserieslist:
                for k in ts.parameters.index:
                    p = ts.parameters.loc[k]
                    pvalues = np.where(np.isnan(p.values), None, p.values)  # needed because lmfit doesn't take nan as input
                    parameters.add(k, value=pvalues[0], min=pvalues[1], max=pvalues[2], vary=pvalues[3])
            if self.noisemodel is not None:
                for k in self.noisemodel.parameters.index:
                    p = self.noisemodel.parameters.loc[k]
                    pvalues = np.where(np.isnan(p.values), None, p.values)  # needed because lmfit doesn't take nan as input
                    parameters.add(k, value=pvalues[0], min=pvalues[1], max=pvalues[2], vary=pvalues[3])
            self.lmfit_params = parameters
            self.fit = lmfit.minimize(fcn=self.residuals, params=parameters, ftol=1e-3, epsfcn=1e-4, args=(tmin, tmax, self.solvemethod, noise))
            if report: print lmfit.fit_report(self.fit)
            self.parameters = np.array([p.value for p in self.fit.params.values()])
            self.paramdict = self.fit.params.valuesdict()
    def plot(self, oseries=True):
        h = self.simulate()
        plt.figure()
        h.plot()
        if oseries:
            self.oseries.plot(style='ro')
        plt.show()