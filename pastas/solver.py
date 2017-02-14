from __future__ import print_function, division

import lmfit
import numpy as np


class LmfitSolve:
    def __init__(self, model, tmin=None, tmax=None, noise=True, freq='D'):
        parameters = lmfit.Parameters()
        p = model.parameters[['initial', 'pmin', 'pmax', 'vary']]
        for k in p.index:
            pp = np.where(np.isnan(p.loc[k]), None, p.loc[k])

            parameters.add(k, value=pp[0], min=pp[1],
                           max=pp[2], vary=pp[3])
        fit = lmfit.minimize(fcn=self.objfunction, params=parameters,
                             ftol=1e-3, epsfcn=1e-4,
                             args=(tmin, tmax, noise, model, freq))
        self.optimal_params = np.array([p.value for p in fit.params.values()])
        self.report = lmfit.fit_report(fit)

    def objfunction(self, parameters, tmin, tmax, noise, model, freq):
        p = np.array([p.value for p in parameters.values()])
        return model.residuals(p, tmin, tmax, freq, noise,
                               h_observed=model.oseries_calib)


# def lmfit_solve(model, tmin=None, tmax=None, noise=True, report=True):
#    parameters = lmfit.Parameters()
#    for k in model.parameters.index:
#        p = model.parameters.loc[k]
#        # needed because lmfit doesn't take nan as input
#        pvalues = np.where(np.isnan(p.values), None, p.values)
#        parameters.add(k, value=pvalues[0], min=pvalues[1],
#                       max=pvalues[2], vary=pvalues[3])
#    fit = lmfit.minimize(fcn=lmfit_obj_function, params=parameters,
#                         ftol=1e-3, epsfcn=1e-4,
#                         args=(tmin, tmax, noise, model))
#    if report: print lmfit.fit_report(fit)
#    return np.array([p.value for p in fit.params.values()])


from scipy.optimize import differential_evolution


class DESolve:
    def __init__(self, model, tmin=None, tmax=None, noise=True, freq='D'):
        self.freq = freq
        self.model = model
        self.tmin = tmin
        self.tmax = tmax
        self.noise = noise
        self.parameters = self.model.parameters.initial.values
        self.vary = self.model.parameters.vary.values.astype('bool')
        self.pmin = self.model.parameters.pmin.values[self.vary]
        self.pmax = self.model.parameters.pmax.values[self.vary]
        result = differential_evolution(self.objfunction,
                                        zip(self.pmin, self.pmax))
        self.optimal_params = self.model.parameters.initial.values
        self.optimal_params[self.vary] = result.values()[3]
        self.report = str(result)

    def objfunction(self, parameters):
        print('.'),
        self.parameters[self.vary] = parameters
        res = self.model.residuals(self.parameters, tmin=self.tmin,
                                   tmax=self.tmax, freq=self.freq,
                                   noise=self.noise,
                                   h_observed=self.model.oseries_calib)
        return sum(res ** 2)
