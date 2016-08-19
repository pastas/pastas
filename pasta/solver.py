import numpy as np
import pandas as pd
import lmfit


class LmfitSolve:
    def __init__(self, model, tmin=None, tmax=None, noise=True):
        parameters = lmfit.Parameters()
        p = model.parameters[['initial', 'pmin', 'pmax', 'vary']]
        for k in p.index:
            pp = np.where(np.isnan(p.loc[k]), None, p.loc[k])
 
            parameters.add(k, value=pp[0], min=pp[1],
                           max=pp[2], vary=pp[3])
        print parameters
        fit = lmfit.minimize(fcn=self.objfunction, params=parameters,
                             ftol=1e-3, epsfcn=1e-4,
                             args=(tmin, tmax, noise, model))
        self.optimal_params = np.array([p.value for p in fit.params.values()])
        self.report = lmfit.fit_report(fit)

    def objfunction(self, parameters, tmin, tmax, noise, model):
        p = np.array([p.value for p in parameters.values()])
        return model.residuals(p, tmin, tmax, noise)


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
#
# def lmfit_obj_function(parameters, tmin, tmax, noise, model):
#    p = np.array([p.value for p in parameters.values()])
#    return model.residuals(p, tmin, tmax, noise)

from scipy.optimize import differential_evolution


class DESolve:
    def __init__(self, model, tmin=None, tmax=None, noise=True):
        self.model = model
        self.tmin = tmin
        self.tmax = tmax
        self.noise = noise
        result = differential_evolution(self.objfunction,
                                        zip(self.model.parameters['pmin'],
                                            self.model.parameters['pmax']))
        self.optimal_params = result.values()[3]
        self.report = str(result)

    def objfunction(self, parameters):
        print '.',
        return self.model.sse(parameters, tmin=self.tmin, tmax=self.tmax,
                              noise=self.noise)
