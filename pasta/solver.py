import numpy as np
import pandas as pd
import lmfit

def lmfit_solve(model, tmin=None, tmax=None, noise=True, initialize=True,
                report=True):
    if initialize: model.initialize()
    parameters = lmfit.Parameters()
    for k in model.parameters.index:
        p = model.parameters.loc[k]
        # needed because lmfit doesn't take nan as input
        pvalues = np.where(np.isnan(p.values), None, p.values)
        parameters.add(k, value=pvalues[0], min=pvalues[1],
                       max=pvalues[2], vary=pvalues[3])
    fit = lmfit.minimize(fcn=lmfit_obj_function, params=parameters,
                         ftol=1e-3, epsfcn=1e-4,
                         args=(tmin, tmax, noise, model))
    if report: print lmfit.fit_report(fit)
    return np.array([p.value for p in fit.params.values()])

    # self.parameters = np.array(
    #     [p.value for p in self.fit.params.values()])
    # self.paramdict = self.fit.params.valuesdict()
    # # Return parameters to tseries
    # for ts in self.tserieslist:
    #     for k in ts.parameters.index:
    #         ts.parameters.loc[k].value = self.paramdict[k]
    # if self.noisemodel is not None:
    #     for k in self.noisemodel.parameters.index:
    #         self.noisemodel.parameters.loc[k].value = self.paramdict[k]

    # Make the Statistics class available after optimization
    #self.stats = Statistics(self)

def lmfit_obj_function(parameters, tmin, tmax, noise, model):
    p = np.array([p.value for p in parameters.values()])
    return model.residuals(p, tmin, tmax, noise)

