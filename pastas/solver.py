from __future__ import print_function, division

import lmfit
import numpy as np
import pandas as pd

class Fit(object):
    def __init__(self, optimal_params, report):
        self.optimal_params = optimal_params
        self.report = report

class LmfitSolve:
    def __init__(self, parameters, ftol=1e-3, epsfcn=1e-4):
        """Solver based on lmfit

        Parameters
        ----------
        parameters : pd.DataFrame
            DataFrame with parameter bounds and initial values
        ftol : float, optional
            Relative error in the desired sum of squares
            see: http://cars9.uchicago.edu/software/python/lmfit/fitting.html
        epsfcn : float, optional
            variable used in determining a suitable step length for the forward- difference approximation of the Jacobian
            see: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.leastsq.html
        """
        self.ftol = ftol
        self.epsfcn = epsfcn

        # deal with parameters
        self.parameters = lmfit.Parameters()
        solve_params = parameters[['initial', 'pmin', 'pmax', 'vary']]
        for param_name, param_values in solve_params.iterrows():

            # set NaN to None
            param_kwargs = {k: None if np.isnan(v) else v
                for k, v in param_values.items()}

            # rename parameter kwargs
            param_kwargs.update({'value': param_kwargs.pop('initial')})
            param_kwargs.update({'min': param_kwargs.pop('pmin')})
            param_kwargs.update({'max': param_kwargs.pop('pmax')})

            # add to parameters
            self.parameters.add(param_name, **param_kwargs)

    def solve(self, objfunc, *objfunc_args, **objfunc_kwargs):
        """Summary

        Parameters
        ----------
        objfunc : function
            Objective function to be evaluated using lmfit.minize
        *objfunc_args
            Additional positional arguments for objective function
        **objfunc_kwargs
            Additional keyword arguments for objective function

        """

        # deploy minimize using objfunc
        fit = lmfit.minimize(fcn=objfunc, params=self.parameters,
                             ftol=self.ftol, epsfcn=self.epsfcn,
                             args=objfunc_args,
                             kws=objfunc_kwargs)

        # assign output attributes
        optimal_params = np.array([p.value for p in fit.params.values()])
        report = lmfit.fit_report(fit)

        return Fit(optimal_params, report=report)



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

        if self.noise:
            res = self.model.innovations(self.parameters, tmin=self.tmin,
                                         tmax=self.tmax, freq=self.freq,
                                         h_observed=self.model.oseries_calib)
        else:
            res = self.model.residuals(self.parameters, tmin=self.tmin,
                                       tmax=self.tmax, freq=self.freq,
                                       h_observed=self.model.oseries_calib)

        return sum(res ** 2)
