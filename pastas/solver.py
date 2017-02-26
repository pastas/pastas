from __future__ import print_function, division

from warnings import warn

import lmfit
import numpy as np
from scipy.optimize import least_squares


class LeastSquares:
    """Solving the model using Scipy's least_squares method

    Notes
    -----
    This method uses the
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    """

    def __init__(self, model, tmin=None, tmax=None, noise=True, freq='D'):
        parameters = model.parameters.initial.values

        # Set the boundaries
        pmin = model.parameters.pmin.values
        pmin[np.isnan(pmin)] = -np.inf
        pmax = model.parameters.pmax.values
        pmax[np.isnan(pmax)] = np.inf
        bounds = (pmin, pmax)

        # Set boundaries to initial values if vary is False
        # TODO: make notification that fixing parameters is not (yet)
        # supported.
        if False in model.parameters.vary.values.astype('bool'):
            warn("Fixing parameters is not supported with this solver. Please"
                 "use LmfitSolve or apply small boundaries as a solution.")

        self.fit = least_squares(self.objfunction, x0=parameters,
                                 bounds=bounds,
                                 args=(tmin, tmax, noise, model, freq))
        self.optimal_params = self.fit.x
        self.report = None

    def objfunction(self, parameters, tmin, tmax, noise, model, freq):
        if noise:
            return model.innovations(parameters, tmin, tmax, freq,
                                     model.oseries_calib)
        else:
            return model.residuals(parameters, tmin, tmax, freq,
                                   model.oseries_calib)


def params_to_array(objfunc):
    """Objective function wrapper for lmfit.
    Unpacking lmfit.Parameters object to array before passing to
    objective function.

    Parameters
    ----------
    objfunc : function
        objective function

    Returns
    -------
    function
        wrapped objective function taking parameter values as array
    """
    def wrapper(parameters, *args, **kwargs):
        p = np.array([p.value for p in parameters.values()])
        return objfunc(p, *args, **kwargs)
    return wrapper


class Fit(object):
    """Generic fit class containing the solver results

    Attributes
    ----------
    optimal_params : np.array
        Array with optimal parameter values
    report : str
        fit report string
    """
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
        """Solve using objective functions

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
        fit = lmfit.minimize(fcn=params_to_array(objfunc),
                             params=self.parameters,
                             ftol=self.ftol, epsfcn=self.epsfcn,
                             args=objfunc_args,
                             kws=objfunc_kwargs)

        # assign output attributes
        optimal_params = np.array([p.value for p in fit.params.values()])
        report = lmfit.fit_report(fit)

        return Fit(optimal_params, report=report)


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


