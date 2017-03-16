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
            return model.innovations(parameters, tmin, tmax, freq)
        else:
            return model.residuals(parameters, tmin, tmax, freq)


class LmfitSolve:
    def __init__(self, model, tmin=None, tmax=None, noise=True, freq='D'):
        # Deal with the parameters
        parameters = lmfit.Parameters()
        p = model.parameters[['initial', 'pmin', 'pmax', 'vary']]
        for k in p.index:
            pp = np.where(np.isnan(p.loc[k]), None, p.loc[k])
            parameters.add(k, value=pp[0], min=pp[1], max=pp[2], vary=pp[3])

        self.fit = lmfit.minimize(fcn=self.objfunction, params=parameters,
                                  ftol=1e-3, epsfcn=1e-4,
                                  args=(tmin, tmax, noise, model, freq))
        self.optimal_params = np.array([p.value for p in
                                        self.fit.params.values()])
        self.report = lmfit.fit_report(self.fit)

    def objfunction(self, parameters, tmin, tmax, noise, model, freq):
        p = np.array([p.value for p in parameters.values()])
        if noise:
            return model.innovations(p, tmin, tmax, freq)
        else:
            return model.residuals(p, tmin, tmax, freq)


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
        self.fit = differential_evolution(self.objfunction,
                                        zip(self.pmin, self.pmax))
        self.optimal_params = self.model.parameters.initial.values
        self.optimal_params[self.vary] = self.fit.values()[3]
        self.report = str(self.fit)

    def objfunction(self, parameters):
        print('.'),
        self.parameters[self.vary] = parameters

        if self.noise:
            res = self.model.innovations(self.parameters, tmin=self.tmin,
                                         tmax=self.tmax, freq=self.freq)
        else:
            res = self.model.residuals(self.parameters, tmin=self.tmin,
                                       tmax=self.tmax, freq=self.freq)

        return sum(res ** 2)
