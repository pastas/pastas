from __future__ import print_function, division

from warnings import warn

import lmfit
import numpy as np
from scipy.optimize import least_squares, differential_evolution


class BaseSolver:
    """ Basesolver class that contains the basic function for each solver.

    A solver is implemented with a separate init method and objective function
    that returns the necessary format that is required by the specific solver.
    The objective function calls the get_residuals method of the BaseSolver
    class, which calculates the residuals or innovations (depending on the
    noise keyword) and applies weights (depending on the weights keyword).

    """

    def __init__(self):
        pass

    def get_residuals(self, parameters, tmin, tmax, noise, model, freq,
                      weights=None):
        # Determine if a noise model needs to be applied
        if noise:
            res = model.innovations(parameters, tmin, tmax, freq)
        else:
            res = model.residuals(parameters, tmin, tmax, freq)

        # Determine if weights need to be applied
        if weights is None:
            return res
        elif type(weights) == list:
            if len(weights) != res.size:
                warn("Provided weights list does not match the size of the "
                     "residuals series. A list with size %s is needed."
                     % res.size)
            return weights * res
        elif hasattr(self, weights):
            weights = getattr(self, weights)
            w = weights(parameters, model, res)
            res = res.multiply(w, fill_value=0.0)
            return res
        else:
            warn("The weighting option is not valid. Please provide a valid "
                 "weighting argument.")

    def swsi(self, parameters, model, res):
        """

        Returns
        -------

        """
        p = parameters[-1]
        delt = model.odelt[res.index][1:]
        power = (1.0 / (2.0 * (len(delt) - 1.0)))
        w = np.exp(power * np.sum(np.log(1.0 - np.exp(-2.0 * delt / p)))) / \
            np.sqrt(1.0 - np.exp(-2.0 * delt / p))
        return w

    def swsi2(self, parameters, model, res):
        alpha = parameters[-1]
        dt = model.odelt[res.index][1:]
        N = res.index.size  # Number of innovations
        numerator = np.exp(
            (1.0 / N) * sum(np.log(1.0 - np.exp(-2.0 * dt / alpha))))
        w = np.sqrt((numerator / (1.0 - np.exp(-2.0 * dt / alpha))))
        return w

    def time_step(self, parameters, model, res):
        delt = model.odelt[res.index][1:]
        return delt


class LeastSquares(BaseSolver):
    """Solving the model using Scipy's least_squares method

    Notes
    -----
    This method uses the
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    """

    def __init__(self, model, tmin=None, tmax=None, noise=True, freq='D',
                 weights=None):
        BaseSolver.__init__(self)
        parameters = model.parameters.initial.values

        # Set the boundaries
        pmin = np.where(model.parameters.pmin.isnull(), -np.inf, model.parameters.pmin)
        pmax = np.where(model.parameters.pmax.isnull(), np.inf, model.parameters.pmax)
        bounds = (pmin, pmax)

        # Set boundaries to initial values if vary is False
        # TODO: make notification that fixing parameters is not (yet)
        # supported.
        if False in model.parameters.vary.values.astype('bool'):
            warn("Fixing parameters is not supported with this solver. Please"
                 "use LmfitSolve or apply small boundaries as a solution.")
        self.fit = least_squares(self.objfunction, x0=parameters,
                                 bounds=bounds, ftol=1e-3,
                                 args=(
                                     tmin, tmax, noise, model, freq, weights))
        self.optimal_params = self.fit.x
        self.report = None

    def objfunction(self, parameters, tmin, tmax, noise, model, freq, weights):
        res = self.get_residuals(parameters, tmin, tmax, noise, model, freq,
                                 weights)
        return res


class LmfitSolve:
    def __init__(self, model, tmin=None, tmax=None, noise=True, freq='D',
                 weights=None):
        # Deal with the parameters
        parameters = lmfit.Parameters()
        p = model.parameters[['initial', 'pmin', 'pmax', 'vary']]
        for k in p.index:
            pp = np.where(p.loc[k].isnull(), None, p.loc[k])
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
