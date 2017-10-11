"""This module contains the different solvers that are available for PASTAS.

All solvers inherit from the BaseSolver class, which contains general method
for selecting the correct time series to minimize and options to weight the
residuals series.

Notes
-----
By default, when a model is solve with a noisemodel, the swsi-weights are
applied. The use of these weights is necessary to obtain statistically sound
results when applying a noisemodel.

Examples
--------
To solve a model without a noisemodel and no weighting of the residuals,
the following syntax can be used.

>>> ml.solve(solver=LeastSquares, noise=False, weights=None)

To solve the model with a noise model, an the weights according to the swsi
criterion (default), use te following syntax.

>>> ml.solve(solver=LmfitSolve)

"""

from __future__ import print_function, division

import logging

import lmfit
import numpy as np
from scipy.optimize import least_squares, differential_evolution

logger = logging.getLogger(__name__)


class BaseSolver:
    _name = "BaseSolver"
    __doc__ = """Basesolver class that contains the basic function for each solver.

    A solver is implemented with a separate init method and objective function
    that returns the necessary format that is required by the specific solver.
    The objective function calls the get_residuals method of the BaseSolver
    class, which calculates the residuals or innovations (depending on the
    noise keyword) and applies weights (depending on the weights keyword).

    """

    def __init__(self):
        self.default_kwargs = dict()

    def update_kwargs(self, kwargs):
        if kwargs:
            self.default_kwargs.update(kwargs)
            return self.default_kwargs
        else:
            return self.default_kwargs

    def minimize(self, parameters, tmin, tmax, noise, model, freq,
                 weights=None):
        """This method is called by all solvers to obtain a series that are
        minimized in the optimization proces. It handles the application of
        the weigths, a noisemodel and other optimization options.

        Parameters
        ----------
        parameters: list, numpy.ndarray
            list with the parameters
        tmin: str

        tmax: str

        noise: Boolean

        model: pastas.Model
            Pastas Model instance
        freq: str

        weights: pandas.Series
            pandas Series by which the residual or innovation series are
            multiplied. Typically values between 0 and 1.


        Returns
        -------
        res:
            residuals series

        """

        # Get the residuals or the innovations
        if noise:
            res = model.innovations(parameters, tmin, tmax, freq)
        else:
            res = model.residuals(parameters, tmin, tmax, freq)

        # Determine if weights need to be applied
        if weights is not None:
            weights = weights.reindex(res.index)
            weights.fillna(1.0, inplace=True)
            res = res.multiply(weights)

        return res


class LeastSquares(BaseSolver):
    _name = "LeastSquares"
    __doc__ = """Solving the model using Scipy's least_squares method.

    Notes
    -----
    This class is usually called by the pastas Model solve method. E.g.

    >>> ml.solve(solver=LeastSquares)

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    """

    def __init__(self, model, tmin=None, tmax=None, noise=True, freq='D',
                 weights=None, **kwargs):
        BaseSolver.__init__(self)

        # Update the kwargs going to the solver
        self.default_kwargs = dict(ftol=1e-3)
        kwargs = self.update_kwargs(kwargs)

        parameters = model.parameters.initial.values

        # Set the boundaries
        pmin = np.where(model.parameters.pmin.isnull(), -np.inf,
                        model.parameters.pmin)
        pmax = np.where(model.parameters.pmax.isnull(), np.inf,
                        model.parameters.pmax)
        bounds = (pmin, pmax)

        if False in model.parameters.vary.values.astype('bool'):
            logger.warning("Fixing parameters is not supported with this"
                           "solver. Please use LmfitSolve or apply small"
                           "boundaries as a solution.")

        self.fit = least_squares(self.objfunction, x0=parameters,
                                 bounds=bounds,
                                 args=(tmin, tmax, noise, model, freq,
                                       weights), **kwargs)
        self.optimal_params = self.fit.x
        self.report = None

    def objfunction(self, parameters, tmin, tmax, noise, model, freq, weights):
        """

        Parameters
        ----------
        parameters
        tmin
        tmax
        noise
        model
        freq
        weights

        Returns
        -------

        """
        res = self.minimize(parameters, tmin, tmax, noise, model, freq,
                            weights)
        return res


class LmfitSolve(BaseSolver):
    _name = "LmfitSolve"
    __doc__ = """Solving the model using the LmFit solver. This is basically a
    wrapper around the scipy solvers, adding some cool functionality for
    boundary conditions.

    Notes
    -----
    https://github.com/lmfit/lmfit-py/

    """

    def __init__(self, model, tmin=None, tmax=None, noise=True, freq='D',
                 weights=None, **kwargs):
        BaseSolver.__init__(self)

        # Update the kwargs going to the solver
        self.default_kwargs = dict(ftol=1e-3, epsfcn=1e-4)
        kwargs = self.update_kwargs(kwargs)

        # Deal with the parameters
        parameters = lmfit.Parameters()
        p = model.parameters[['initial', 'pmin', 'pmax', 'vary']]
        for k in p.index:
            pp = np.where(p.loc[k].isnull(), None, p.loc[k])
            parameters.add(k, value=pp[0], min=pp[1], max=pp[2], vary=pp[3])

        self.fit = lmfit.minimize(fcn=self.objfunction, params=parameters,
                                  args=(tmin, tmax, noise, model, freq,
                                        weights), **kwargs)
        self.optimal_params = np.array([p.value for p in
                                        self.fit.params.values()])
        self.report = lmfit.fit_report(self.fit)

    def objfunction(self, parameters, tmin, tmax, noise, model, freq, weights):
        param = np.array([p.value for p in parameters.values()])
        res = self.minimize(param, tmin, tmax, noise, model, freq,
                            weights)
        return res


class DESolve(BaseSolver):
    _name = "DESolve"

    def __init__(self, model, tmin=None, tmax=None, noise=True, freq='D'):
        BaseSolver.__init__(self)

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
