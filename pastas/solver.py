"""The solver module contains the different solvers that are available for
PASTAS.

All solvers inherit from the BaseSolver class, which contains general method
for selecting the correct time series to minimize and options to weight the
residuals or innovations series.

Notes
-----
By default, when a model is solved with a noisemodel, the swsi-weights are
applied.

Examples
--------
To solve a model the following syntax can be used:

>>> ml.solve(solver=LeastSquares)

"""

from __future__ import print_function, division

import logging

import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares, differential_evolution

logger = logging.getLogger(__name__)


class BaseSolver:
    _name = "BaseSolver"
    __doc__ = """Basesolver class that contains the basic function for each 
    solver.

    A solver is implemented with a separate init method and objective function
    that returns the necessary format that is required by the specific solver.
    The objective function calls the get_residuals method of the BaseSolver
    class, which calculates the residuals or innovations (depending on the
    noise keyword) and applies weights (depending on the weights keyword).

    """

    def __init__(self):

        # Parameters attributes
        self.popt = None # Optimal values of the parameters
        self.stderr = None # Standard error of parameters
        self.pcor = None #Correlation between parameters
        self.pcov = None # Covariances of the parameters

        # Optimization attributes
        self.nfev = None # number of function evaluations
        self.fit = None # Object that is returned by the optimization method




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

    def __init__(self, model, tmin=None, tmax=None, noise=True, freq=None,
                 weights=None, **kwargs):
        BaseSolver.__init__(self)

        parameters = model.parameters.initial.values

        # Set the boundaries
        pmin = np.where(model.parameters.pmin.isnull(), -np.inf,
                        model.parameters.pmin)
        pmax = np.where(model.parameters.pmax.isnull(), np.inf,
                        model.parameters.pmax)
        bounds = (pmin, pmax)

        if False in model.parameters.vary.values.astype('bool'):
            logger.warning("""Fixing parameters is not supported with this
                           solver. Please use LmfitSolve or apply small
                           boundaries as a solution.""")

        self.fit = least_squares(self.objfunction, x0=parameters,
                                 bounds=bounds,
                                 args=(tmin, tmax, noise, model, freq,
                                       weights), **kwargs)

        self.nfev = self.fit.nfev

        self.pcov = self.get_covariances(self.fit, model)
        self.pcor = self.get_correlations(self.pcov)
        self.stderr = np.sqrt(np.diag(self.pcov))
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

    def get_covariances(self, res, model, absolute_sigma=False):
        """Method to get the covariance matrix from the jacobian.

        Parameters
        ----------
        res

        Returns
        -------
        pcov: numpy.array
            numpy array with the covariance matrix.

        Notes
        -----
        This method os copied from Scipy, please refer to:
        https://github.com/scipy/scipy/blob/v1.0.0/scipy/optimize/optimize.py

        """
        cost = 2 * res.cost  # res.cost is half sum of squares!

        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s ** 2, VT)
        n_param = model.parameters.index.size
        warn_cov = False
        if pcov is None:
            # indeterminate covariance
            pcov = np.zeros((n_param, n_param), dtype=float)
            pcov.fill(np.inf)
            warn_cov = True
        elif not absolute_sigma:
            if model.oseries.index.size > n_param:
                s_sq = cost / (model.oseries.index.size - n_param)
                pcov = pcov * s_sq
            else:
                pcov.fill(np.inf)
                warn_cov = True

        if warn_cov:
            logger.warning(
                'Covariance of the parameters could not be estimated')

        return pcov

    def get_correlations(self, pcov):
        pcor = None
        return pcor


class LmfitSolve(BaseSolver):
    _name = "LmfitSolve"
    __doc__ = """Solving the model using the LmFit solver. This is basically a
    wrapper around the scipy solvers, adding some cool functionality for
    boundary conditions.

    Notes
    -----
    https://github.com/lmfit/lmfit-py/

    """

    def __init__(self, model, tmin=None, tmax=None, noise=True, freq=None,
                 weights=None, ftol=1e-3, epsfcn=1e-4, **kwargs):
        import lmfit # Import Lmfit here, so it is no dependency
        BaseSolver.__init__(self)

        # Deal with the parameters
        parameters = lmfit.Parameters()
        p = model.parameters[['initial', 'pmin', 'pmax', 'vary']]
        for k in p.index:
            pp = np.where(p.loc[k].isnull(), None, p.loc[k])
            parameters.add(k, value=pp[0], min=pp[1], max=pp[2], vary=pp[3])

        self.fit = lmfit.minimize(fcn=self.objfunction, params=parameters,
                                  args=(tmin, tmax, noise, model, freq,
                                        weights), ftol=ftol, epsfcn=epsfcn,
                                  **kwargs)

        # Set all parameter attributes
        self.optimal_params = np.array([p.value for p in
                                        self.fit.params.values()])
        self.stderr = np.array([p.stderr for p in self.fit.params.values()])
        if self.fit.covar is not None:
            self.pcov = self.fit.covar

        # Set all optimization attributes
        self.nfev = self.fit.nfev
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
