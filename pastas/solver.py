"""The solver module contains the different solvers that are available for
PASTAS.

All solvers inherit from the BaseSolver class, which contains general method
for selecting the correct time series to misfit and options to weight the
residuals or noise series.

Notes
-----
By default, when a model is solved with a noisemodel, the swsi-weights are
applied.

Examples
--------
To solve a model the following syntax can be used:

>>> ml.solve(solver=LeastSquares)

"""

from logging import getLogger

import numpy as np
from pandas import DataFrame, Series
from scipy.linalg import svd
from scipy.optimize import least_squares, differential_evolution, fmin

logger = getLogger(__name__)


class BaseSolver:
    _name = "BaseSolver"
    __doc__ = """Basesolver class that contains the basic function for each 
    solver.

    A solver is implemented with a separate init method and objective function
    that returns the necessary format that is required by the specific solver.
    The objective function calls the get_residuals method of the BaseSolver
    class, which calculates the residuals or noise (depending on the
    noise keyword) and applies weights (depending on the weights keyword).

    """

    def __init__(self):

        # Parameters attributes
        self.popt = None  # Optimal values of the parameters
        self.stderr = None  # Standard error of parameters
        self.pcor = None  # Correlation between parameters
        self.pcov = None  # Covariances of the parameters

        # Optimization attributes
        self.nfev = None  # number of function evaluations
        self.fit = None  # Object that is returned by the optimization method

    def misfit(self, parameters, noise, model, weights=None, callback=None):
        """This method is called by all solvers to obtain a series that are
        minimized in the optimization proces. It handles the application of
        the weigths, a noisemodel and other optimization options.

        Parameters
        ----------
        parameters: list, numpy.ndarray
            list with the parameters
        noise: Boolean
        model: pastas.Model
            Pastas Model instance
        weights: pandas.Series, optional
            pandas Series by which the residual or noise series are
            multiplied. Typically values between 0 and 1.
        callback: ufunc, optional
            function that is called after each iteration. the parameters are
            provided to the func. E.g. "callback(parameters)"

        Returns
        -------
        rv:
            residuals series (if noise=False) or noise series (if noise=True)

        """

        # Get the residuals or the noise
        if noise:
            rv = model.noise(parameters)
        else:
            rv = model.residuals(parameters)

        # Determine if weights need to be applied
        if weights is not None:
            weights = weights.reindex(rv.index)
            weights.fillna(1.0, inplace=True)
            rv = rv.multiply(weights)

        if callback:
            callback(parameters)

        return rv


class LeastSquares(BaseSolver):
    """Solver based on Scipy's least_squares method [1]_.

    Notes
    -----
    This class is the default solve method called by the pastas Model solve
    method. All kwargs provided to the Model.solve() method are forwarded to
    the solver. From there, they are forwarded to scipy least_squares solver.

    Examples
    --------

    >>> ml.solve(solver=LeastSquares)

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    """
    _name = "LeastSquares"

    def __init__(self, model, noise=True, weights=None, callback=None,
                 **kwargs):
        BaseSolver.__init__(self)

        self.modelparameters = model.parameters
        self.vary = self.modelparameters.vary.values.astype('bool')
        self.initial = self.modelparameters.initial.values.copy()
        parameters = self.modelparameters.loc[self.vary]

        # Set the boundaries
        pmin = np.where(parameters.pmin.isnull(), -np.inf, parameters.pmin)
        pmax = np.where(parameters.pmax.isnull(), np.inf, parameters.pmax)
        bounds = (pmin, pmax)

        self.fit = least_squares(self.objfunction,
                                 x0=parameters.initial.values, bounds=bounds,
                                 args=(noise, model, weights, callback),
                                 **kwargs)

        self.nfev = self.fit.nfev

        pcov = self.get_covariances(self.fit, model)
        # self.pcor = self.get_correlations(self.pcov)

        # sig, pcov, pcor = self.get_covcorrmatrix(model)
        self.pcov = DataFrame(pcov, index=parameters.index,
                              columns=parameters.index)
        self.pcor = DataFrame(None, index=parameters.index,
                              columns=parameters.index)

        self.optimal_params = self.initial
        self.optimal_params[self.vary] = self.fit.x
        self.stderr = np.zeros(len(self.optimal_params))
        self.stderr[self.vary] = np.sqrt(np.diag(self.pcov))
        self.report = None

    def objfunction(self, parameters, noise, model, weights, callback):
        """

        Parameters
        ----------
        parameters
        noise
        model
        weights

        Returns
        -------

        """
        p = self.initial
        p[self.vary] = parameters

        res = self.misfit(p, noise, model, weights, callback)

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
            if model.oseries.series.index.size > n_param:
                s_sq = cost / (model.oseries.series.index.size - n_param)
                pcov = pcov * s_sq
            else:
                pcov.fill(np.inf)
                warn_cov = True

        if warn_cov:
            logger.warning(
                'Covariance of the parameters could not be estimated')

        return pcov


class LmfitSolve(BaseSolver):
    """Solving the model using the LmFit solver [LM]_. This is basically a
    wrapper around the scipy solvers, adding some cool functionality for
    boundary conditions.

    References
    ----------
    .. [LM] https://github.com/lmfit/lmfit-py/

    """
    _name = "LmfitSolve"

    def __init__(self, model, noise=True, weights=None, **kwargs):
        import lmfit  # Import Lmfit here, so it is no dependency
        BaseSolver.__init__(self)

        # Deal with the parameters
        parameters = lmfit.Parameters()
        p = model.parameters[['initial', 'pmin', 'pmax', 'vary']]
        for k in p.index:
            pp = np.where(p.loc[k].isnull(), None, p.loc[k])
            parameters.add(k, value=pp[0], min=pp[1], max=pp[2], vary=pp[3])

        # set ftol and epsfcn if no options for lmfit are provided. Only
        # work with Lmfit's least squares solver method.
        if not kwargs:
            kwargs = {"ftol": 1e-3, "epsfcn": 1e-4}

        self.fit = lmfit.minimize(fcn=self.objfunction, params=parameters,
                                  args=(noise, model, weights), **kwargs)

        # Set all parameter attributes
        self.optimal_params = np.array([p.value for p in
                                        self.fit.params.values()])
        self.stderr = np.array([p.stderr for p in self.fit.params.values()])
        if self.fit.covar is not None:
            self.pcov = self.fit.covar

        # Set all optimization attributes
        self.nfev = self.fit.nfev
        self.report = lmfit.fit_report(self.fit)

    def objfunction(self, parameters, noise, model, weights):
        param = np.array([p.value for p in parameters.values()])
        res = self.misfit(param, noise, model, weights)
        return res


class DESolve(BaseSolver):
    _name = "DESolve"

    def __init__(self, model, noise=True):
        BaseSolver.__init__(self)

        self.model = model
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
            res = self.model.noise(self.parameters)
        else:
            res = self.model.residuals(self.parameters)

        return sum(res ** 2)


class MarkSolver(BaseSolver):
    """Experimental solver
    """
    _name = "MarkSolver"

    def __init__(self, model, noise=True, weights=None, **kwargs):
        BaseSolver.__init__(self)

        self.modelparameters = model.parameters
        self.vary = self.modelparameters.vary.values.astype('bool')
        self.initial = self.modelparameters.initial.values.copy()
        parameters = self.modelparameters.loc[self.vary]

        # Set the boundaries
        pmin = np.where(parameters.pmin.isnull(), -np.inf, parameters.pmin)
        pmax = np.where(parameters.pmax.isnull(), np.inf, parameters.pmax)
        bounds = (pmin, pmax)

        self.fit = fmin(self.objfunction, x0=parameters.initial.values,
                        args=(noise, model, weights), **kwargs)

        self.optimal_params = self.initial
        self.optimal_params[self.vary] = self.fit
        self.stderr = np.zeros(len(self.optimal_params))
        self.report = None

    def objfunction(self, parameters, noise, model, weights):
        """

        Parameters
        ----------
        parameters
        noise
        model
        weights

        Returns
        -------

        """
        p = self.initial
        p[self.vary] = parameters

        rv = self.misfit(p, noise, model, weights)

        return rv

    def get_covcorrmatrix(self, model):
        """Method to compute sigma, covariance and correlation matrix

        TODO: make it work
        """
        nparam = len(self.fit.x)
        H = self.fit.jac.T @ self.fit.jac
        sigsq = np.var(self.fit.fun, ddof=nparam)
        covmat = np.linalg.inv(H) * sigsq
        stderr = np.sqrt(np.diag(covmat))
        D = np.diag(1 / stderr)
        corrmat = D @ covmat @ D

        return stderr, covmat, corrmat

    def misfit(self, parameters, noise, model, weights=None):
        res = model.residuals(parameters)
        alpha = parameters[-1]
        print('alpha:', alpha)
        odelt = model.odelt.loc[res.index]
        noise = Series(data=res)
        noise.iloc[1:] -= np.exp(-odelt[1:] / alpha) * res.values[:-1]

        res = res[1:]
        noise = noise[1:]
        delt = odelt[1:]
        sigres = np.std(res)
        sigi = sigres * np.sqrt(1 - np.exp(-2 * delt / alpha))
        rv = -np.sum(np.log(sigi)) - np.sum(noise ** 2 / (2 * sigi ** 2))

        return -rv  # minus the log likelihood
