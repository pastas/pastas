"""The solver module contains the different solvers that are available for
Pastas.

All solvers inherit from the BaseSolver class, which contains general method
for selecting the correct time series to misfit and options to weight the
residuals or noise series.

Examples
--------
To solve a model the following syntax can be used:

>>> ml.solve(solver=ps.LeastSquares)

"""

from logging import getLogger

import numpy as np
from pandas import DataFrame
from scipy.linalg import svd
from scipy.optimize import least_squares

logger = getLogger(__name__)


class BaseSolver:
    _name = "BaseSolver"
    __doc__ = """All solver instances inherit from the BaseSolver class.

    Attributes
    ----------
    model: pastas.Model instance
    pcor: pandas.DataFrame
        Pandas DataFrame with the correlation between the optimized parameters.
    pcov: pandas.DataFrame
        Pandas DataFrame with the correlation between the optimized parameters.
    nfev: int
        Number of times the model is called during optimization.
    result: object
        The object returned by the minimization method that is used. It depends
        on the solver what is actually returned.
    
    Methods
    -------
    misfit
    get_correlations

    \b
    """

    def __init__(self, model):
        self.model = model
        self.pcor = None  # Correlation between parameters
        self.pcov = None  # Covariances of the parameters
        self.nfev = None  # number of function evaluations
        self.result = None  # Object returned by the optimization method

    def misfit(self, parameters, noise, weights=None, callback=None):
        """This method is called by all solvers to obtain a series that are
        minimized in the optimization proces. It handles the application of
        the weigths, a noisemodel and other optimization options.

        Parameters
        ----------
        parameters: list, numpy.ndarray
            list with the parameters
        noise: Boolean
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
            rv = self.model.noise(parameters)
        else:
            rv = self.model.residuals(parameters)

        # Determine if weights need to be applied
        if weights is not None:
            weights = weights.reindex(rv.index)
            weights.fillna(1.0, inplace=True)
            rv = rv.multiply(weights)

        if callback:
            callback(parameters)

        return rv.values

    @classmethod
    def get_correlations(self, pcov):
        """Method to obtain the parameter correlations from the covariance
        matrix.

        Parameters
        ----------
        pcov: np.ndarray
            n x n numpy array with the covariances.

        Returns
        -------
        pcor: np.ndarray
            Numpy array with the correlations

        """
        pcor = np.zeros_like(pcov)

        for i in range(pcov.shape[0]):
            for j in range(pcov.shape[1]):
                pcor[i, j] = pcov[i, j] / np.sqrt(pcov[i, i] * pcov[j, j])
        return pcor


class LeastSquares(BaseSolver):
    _name = "LeastSquares"

    def __init__(self, model):
        """Solver based on Scipy's least_squares method [1]_.

        Notes
        -----
        This class is the default solve method called by the pastas Model solve
        method. All kwargs provided to the Model.solve() method are forwarded
        to the solver. From there, they are forwarded to Scipy least_squares
        solver.

        Examples
        --------

        >>> ml.solve(solver=ps.LeastSquares)

        References
        ----------
        .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        """
        BaseSolver.__init__(self, model=model)

    def solve(self, noise=True, weights=None, callback=None, **kwargs):
        self.modelparameters = self.model.parameters
        self.vary = self.modelparameters.vary.values.astype(bool)
        self.initial = self.modelparameters.initial.values.copy()
        parameters = self.modelparameters.loc[self.vary]

        # Set the boundaries
        pmin = np.where(parameters.pmin.isnull(), -np.inf, parameters.pmin)
        pmax = np.where(parameters.pmax.isnull(), np.inf, parameters.pmax)
        bounds = (pmin, pmax)

        self.result = least_squares(self.objfunction, bounds=bounds,
                                    x0=parameters.initial.values,
                                    args=(noise, weights, callback), **kwargs)

        pcov = self.get_covariances(self.result.jac, self.result.cost)
        pcor = self.get_correlations(pcov)

        self.pcov = DataFrame(pcov, index=parameters.index,
                              columns=parameters.index)
        self.pcor = DataFrame(pcor, index=parameters.index,
                              columns=parameters.index)
        self.nfev = self.result.nfev

        # Prepare return values
        success = self.result.success
        optimal = self.initial
        optimal[self.vary] = self.result.x
        stderr = np.zeros(len(optimal))
        stderr[self.vary] = np.sqrt(np.diag(self.pcov))

        return success, optimal, stderr

    def objfunction(self, parameters, noise, weights, callback):
        p = self.initial
        p[self.vary] = parameters
        res = self.misfit(p, noise, weights, callback)
        return res

    def get_covariances(self, jacobian, cost, absolute_sigma=False):
        """Method to get the covariance matrix from the jacobian.

        Parameters
        ----------
        jacobian

        Returns
        -------
        pcov: numpy.array
            numpy array with the covariance matrix.

        Notes
        -----
        This method os copied from Scipy, please refer to:
        https://github.com/scipy/scipy/blob/v1.0.0/scipy/optimize/optimize.py

        """
        cost = 2 * cost  # res.cost is half sum of squares!

        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(jacobian, full_matrices=False)
        threshold = np.finfo(float).eps * max(jacobian.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s ** 2, VT)
        n_param = self.model.parameters.index.size
        warn_cov = False
        if pcov is None:
            # indeterminate covariance
            pcov = np.zeros((n_param, n_param), dtype=float)
            pcov.fill(np.inf)
            warn_cov = True
        elif not absolute_sigma:
            if self.model.oseries.series.index.size > n_param:
                s_sq = cost / (self.model.oseries.series.index.size - n_param)
                pcov = pcov * s_sq
            else:
                pcov.fill(np.inf)
                warn_cov = True

        if warn_cov:
            logger.warning(
                'Covariance of the parameters could not be estimated')

        return pcov


class LmfitSolve(BaseSolver):
    _name = "LmfitSolve"

    def __init__(self, model):
        """Solving the model using the LmFit solver [LM]_.

         This is basically a wrapper around the scipy solvers, adding some
         cool functionality for boundary conditions.

        References
        ----------
        .. [LM] https://github.com/lmfit/lmfit-py/
        """
        try:
            global lmfit
            import lmfit as lmfit  # Import Lmfit here, so it is no dependency
        except ImportError:
            msg = "lmfit not installed. Please install lmfit first."
            raise ImportError(msg)
        BaseSolver.__init__(self, model=model)

    def solve(self, noise=True, weights=None, callback=None, **kwargs):

        # Deal with the parameters
        parameters = lmfit.Parameters()
        p = self.model.parameters[['initial', 'pmin', 'pmax', 'vary']]
        for k in p.index:
            pp = np.where(p.loc[k].isnull(), None, p.loc[k])
            parameters.add(k, value=pp[0], min=pp[1], max=pp[2], vary=pp[3])

        # set ftol and epsfcn if no options for lmfit are provided. Only
        # work with Lmfit's least squares solver method.
        if not kwargs:
            kwargs = {"ftol": 1e-3, "epsfcn": 1e-4}

        self.fit = lmfit.minimize(fcn=self.objfunction, params=parameters,
                                  args=(noise, weights, callback), **kwargs)

        # Set all parameter attributes
        if hasattr(self.fit, "covar"):
            if self.fit.covar is not None:
                pcov = self.fit.covar
                pcor = self.get_correlations(pcov)
            else:
                pcov = None
                pcor = None

        names = self.fit.var_names
        self.pcov = DataFrame(pcov, index=names, columns=names)
        self.pcor = DataFrame(pcor, index=names, columns=names)

        # Set all optimization attributes
        self.nfev = self.fit.nfev

        success = self.fit.success
        optimal = np.array([p.value for p in self.fit.params.values()])
        stderr = np.array([p.stderr for p in self.fit.params.values()])

        return success, optimal, stderr

    def objfunction(self, parameters, noise, weights, callback):
        param = np.array([p.value for p in parameters.values()])
        res = self.misfit(param, noise, weights, callback)
        return res
