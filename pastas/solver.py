"""This module contains the different solvers that are available for Pastas.

All solvers inherit from the BaseSolver class, which contains general method for
selecting the correct time series to misfit and options to weight the residuals or
noise series.

To solve a model the following syntax can be used:

>>> ml.solve(solver=ps.LeastSquares())
"""

from logging import getLogger

# Type Hinting
from typing import Optional, Tuple, Union

import numpy as np
from pandas import DataFrame, Series
from scipy.linalg import svd
from scipy.optimize import least_squares

from pastas.typing import ArrayLike, CallBack, Function, Model

logger = getLogger(__name__)


class BaseSolver:
    _name = "BaseSolver"
    __doc__ = """All solver instances inherit from the BaseSolver class.

    Attributes
    ----------
    pcov: pandas.DataFrame
        Pandas DataFrame with the correlation between the optimized parameters.
    pcor: pandas.DataFrame
        Based on pcov, cannot be parsed.
        Pandas DataFrame with the correlation between the optimized parameters.
    nfev: int
        Number of times the model is called during optimization.
    result: object
        The object returned by the minimization method that is used. It depends
        on the solver what is actually returned.

    """

    def __init__(
        self,
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
        obj_func: Optional[Function] = None,
        **kwargs,
    ) -> None:
        self.ml = None
        self.pcov = pcov  # Covariances of the parameters
        if pcov is None:
            self.pcor = None  # Correlation between parameters
        else:
            self.pcor = self._get_correlations(pcov)
        self.nfev = nfev  # number of function evaluations
        self.obj_func = obj_func
        self.result = None  # Object returned by the optimization method

    def set_model(self, ml: Model):
        """Method to set the Pastas Model instance.

        Parameters
        ----------
        ml: pastas.Model instance

        """
        if self.ml is not None:
            raise UserWarning(
                "This solver instance is already used by another model. Please create "
                "a separate solver instance for each Pastas Model."
            )
        self.ml = ml

    def misfit(
        self,
        p: ArrayLike,
        noise: bool,
        weights: Optional[Series] = None,
        callback: Optional[CallBack] = None,
        returnseparate: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """This method is called by all solvers to obtain a series that are
        minimized in the optimization process. It handles the application of
        the weights, a noisemodel and other optimization options.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the
            model parameters.
        noise: Boolean
        weights: pandas.Series, optional
            pandas Series by which the residual or noise series are
            multiplied. Typically values between 0 and 1.
        callback: ufunc, optional
            function that is called after each iteration. the parameters are
            provided to the func. E.g. "callback(parameters)"
        returnseparate: bool, optional
            return residuals, noise, noiseweights

        Returns
        -------
        rv: array_like
            residuals array (if noise=False) or noise array (if noise=True)
        """
        # Get the residuals or the noise
        if noise:
            rv = self.ml.noise(p) * self.ml.noise_weights(p)

        else:
            rv = self.ml.residuals(p)

        # Determine if weights need to be applied
        if weights is not None:
            weights = weights.reindex(rv.index)
            weights.fillna(1.0, inplace=True)
            rv = rv.multiply(weights)

        if callback:
            callback(p)

        if returnseparate:
            return (
                self.ml.residuals(p).values,
                self.ml.noise(p).values,
                self.ml.noise_weights(p).values,
            )

        return rv.values

    def prediction_interval(
        self, n: int = 1000, alpha: float = 0.05, max_iter: int = 10, **kwargs
    ) -> DataFrame:
        """Method to calculate the prediction interval for the simulation.

        Returns
        -------
        data : Pandas.DataFrame
            DataFrame of length number of observations and two columns labeled
            0.025 and 0.975 (numerical values) containing the 2.5% and 97.5%
            prediction interval (for alpha=0.05)

        Notes
        -----
        Add residuals assuming a Normal distribution with standard deviation
        equal to the standard deviation of the residuals.
        """

        sigr = self.ml.residuals().std()

        data = self._get_realizations(
            func=self.ml.simulate, n=n, name=None, max_iter=max_iter, **kwargs
        )
        data = data + sigr * np.random.randn(data.shape[0], data.shape[1])

        q = [alpha / 2, 1 - alpha / 2]
        rv = data.quantile(q, axis=1).transpose()
        return rv

    def ci_simulation(
        self, n: int = 1000, alpha: float = 0.05, max_iter: int = 10, **kwargs
    ) -> DataFrame:
        """Method to calculate the confidence interval for the simulation.

        Returns
        -------
        data : Pandas.DataFrame
            DataFrame of length number of observations and two columns labeled
            0.025 and 0.975 (numerical values) containing the 2.5% and 97.5%
            interval (for alpha=0.05)

        Notes
        -----
        The confidence interval shows the uncertainty in the simulation due
        to parameter uncertainty. In other words, there is a 95% probability
        that the true best-fit line for the observed data lies within the
        95% confidence interval.
        """
        return self._get_confidence_interval(
            func=self.ml.simulate, n=n, alpha=alpha, max_iter=max_iter, **kwargs
        )

    def ci_block_response(
        self,
        name: str,
        n: int = 1000,
        alpha: float = 0.05,
        max_iter: int = 10,
        **kwargs,
    ) -> DataFrame:
        """Method to calculate the confidence interval for the block response.

        Returns
        -------
        data : Pandas.DataFrame
            DataFrame of length number of observations and two columns labeled
            0.025 and 0.975 (numerical values) containing the 2.5% and 97.5%
            interval (for alpha=0.05)

        Notes
        -----
        The confidence interval shows the uncertainty in the simulation due
        to parameter uncertainty. In other words, there is a 95% probability
        that the true best-fit line for the observed data lies within the
        95% confidence interval.
        """
        dt = self.ml.get_block_response(name=name).index.values
        return self._get_confidence_interval(
            func=self.ml.get_block_response,
            n=n,
            alpha=alpha,
            name=name,
            max_iter=max_iter,
            dt=dt,
            **kwargs,
        )

    def ci_step_response(
        self,
        name: str,
        n: int = 1000,
        alpha: float = 0.05,
        max_iter: int = 10,
        **kwargs,
    ) -> DataFrame:
        """Method to calculate the confidence interval for the step response.

        Returns
        -------
        data : Pandas.DataFrame
            DataFrame of length number of observations and two columns labeled
            0.025 and 0.975 (numerical values) containing the 2.5% and 97.5%
            interval (for alpha=0.05)

        Notes
        -----
        The confidence interval shows the uncertainty in the simulation due
        to parameter uncertainty. In other words, there is a 95% probability
        that the true best-fit line for the observed data lies within the
        95% confidence interval.
        """
        dt = self.ml.get_block_response(name=name).index.values
        return self._get_confidence_interval(
            func=self.ml.get_step_response,
            n=n,
            alpha=alpha,
            name=name,
            max_iter=max_iter,
            dt=dt,
            **kwargs,
        )

    def ci_contribution(
        self,
        name: str,
        n: int = 1000,
        alpha: float = 0.05,
        max_iter: int = 10,
        **kwargs,
    ) -> DataFrame:
        """Method to calculate the confidence interval for the contribution.

        Returns
        -------
        data : Pandas.DataFrame
            DataFrame of length number of observations and two columns labeled
            0.025 and 0.975 (numerical values) containing the 2.5% and 97.5%
            interval (for alpha=0.05)

        Notes
        -----
        The confidence interval shows the uncertainty in the simulation due
        to parameter uncertainty. In other words, there is a 95% probability
        that the true best-fit line for the observed data lies within the
        95% confidence interval.
        """
        return self._get_confidence_interval(
            func=self.ml.get_contribution,
            n=n,
            alpha=alpha,
            name=name,
            max_iter=max_iter,
            **kwargs,
        )

    def get_parameter_sample(
        self, name: Optional[str] = None, n: int = None, max_iter: int = 10
    ) -> ArrayLike:
        """Method to obtain a parameter sets for monte carlo analyses.

        Parameters
        ----------
        name: str, optional
            Name of the stressmodel or model component to obtain the
            parameters for.
        n: int, optional
            Number of random samples drawn from the bivariate normal
            distribution.
        max_iter : int, optional
            maximum number of iterations for truncated multivariate
            sampling, default is 10. Increase this value if number of
            accepted parameter samples is lower than n.

        Returns
        -------
        array_like
            array with N parameter samples.
        """
        p = self.ml.get_parameters(name=name)
        pcov = self._get_covariance_matrix(name=name)

        if name is None:
            parameters = self.ml.parameters
        else:
            parameters = self.ml.parameters.loc[self.ml.parameters.name == name]

        pmin = parameters.pmin.fillna(-np.inf).values
        pmax = parameters.pmax.fillna(np.inf).values

        if n is None:
            # only use parameters that are varied.
            n = int(10 ** parameters.vary.sum())

        samples = np.zeros((0, p.size))

        # Start truncated multivariate sampling
        it = 0
        while samples.shape[0] < n:
            s = np.random.multivariate_normal(p, pcov, size=(n,), check_valid="ignore")
            accept = s[
                (np.min(s - pmin, axis=1) >= 0) & (np.max(s - pmax, axis=1) <= 0)
            ]
            samples = np.concatenate((samples, accept), axis=0)

            # Make sure there's no endless while loop
            if it > max_iter:
                break
            else:
                it += 1

        if samples.shape[0] < n:
            logger.warning(
                "Parameter sample size is smaller than n: "
                f"{samples.shape[0]}/{n}. Increase 'max_iter'."
            )
        return samples[:n, :]

    def _get_realizations(
        self,
        func: Function,
        n: Optional[int] = None,
        name: Optional[str] = None,
        max_iter: int = 10,
        **kwargs,
    ) -> DataFrame:
        """Internal method to obtain n number of parameter realizations."""
        if name:
            kwargs["name"] = name

        parameter_sample = self.get_parameter_sample(n=n, name=name, max_iter=max_iter)
        data = {}

        for i, p in enumerate(parameter_sample):
            data[i] = func(p=p, **kwargs)

        return DataFrame.from_dict(data, orient="columns", dtype=float)

    def _get_confidence_interval(
        self,
        func: Function,
        n: Optional[int] = None,
        name: Optional[str] = None,
        max_iter: int = 10,
        alpha: float = 0.05,
        **kwargs,
    ) -> DataFrame:
        """Internal method to obtain a confidence interval."""
        q = [alpha / 2, 1 - alpha / 2]
        data = self._get_realizations(
            func=func, n=n, name=name, max_iter=max_iter, **kwargs
        )
        return data.quantile(q=q, axis=1).transpose()

    def _get_covariance_matrix(self, name: Optional[str] = None) -> DataFrame:
        """Internal method to obtain the covariance matrix from the model.

        Parameters
        ----------
        name: str, optional
            Name of the stressmodel or model component to obtain the
            parameters for.

        Returns
        -------
        pcov: pandas.DataFrame
            Pandas DataFrame with the covariances for the parameters.
        """
        if name:
            index = self.ml.parameters.loc[
                self.ml.parameters.loc[:, "name"] == name
            ].index
        else:
            index = self.ml.parameters.index

        pcov = self.pcov.reindex(index=index, columns=index).fillna(0)

        return pcov

    @staticmethod
    def _get_correlations(pcov: DataFrame) -> DataFrame:
        """Internal method to obtain the parameter correlations from the
        covariance matrix.

        Parameters
        ----------
        pcov: pandas.DataFrame
            n x n Pandas DataFrame with the covariances.

        Returns
        -------
        pcor: pandas.DataFrame
            n x n Pandas DataFrame with the correlations.
        """
        index = pcov.index
        pcov = pcov.to_numpy()
        v = np.sqrt(np.diag(pcov))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = pcov / np.outer(v, v)
        corr[pcov == 0] = 0
        pcor = DataFrame(data=corr, index=index, columns=index)
        return pcor

    def to_dict(self) -> dict:
        data = {
            "class": self._name,
            "pcov": self.pcov,
            "nfev": self.nfev,
            "obj_func": self.obj_func,
        }
        return data


class LeastSquares(BaseSolver):
    _name = "LeastSquares"

    def __init__(
        self,
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Solver based on Scipy's least_squares method :cite:p:`virtanen_scipy_2020`.

        Notes
        -----
        This class is the default solve method called by the pastas Model solve
        method. All kwargs provided to the Model.solve() method are forwarded to the
        solver. From there, they are forwarded to Scipy least_squares solver.

        Examples
        --------

        >>> ml.solve(solver=ps.LeastSquares())

        References
        ----------
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        """
        BaseSolver.__init__(self, pcov=pcov, nfev=nfev, **kwargs)

    def solve(
        self,
        noise: bool = True,
        weights: Optional[Series] = None,
        callback: Optional[CallBack] = None,
        **kwargs,
    ) -> Tuple[bool, ArrayLike, ArrayLike]:
        self.vary = self.ml.parameters.vary.values.astype(bool)
        self.initial = self.ml.parameters.initial.values.copy()
        parameters = self.ml.parameters.loc[self.vary]

        # Set the boundaries
        bounds = (
            np.where(parameters.pmin.isnull(), -np.inf, parameters.pmin),
            np.where(parameters.pmax.isnull(), np.inf, parameters.pmax),
        )

        self.result = least_squares(
            self.objfunction,
            bounds=bounds,
            x0=parameters.initial.values,
            args=(noise, weights, callback),
            **kwargs,
        )

        self.pcov = DataFrame(
            self._get_covariances(self.result.jac, self.result.cost),
            index=parameters.index,
            columns=parameters.index,
        )
        self.pcor = self._get_correlations(self.pcov)
        self.nfev = self.result.nfev
        self.obj_func = self.result.cost

        # Prepare return values
        success = self.result.success
        optimal = self.initial
        optimal[self.vary] = self.result.x
        stderr = np.zeros(len(optimal)) * np.nan
        stderr[self.vary] = np.sqrt(np.diag(self.pcov))

        return success, optimal, stderr

    def objfunction(
        self, p: ArrayLike, noise: bool, weights: Series, callback: CallBack
    ) -> ArrayLike:
        par = self.initial
        par[self.vary] = p
        return self.misfit(p=par, noise=noise, weights=weights, callback=callback)

    def _get_covariances(
        self, jacobian: ArrayLike, cost: float, absolute_sigma: bool = False
    ) -> ArrayLike:
        """Internal method to get the covariance matrix from the jacobian.

        Parameters
        ----------
        jacobian: array_like
        cost: float
        absolute_sigma: bool
            Default is False.

        Returns
        -------
        pcov: array_like
            array with the covariance matrix.

        Notes
        -----
        This method is copied from Scipy, please refer to:
        https://github.com/scipy/scipy/blob/v1.0.0/scipy/optimize/optimize.py
        """
        cost = 2 * cost  # res.cost is half sum of squares!

        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(jacobian, full_matrices=False)
        threshold = np.finfo(float).eps * max(jacobian.shape) * s[0]
        s = s[s > threshold]
        VT = VT[: s.size]
        pcov = np.dot(VT.T / s**2, VT)
        n_param = self.ml.parameters.index.size
        warn_cov = False
        if pcov is None:
            # indeterminate covariance
            pcov = np.zeros((n_param, n_param), dtype=float)
            pcov.fill(np.inf)
            warn_cov = True
        elif not absolute_sigma:
            if self.ml.oseries.series.index.size > n_param:
                s_sq = cost / (self.ml.oseries.series.index.size - n_param)
                pcov = pcov * s_sq
            else:
                pcov.fill(np.inf)
                warn_cov = True

        if warn_cov:
            logger.warning("Covariance of the parameters could not be estimated")

        return pcov


class LmfitSolve(BaseSolver):
    _name = "LmfitSolve"

    def __init__(
        self,
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Solving the model using the LmFit :cite:p:`newville_lmfitlmfit-py_2019`.

         This is basically a wrapper around the scipy solvers, adding some cool
         functionality for boundary conditions.

        Notes
        -----
        https://github.com/lmfit/lmfit-py/
        """
        try:
            global lmfit
            import lmfit as lmfit  # Import Lmfit here, so it is no dependency
        except ImportError:
            msg = "lmfit not installed. Please install lmfit first."
            raise ImportError(msg)
        BaseSolver.__init__(self, pcov=pcov, nfev=nfev, **kwargs)

    def solve(
        self,
        noise: bool = True,
        weights: Optional[Series] = None,
        callback: Optional[CallBack] = None,
        method: Optional[str] = "leastsq",
        **kwargs,
    ) -> Tuple[bool, ArrayLike, ArrayLike]:
        # Deal with the parameters
        parameters = lmfit.Parameters()
        p = self.ml.parameters.loc[:, ["initial", "pmin", "pmax", "vary"]]
        for k in p.index:
            pp = np.where(p.loc[k].isnull(), None, p.loc[k])
            parameters.add(k, value=pp[0], min=pp[1], max=pp[2], vary=pp[3])

        # Create the Minimizer object and minimize
        self.mini = lmfit.Minimizer(
            userfcn=self.objfunction,
            calc_covar=True,
            fcn_args=(noise, weights, callback),
            params=parameters,
            **kwargs,
        )
        self.result = self.mini.minimize(method=method)

        # Set all parameter attributes
        pcov = None
        if hasattr(self.result, "covar"):
            if self.result.covar is not None:
                pcov = self.result.covar

        names = self.result.var_names
        self.pcov = DataFrame(pcov, index=names, columns=names, dtype=float)
        self.pcor = self._get_correlations(self.pcov)

        # Set all optimization attributes
        self.nfev = self.result.nfev
        self.obj_func = self.result.chisqr

        if hasattr(self.result, "success"):
            success = self.result.success
        else:
            success = True
        optimal = np.array([p.value for p in self.result.params.values()])
        stderr = np.array([p.stderr for p in self.result.params.values()])

        idx = None
        if "is_weighted" in kwargs:
            if not kwargs["is_weighted"]:
                idx = -1

        return success, optimal[:idx], stderr[:idx]

    def objfunction(
        self, parameters: DataFrame, noise: bool, weights: Series, callback: CallBack
    ) -> ArrayLike:
        p = np.array([p.value for p in parameters.values()])
        return self.misfit(p=p, noise=noise, weights=weights, callback=callback)
