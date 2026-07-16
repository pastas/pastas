"""Module containing the least squares based solvers for Pastas."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from itertools import combinations
from logging import getLogger
from typing import Literal

import numpy as np
from pandas import DataFrame, Series
from scipy.linalg import LinAlgError, get_lapack_funcs, svd
from scipy.optimize import Bounds, OptimizeResult, least_squares

from pastas.decorators import PastasDeprecationWarning, temporarily_disable_cache
from pastas.plotting.plotutil import _table_formatter_stderr
from pastas.typing import ArrayLike

from .base import SolverBase
from .objective_function import misfit

logger = getLogger(__name__)


class LeastSquaresBase(SolverBase):
    """Base class for least squares solvers."""

    def __init__(
        self,
        name: str = "solver",
        pcov: DataFrame | None = None,
        **kwargs,
    ) -> None:
        """Initialize base class for least squares solvers.

        Parameters
        ----------
        name: str, optional
            Name of the solver instance. Default is "solver".
        pcov: DataFrame, optional
            DataFrame with the covariance matrix of the parameters. Default is None.

        """
        if "nfev" in kwargs:
            logger.debug(
                "The 'nfev' argument is not used in the LeastSquaresBase class and will be ignored."
            )
            kwargs.pop("nfev")
        if "obj_func" in kwargs:
            logger.debug(
                "The 'obj_func' argument is not used in the LeastSquaresBase class and will be ignored."
            )
            kwargs.pop("obj_func")
        super().__init__(name=name, **kwargs)
        self.pcov: DataFrame | None = pcov
        self.result: OptimizeResult | "lmfit.minimize.MinimizerResult" | None = None

    @property
    def pcor(self) -> DataFrame | None:
        """Property to obtain the parameter correlations from the covariance matrix.

        Returns
        -------
        pcor: pandas.DataFrame or None
            Pandas DataFrame with the correlations for the parameters. If `pcov` is None, returns None.

        """
        if self.pcov is None:
            return None
        else:
            return self._get_correlations(self.pcov)

    def _get_realizations(
        self,
        func: Callable,
        n: int | None = None,
        name: str | None = None,
        max_iter: int = 10,
        **kwargs,
    ) -> DataFrame:
        """Obtain n number of parameter realizations.

        Parameters
        ----------
        func: Callable
            Function for which to obtain the realizations. For example, `ml.simulate` or `ml.get_step_response`.
        n: int, optional
            Number of random samples drawn from the bivariate normal distribution to compute the confidence interval. Default is 1000.
        name: str, optional
            Name of the stressmodel or model component to obtain the
            parameters for.
        max_iter : int, optional
            maximum number of iterations for truncated multivariate
            sampling, default is 10. Increase this value if number of
            accepted parameter samples is lower than n.
        **kwargs
            Additional keyword arguments are passed to the function specified in `func`.

        """
        if name:
            kwargs["name"] = name

        parameter_sample = self.get_parameter_sample(n=n, name=name, max_iter=max_iter)
        data = {}

        # Disable caching during parameter sampling as each sample is unique
        with temporarily_disable_cache():
            for i, p in enumerate(parameter_sample):
                data[i] = func(p=p, **kwargs)

        return DataFrame.from_dict(data, orient="columns", dtype=float)

    def _get_confidence_interval(
        self,
        func: Callable,
        n: int | None = None,
        name: str | None = None,
        max_iter: int = 10,
        alpha: float = 0.05,
        **kwargs,
    ) -> DataFrame:
        """Obtain a confidence interval."""
        q = [alpha / 2, 1 - alpha / 2]
        data = self._get_realizations(
            func=func, n=n, name=name, max_iter=max_iter, **kwargs
        )
        return data.quantile(q=q, axis=1).transpose()

    def _get_covariance_matrix(self, name: str | None = None) -> DataFrame:
        """Obtain the covariance matrix from the model.

        Parameters
        ----------
        name: str, optional
            Name of the stressmodel or model component to obtain the parameters for.

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
        """Obtain the parameter correlations from the covariance matrix.

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
        pcov_values = pcov.to_numpy(dtype=float, copy=True)
        v = np.sqrt(np.diag(pcov_values))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = pcov_values / np.outer(v, v)
        corr[pcov_values == 0] = 0
        pcor = DataFrame(data=corr, index=index, columns=index)
        return pcor

    def get_parameter_sample(
        self, name: str | None = None, n: int | None = None, max_iter: int = 10
    ) -> ArrayLike:
        """Obtain a parameter sets for monte carlo analyses.

        Parameters
        ----------
        name: str, optional
            Name of the stressmodel or model component to obtain the parameters for.
        n: int, optional
            Number of random samples drawn from the bivariate normal distribution. If
            None, the number of samples is determined by the number of parameters that
            are varied, using 10^k where k is the number of parameters that are varied.
        max_iter : int, optional
            maximum number of iterations for truncated multivariate sampling, default
            is 10. Increase this value if number of accepted parameter samples is lower
            than n.

        Returns
        -------
        array_like
            array with N parameter samples.

        Notes
        -----
        The parameter samples are drawn from a multivariate normal distribution, and
        thus assume that the a normal distribution applies for the parameter
        uncertainty.

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
        elif isinstance(n, float):
            n = int(n)

        samples = np.zeros((0, p.size))

        # Start truncated multivariate sampling
        it = 0
        rng = np.random.default_rng()
        while samples.shape[0] < n:
            s = rng.multivariate_normal(
                mean=p, cov=pcov, size=(n,), check_valid="ignore"
            )
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
            suggestion = "You could try increasing 'max_iter'."
            if samples.shape[0] == 0:
                raise RuntimeError(
                    "No parameter samples were found within %s runs. " % max_iter
                    + suggestion
                )
            else:
                logger.warning(
                    "Parameter sample size is smaller than n: %s/%s. " % (max_iter, n)
                    + suggestion
                )
        return samples[:n, :]

    def prediction_interval(
        self, n: int = 1000, alpha: float = 0.05, max_iter: int = 10, **kwargs
    ) -> DataFrame:
        """Calculate the prediction interval for the simulation.

        Parameters
        ----------
        n: int, optional
            Number of random samples drawn from the bivariate normal distribution to
            compute the prediction interval. Default is 1000.
        alpha: float, optional
            Significance level for the prediction interval. Default is 0.05, which
            corresponds to a 95% prediction interval.
        max_iter: int, optional
            maximum number of iterations for truncated multivariate sampling, default
            is 10. Increase this value if number of accepted parameter samples is
            lower than n.
        **kwargs
            Additional keyword arguments are passed to the `ml.simulate()` method.
            For example, `tmin` and `tmax` can be passed as keyword arguments to
            compute the prediction interval for a specific period.

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
        rng = np.random.default_rng()
        datan = data + rng.normal(loc=0, scale=sigr, size=data.shape)
        q = [alpha / 2, 1 - alpha / 2]
        rv = datan.quantile(q, axis=1).transpose()
        return rv

    def ci_simulation(
        self, n: int = 1000, alpha: float = 0.05, max_iter: int = 10, **kwargs
    ) -> DataFrame:
        """Calculate the confidence interval for the simulation.

        Parameters
        ----------
        n: int, optional
            Number of random samples drawn from the bivariate normal distribution to
            compute the confidence interval. Default is 1000.
        alpha: float, optional
            Significance level for the confidence interval. Default is 0.05, which
            corresponds to a 95% confidence interval.
        max_iter: int, optional
            Maximum number of iterations for truncated multivariate sampling, default
            is 10. Increase this value if number of accepted parameter samples is
            lower than n.
        **kwargs
            Additional keyword arguments are passed to the `ml.simulate()` method.
            For example, `tmin` and `tmax` can be passed as keyword arguments to compute
            the confidence interval for a specific period.

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
        """Calculate the confidence interval for the block response.

        Parameters
        ----------
        name: str
            Name of the block response for which to calculate the confidence interval.
        n: int, optional
            Number of random samples drawn from the bivariate normal distribution to
            compute the confidence interval. Default is 1000.
        alpha: float, optional
            Significance level for the confidence interval. Default is 0.05, which
            corresponds to a 95% confidence interval.
        max_iter: int, optional
            Maximum number of iterations for truncated multivariate sampling, default
            is 10. Increase this value if number of accepted parameter samples is
            lower than n.
        **kwargs
            Additional keyword arguments are passed to the `ml.get_block_response()` method.

        Returns
        -------
        data : Pandas.DataFrame
            DataFrame of length number of observations and two columns labeled
            0.025 and 0.975 (numerical values) containing the 2.5% and 97.5%
            interval (for alpha=0.05)
        **kwargs
            Additional keyword arguments are passed to the `ml.get_block_response()`
            method.

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
        """Calculate the confidence interval for the step response.

        Parameters
        ----------
        name: str
            Name of the step response for which to calculate the confidence interval.
        n: int, optional
            Number of random samples drawn from the bivariate normal distribution to
            compute the confidence interval. Default is 1000.
        alpha: float, optional
            Significance level for the confidence interval. Default is 0.05, which
            corresponds to a 95% confidence interval.
        max_iter: int, optional
            Maximum number of iterations for truncated multivariate sampling, default
            is 10. Increase this value if number of accepted parameter samples is lower
            than n.
        **kwargs
            Additional keyword arguments are passed to the `ml.get_step_response()`
            method.

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
        """Calculate the confidence interval for the contribution.

        Parameters
        ----------
        name: str
            Name of the contribution for which to calculate the confidence interval.
        n: int, optional
            Number of random samples drawn from the bivariate normal distribution to
            compute the confidence interval. Default is 1000.
        alpha: float, optional
            Significance level for the confidence interval. Default is 0.05, which
            corresponds to a 95% confidence interval.
        max_iter: int, optional
            Maximum number of iterations for truncated multivariate sampling, default
            is 10. Increase this value if number of accepted parameter samples is lower
            than n.
        **kwargs
            Additional keyword arguments are passed to the `ml.get_contribution()`
            method.

        Returns
        -------
        data : Pandas.DataFrame
            DataFrame of length number of observations and two columns labeled
            0.025 and 0.975 (numerical values) containing the 2.5% and 97.5%
            interval (for alpha=0.05).

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

    @abstractmethod
    def solve(self) -> tuple[bool, DataFrame]:
        """Solve the optimization problem.

        Abstract method that has to be implemented by all least squares solvers.

        Returns
        -------
        tuple [bool, DataFrame]
        success: bool
            Boolean indicating whether the optimization was successful.
        result: pandas.DataFrame
            DataFrame with the optimal parameter values and their standard error.
            The index of the DataFrame corresponds to the parameter names, and it
            contains at least the following columns: "optimal", "stderr"

        """
        pass

    def fit_report(
        self,
        full_output: bool = False,
        corr: bool = False,
        stderr: bool = False,
        warnings: bool = True,
        obj_func: float = np.nan,
    ) -> str:
        """Report on the fit after a model is optimized.

        Parameters
        ----------
        full_output : bool, optional
            If True, all options are shown in the fit report. This is a shortcut for
            `corr=True`, `stderr=True`, and `warnings=True`.
        corr : bool, optional
            If True the parameter correlations are shown.
        stderr : bool, optional
            If True the standard error of the parameter values are shown. Please be
            aware of the conditions for reliable uncertainty estimates, more information
            here:
            https://pastas.readthedocs.io/stable/examples/diagnostic_checking.html
        warnings : bool, optional
            print warnings in case of optimization failure, parameters hitting
            bounds, or length of responses exceeding calibration period.
        obj_func : float, optional
            Value of the found minimal loss function value from the
            optimization algorithm. Generally obtained from the result attribute
            which is not present when loading the solver, thus by default nan.

        Returns
        -------
        report: str
            String with the report.

        Examples
        --------
        This method is called by the solve method if report=True, but can also be
        called on its own::

        >>> print(ml.fit_report)

        Notes
        -----
        The reported values for the fit use the residuals time series where possible.
        If interpolation is used this means that the result may slightly differ
        compared to using ml.simulate() and ml.observations().
        """
        model = {
            "nfev": self.result.nfev if self.result is not None else 0,
            "nobs": self.ml.observations().index.size,
            "noise": str(True if self.ml.noisemodel else False),
            "tmin": str(self.ml.settings["tmin"]),
            "tmax": str(self.ml.settings["tmax"]),
            "freq": self.ml.settings["freq"],
            "freq_obs": str(self.ml.settings["freq_obs"]),
            "warmup": str(self.ml.settings["warmup"]),
            "solver": self._name,
        }
        fit = {
            "EVP": f"{self.ml.stats.evp():.2f}",
            "R2": f"{self.ml.stats.rsq():.2f}",
            "RMSE": f"{self.ml.stats.rmse():.2f}",
            "AICc": f"{self.ml.stats.aicc():.2f}",
            "BIC": f"{self.ml.stats.bic():.2f}",
            "Obj": f"{obj_func:.2f}",
            "___": "",
            "Interp.": "Yes" if self.ml._interpolate_simulation else "No",
        }

        if full_output:
            corr = True
            stderr = True
            warnings = True

        parameters = self.ml._parameters.loc[:, ["optimal", "initial", "vary"]].copy()

        if stderr:
            stderr = (
                self.ml._parameters.loc[:, "stderr"]
                / self.ml._parameters.loc[:, "optimal"]
            )
            parameters.loc[:, "stderr"] = stderr.abs().apply(
                _table_formatter_stderr, na_rep="nan"
            )

        # determine width of the fit_report
        len_fit = max([len(v) for v in fit.values()]) + max(
            [len(v) for v in fit.keys()]
        )
        len_model = max([len(v) for v in model.values() if isinstance(v, str)]) + max(
            [len(v) for v in model.keys()]
        )
        len_param = len(parameters.to_string().split("\n")[1])
        width = max((len_fit + len_model + 8), len_param)
        string = "{:{fill}{align}{width}}"
        string = "{:{fill}{align}{width}}"

        # Create the first header with model information and stats
        wspace = max(width - (11 + 14 + len(self.name)), 1)
        mspace = width - wspace - (11 + 14)
        header = (
            f"Fit report {self.name:<{mspace}.{mspace}}"
            f"{string.format('', fill=' ', align='>', width=wspace)}"
            f"Fit Statistics\n"
            f"{string.format('', fill='=', align='>', width=width)}\n"
        )

        basic = ""
        len_val4 = max([len(v) for v in fit.values()])
        wspace = width - (9 + 23 + 9 + len_val4)
        for (val1, val2), (val3, val4) in zip(model.items(), fit.items()):
            basic += f"{val1:<9}{val2:<23}{val3:<9}{val4:>{wspace + len_val4}}\n"

        # Create the parameters block
        params = (
            f"\nParameters ({parameters.vary.sum()} optimized)\n"
            f"{string.format('', fill='=', align='>', width=width)}\n"
            f"{parameters.to_string()}"
        )

        if corr:
            cor = DataFrame(columns=["value"])
            pcor = self.pcor
            for idx, col in combinations(pcor, 2):
                if np.abs(pcor.loc[idx, col]) > 0.5:
                    cor.loc[f"{idx} {col}"] = pcor.loc[idx, col]

            corr_rep = (
                f"\n\nParameter correlations |rho| > 0.5\n"
                f"{string.format('', fill='=', align='>', width=width)}"
                f"\n{cor.to_string(float_format='%.2f', header=False)}"
            )
        else:
            corr_rep = ""

        warnings_rep = ""
        if warnings:
            solve_success = (
                self.result.success
                if self.result is not None and hasattr(self.result, "success")
                else None
            )
            msg = self.ml._generate_warnings_report(
                log=False, solve_success=solve_success
            )

            # create message
            if len(msg) > 0:
                msg = [
                    f"\n\nWarnings! ({len(msg)})\n"
                    f"{string.format('', fill='=', align='>', width=width)}"
                ] + msg
                warnings_rep += "\n".join(msg)

        report = f"{header}{basic}{params}{warnings_rep}{corr_rep}"

        return report

    def to_dict(self) -> dict:
        """Convert solver to a dictionary.

        Returns
        -------
        dict
            Dictionary containing the solver's state including the covariance matrix.
        """
        return super().to_dict() | {"pcov": self.pcov}


class LeastSquares(LeastSquaresBase):
    """Solver based on Scipy's least_squares method :cite:p:`virtanen_scipy_2020`.

    Notes
    -----
    This class is the default solve method called by the pastas Model solve
    method. All kwargs provided to the Model.solve() method are forwarded to the
    solver. From there, they are forwarded to Scipy least_squares solver.

    Examples
    --------
    >>> ml.solve(solver=ps.solver.LeastSquares())

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    """

    def __init__(
        self,
        name: str = "solver",
        jac: Literal["2-point", "3-point", "cs"] = "3-point",
        method: Literal["trf", "dogbox", "lm"] = "trf",
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        x_scale: float | Literal["jac"] | None = "jac",
        loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "linear",
        f_scale: float = 1.0,
        max_nfev: int | None = None,
        diff_step: float | ArrayLike | None = None,
        tr_solver: Literal["exact", "lsmr"] | None = None,
        pcov: DataFrame | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, pcov=pcov, **kwargs)
        self.result: OptimizeResult | None = None
        self.jac = jac
        self.method = method
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.x_scale = x_scale
        self.loss = loss
        self.f_scale = f_scale
        self.max_nfev = max_nfev
        self.diff_step = diff_step
        self.tr_solver = tr_solver

    def objfunction(
        self,
        p: ArrayLike,
        noise: bool,
        weights: Series | None,
        initial: ArrayLike,
        vary: ArrayLike,
        callback: Callable | None = None,
    ) -> ArrayLike:
        """Objective function that is minimized by the least_squares solver.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the model
            parameters.
        noise: Boolean
            If True, minimizes the sum of squared noise computed by the NoiseModel.
        weights: pandas.Series | None
            pandas Series by which the residual or noise series are
            multiplied. Typically values between 0 and 1.
        initial: array_like
            array_like object with the initial parameter values.
        vary: array_like
            array_like object with booleans indicating which parameters (p) are varied.
        callback: ufunc
            function that is called after each iteration. the parameters are
            provided to the func.
        """
        par = initial
        par[vary] = p
        return misfit(
            ml=self.ml, p=par, noise=noise, weights=weights, callback=callback
        )

    def solve(
        self,
        weights: Series | None = None,
        **kwargs,
    ) -> tuple[bool, DataFrame]:
        """Solve method calling scipy.optimize.least_squares."""
        if self.ml is None:
            raise RuntimeError("Solver is not attached to a Pastas model.")

        # Overwrite kwargs of init if parsed to solve
        init_kwargs = [k for k in kwargs if hasattr(self, k)]
        for k in init_kwargs:
            logger.info(f"Setting {k} to {kwargs[k]} for LeastSquares solver.")
            setattr(self, k, kwargs.pop(k))

        noise = self.ml.noisemodel is not None
        vary = self.ml.parameters.vary.to_numpy(dtype=bool, copy=True)
        initial = self.ml.parameters.initial.to_numpy(dtype=float, copy=True)
        parameters = self.ml.parameters.loc[vary]
        pmin = (
            parameters.loc[:, "pmin"].fillna(-np.inf).to_numpy(dtype=float, copy=True)
        )
        pmax = parameters.loc[:, "pmax"].fillna(np.inf).to_numpy(dtype=float, copy=True)

        # Set the boundaries
        if self.method == "lm":
            logger.info(
                "Method 'lm' does not support boundaries. Ignoring Pastas'"
                "`pmin` and `pmax` parameter bounds and setting them to `nan`."
            )
            bounds = Bounds(
                lb=np.full(len(parameters), -np.inf),
                ub=np.full(len(parameters), np.inf),
                keep_feasible=True,
            )
            # set to nan because that's what is used by the solver
            self.ml._parameters.loc[vary, "pmin"] = np.nan
            self.ml._parameters.loc[vary, "pmax"] = np.nan
        else:
            bounds = Bounds(
                lb=pmin,
                ub=pmax,
                keep_feasible=True,
            )

        objfunction = partial(
            self.objfunction,
            noise=noise,
            weights=weights,
            initial=initial,
            vary=vary,
            callback=kwargs.pop("callback", None),
        )

        self.result = least_squares(
            fun=objfunction,
            x0=initial[vary],
            jac=self.jac,
            bounds=bounds,
            method=self.method,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            x_scale=self.x_scale,
            loss=self.loss,
            f_scale=self.f_scale,
            max_nfev=self.max_nfev,
            diff_step=self.diff_step,
            tr_solver=self.tr_solver,
            **kwargs,
        )

        self.pcov = DataFrame(
            self.get_covariances(
                self.result.jac,
                self.result.cost,
                method=self.method,
                absolute_sigma=False,
            ),
            index=parameters.index,
            columns=parameters.index,
        )

        # Prepare return values
        success = self.result.success
        optimal = initial
        optimal[vary] = np.array(self.result.x, dtype=float)
        stderr = np.zeros(len(optimal)) * np.nan
        stderr[vary] = self.get_stderr(self.pcov).to_numpy(dtype=float, copy=True)

        result = DataFrame(
            {
                "optimal": optimal,
                "stderr": stderr,
            },
            index=self.ml.parameters.index,
            dtype=float,
        )

        return success, result

    @staticmethod
    def get_stderr(pcov: DataFrame) -> Series:
        """Calculate the standard error of the parameters from the covariance matrix.

        Parameters
        ----------
        pcov : pandas.DataFrame
            The covariance matrix of the parameters.

        Returns
        -------
        pandas.Series
            Series with the standard errors for each parameter.
        """
        if pcov is None:
            raise RuntimeError("Covariance matrix `pcov` is not available.")
        return Series(np.sqrt(np.diag(pcov)), index=pcov.index)

    @staticmethod
    def get_covariances(
        jacobian: ArrayLike,
        cost: float,
        method: Literal["trf", "dogbox", "lm"] = "trf",
        absolute_sigma: bool = False,
    ) -> ArrayLike:
        r"""Calculate the covariance matrix from the jacobian.

        Parameters
        ----------
        jacobian : ArrayLike
            The jacobian matrix with dimensions nobs, npar.
        cost : float
            The cost value of the scipy.optimize.OptimizeResult which is half the sum
            of squares. That's why the cost is multiplied by a factor of two internally
            to get the sum of squares.
        method : Literal["trf", "dogbox", "lm"], optional
            Algorithm with which the minimization is performed. Default is "trf".
        absolute_sigma : bool, optional
            If True, `sigma` is used in an absolute sense and the estimated parameter
            covariance `pcov` reflects these absolute values. If False (default), only
            the relative magnitudes of the `sigma` values matter. The returned
            parameter covariance matrix `pcov` is based on scaling `sigma` by a
            constant factor. This constant is set by demanding that the reduced `chisq`
            for the optimal parameters `popt` when using the *scaled* `sigma` equals
            unity. In other words, `sigma` is scaled to match the sample variance of
            the residuals after the fit. Default is False. Mathematically, ``pcov
             (absolute_sigma=False) =pcov(absolute_sigma=True) * chisq(popt)/(M-N)``

        Returns
        -------
        pcov: array_like
            numpy array with the covariance matrix.

        Notes
        -----
        This method is copied from Scipy. Please refer to the SciPy optimization module::
        https://docs.scipy.org/doc/scipy/reference/optimize.html

        This method uses SVD (for trf/dogbox) or QR decomposition (for lm) to
        invert the Hessian approximation (JᵀJ), which is more numerically stable
        than direct inversion.

        This method is equivalent to:
        pcov = (Jᵀ W J)⁻¹ * (rᵀ W r) / (nobs - npar)
        where:
        - J is the jacobian matrix (nobs, npar)
        - r is the vector of residuals.
        - W is the diagonal matrix of weights.
        """
        nobs, npar = jacobian.shape
        cost = 2 * cost  # res.cost is half sum of squares!
        s_sq = cost / (nobs - npar)  # variance of the residuals

        if method == "lm":
            # https://github.com/scipy/scipy/blob/939e3891a3aea61bf84a50d3da520ca7adf93ecc/scipy/optimize/_minpack_py.py#L481-L501
            # fjac A permutation of the R matrix of a QR factorization of the
            # final approximate Jacobian matrix.
            _, fjac = np.linalg.qr(jacobian)
            # leastsq expects the fjacobian to be in fortran order (npar, nobs)
            # that why it is transposed in the original code

            ipvt = np.arange(1, npar + 1, dtype=int)
            n = len(ipvt)
            r = np.triu(fjac[:n, :])

            # old method deprecated in scipy 1.10.0 since
            # the explicit dot product was not necessary and sometimes
            # the result was not symmetric positive definite.
            # See https://github.com/scipy/scipy/issues/4555.
            # old method
            # perm = np.take(np.eye(n), ipvt - 1, 0)
            # R = np.dot(r, perm)
            # cov_x = np.linalg.inv(np.dot(np.transpose(R), R))

            # new method:
            perm = ipvt - 1
            inv_triu = get_lapack_funcs("trtri", (r,))
            try:
                # inverse of permuted matrix is a permutation of matrix inverse
                invR, trtri_info = inv_triu(r)  # default: upper, non-unit diag
                if trtri_info != 0:  # explicit comparison for readability
                    logger.warning(
                        f"LinAlgError in trtri. LAPACK trtri returned info: {trtri_info}"
                    )
                    raise LinAlgError
                invR[perm] = invR.copy()
                pcov = invR @ invR.T  # cov_x in the original code
            except (LinAlgError, ValueError):
                pcov = None
        else:
            # https://github.com/scipy/scipy/blob/939e3891a3aea61bf84a50d3da520ca7adf93ecc/scipy/optimize/_minpack_py.py#L1045-L1068
            # Do Moore-Penrose inverse discarding zero singular values.
            _, s, VT = svd(jacobian, full_matrices=False)
            threshold = np.finfo(float).eps * max(jacobian.shape) * s[0]
            s = s[s > threshold]
            VT = VT[: s.size]
            pcov = np.dot(
                VT.T / s**2, VT
            )  # $V^T S^{-2} V$ is perfectly equivalent to $(J^T J)^{-1}$.

        if pcov is None or np.isnan(pcov).any():
            # indeterminate covariance
            pcov = np.full(shape=(npar, npar), fill_value=np.inf, dtype=float)
            logger.warning(
                "Covariance of the parameters could not be estimated. "
                "The covariance of the parameters is set to infinity."
            )
        elif not absolute_sigma:
            if nobs > npar:
                pcov = pcov * s_sq
            else:
                pcov = np.full(shape=(npar, npar), fill_value=np.inf, dtype=float)
                logger.warning(
                    "Covariance of the parameters could not be estimated. "
                    "The covariance of the parameters is set to infinity."
                )

        return pcov

    def fit_report(
        self,
        corr: bool = False,
        stderr: bool = False,
        warnings: bool = True,
        obj_func: float = np.nan,
        full_output: bool = False,
    ) -> str:
        """Report on the fit after a model is optimized.

        Parameters
        ----------
        corr : bool, optional
            If True the parameter correlations are shown.
        stderr : bool, optional
            If True the standard error of the parameter values are shown. Please be
            aware of the conditions for reliable uncertainty estimates, more information
            here:
            https://pastas.readthedocs.io/stable/examples/diagnostic_checking.html
        warnings : bool, optional
            print warnings in case of optimization failure, parameters hitting
            bounds, or length of responses exceeding calibration period.
        full_output : bool, optional
            If True, all options are shown in the fit report. This is a shortcut for
            `corr=True`, `stderr=True`, and `warnings=True`.

        Returns
        -------
        report: str
            String with the report.

        Examples
        --------
        This method is called by the solve method if report=True, but can also be
        called on its own::

        >>> print(ml.fit_report)

        Notes
        -----
        The reported values for the fit use the residuals time series where possible.
        If interpolation is used this means that the result may slightly differ
        compared to using ml.simulate() and ml.observations().
        """
        return super().fit_report(
            corr=corr,
            stderr=stderr,
            warnings=warnings,
            obj_func=obj_func,
            full_output=full_output,
        )

    def to_dict(self) -> dict:
        """Convert the solver settings to a dictionary."""
        settings = super().to_dict() | {
            "jac": self.jac,
            "method": self.method,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "gtol": self.gtol,
            "x_scale": self.x_scale,
            "loss": self.loss,
            "f_scale": self.f_scale,
            "max_nfev": self.max_nfev,
            "diff_step": self.diff_step,
            "tr_solver": self.tr_solver,
        }
        return settings


@PastasDeprecationWarning(
    version="2.3.0", reason="The LmfitSolve class is renamed to Lmfit."
)
def LmfitSolve(*args, **kwargs):
    """Alias for Lmfit."""
    return Lmfit(*args, **kwargs)


class Lmfit(LeastSquaresBase):
    """Solving the model using the LmFit :cite:p:`newville_lmfitlmfit-py_2019`.

    This is basically a wrapper around the SciPy Levenberg Marquardt solver ("leastsq").
    Lmfit adds some functionality for gracefully handling boundary conditions.

    Notes
    -----
    https://github.com/lmfit/lmfit-py/
    """

    def __init__(
        self,
        name: str = "solver",
        method: Literal["leastsq"] = "leastsq",
        pcov: DataFrame | None = None,
        **kwargs,
    ) -> None:
        self._assert_lmfit_installation()
        super().__init__(name=name, pcov=pcov, **kwargs)
        self.method = method
        self.result: "lmfit.minimize.MinimizerResult" | None = None

    def _assert_lmfit_installation(self) -> None:
        try:
            global lmfit
            import lmfit as lmfit  # Import Lmfit here, so it is no dependency
        except ImportError:
            msg = "lmfit not installed. Please install lmfit first."
            raise ImportError(msg) from None

    def solve(
        self,
        noise: bool = True,
        weights: Series | None = None,
        **kwargs,
    ) -> tuple[bool, DataFrame]:
        """Call lmfit.Minimizer.minimize to solve the model."""
        # Overwrite kwargs of init if parsed to solve
        init_kwargs = [k for k in kwargs if hasattr(self, k)]
        for k in init_kwargs:
            logger.info(f"Setting {k} to {kwargs[k]} for LmfitSolve solver.")
            setattr(self, k, kwargs.pop(k))

        # Deal with the parameters
        parameters = lmfit.Parameters()
        for pname, params in self.ml.parameters.loc[
            :, ["initial", "pmin", "pmax", "vary"]
        ].iterrows():
            pp = np.where(params.isnull(), None, params)
            parameters.add(pname, value=pp[0], min=pp[1], max=pp[2], vary=pp[3])

        # Create the Minimizer object and minimize
        objfunction = partial(
            self.objfunction,
            noise=noise,
            weights=weights,
        )
        mini = lmfit.Minimizer(
            userfcn=objfunction,
            calc_covar=True,
            params=parameters,
            **kwargs,
        )
        self.result = mini.minimize(method=self.method)
        names = self.result.var_names

        # Set all parameter attributes
        covar = (
            self.result.covar
            if hasattr(self.result, "covar") and self.result.covar is not None
            else None
        )
        self.pcov = (
            DataFrame(
                covar,
                index=names,
                columns=names,
                dtype=float,
            )
            if covar is not None
            else None
        )

        # Set all optimization attributes
        success = self.result.success if hasattr(self.result, "success") else True
        optimal = np.array([p.value for p in self.result.params.values()])
        stderr = np.array([p.stderr for p in self.result.params.values()])

        idx = -1 if "is_weighted" in kwargs and not kwargs["is_weighted"] else None

        result = DataFrame(
            {
                "optimal": optimal[:idx],
                "stderr": stderr[:idx],
            },
            index=self.ml.parameters.index,
            dtype=float,
        )

        return success, result

    def objfunction(
        self, parameters: DataFrame, noise: bool, weights: Series
    ) -> ArrayLike:
        """Objective function that is minimized by the Lmfit solver."""
        p = np.array([p.value for p in parameters.values()])
        return misfit(
            ml=self.ml,
            p=p,
            noise=noise,
            weights=weights,
            callback=None,
            returnseparate=False,
        )

    def fit_report(
        self,
        corr: bool = False,
        stderr: bool = False,
        warnings: bool = True,
        obj_func: float = np.nan,
        full_output: bool = False,
    ) -> str:
        """Report on the fit after a model is optimized."""
        # nobs = self.result.ndata
        # aic = self.result.aic
        # bic = self.result.bic
        obj_func = self.result.chisqr if self.result is not None else obj_func
        return super().fit_report(
            corr=corr,
            stderr=stderr,
            warnings=warnings,
            full_output=full_output,
            obj_func=obj_func,
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Lmfit object."""
        return super().to_dict() | {"method": self.method}
