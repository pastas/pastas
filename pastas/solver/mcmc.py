"""This module contains the EmceeSolve class, which is a solver based on the MCMC approach in emcee :cite:p:`foreman-mackey_emcee_2013`."""

import importlib
from logging import getLogger
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from pastas.decorators import deprecate_args_or_kwargs
from pastas.typing import ArrayLike, CallBack

from .base import SolverBase
from .likelihood import GaussianLikelihood, GaussianLikelihoodAr1
from .objective_function import misfit

logger = getLogger(__name__)


class EmceeSolve(SolverBase):
    """Solver based on MCMC approach in emcee :cite:p:`foreman-mackey_emcee_2013`.

    Parameters
    ----------
    objfunction: pastas.solver.likelihood function, optional
        An objective function to be minimized. See the pastas.likelihood module for
        more information.
    nwalkers: int, optional
        Number of walkers to use. Default is 20.
    backend: emcee.backend, optional
        One of the Backends from Emcee used to store MCMC results. See the Emcee
        documentation for more information.
    moves: emcee.moves, optional
        The moves argument determines how the next step for a walker is chosen in
        the MCMC approach. One of the Moves classes from Emcee has to be provided.
        See Emcee documentation for more information.
    parallel: bool, optional
        Run the sampler in parallel or not.
    progress_bar: bool, optional
        Show the progress bar or not. Requires the `tqdm` package to be installed.
    **kwargs, optional
        All other keyword arguments are passed on to the SolverBase class.

    Notes
    -----
    The EmceeSolve solver uses the emcee package to perform a Markov Chain Monte Carlo
    (MCMC) approach to find the optimal parameter values. The solver can be used as
    follows::

        solver = ps.EmceeSolve(nwalkers=20, progress_bar=True)
        ml.solve(solver=solver)

    The arguments provided are mostly passed on to the `emcee.EnsembleSampler`
    and determine how that instance is created. Arguments you want to pass on to
    `run_mcmc` (and indirectly the `sample` method), can be passed on to
    `Model.solve`, like::

        ml.solve(solver=ps.EmceeSolve(), thin_by=2)

    Examples
    --------
    Example usage::

        ml.solve(solver=ps.EmceeSolve(), steps=5000)

    To obtain the MCMC chains, use::

        ml.solver.sampler.get_chain(flat=True, discard=3000)

    References
    ----------
    https://emcee.readthedocs.io/en/stable/

    See Also
    --------
    emcee.EnsembleSampler
    emcee.moves
    emcee.backend
    pastas.solver.objective_function

    """

    def __init__(
        self,
        name: str = "solver",
        objfunction: GaussianLikelihood
        | GaussianLikelihoodAr1
        | None = GaussianLikelihood(),
        nwalkers: int = 20,
        backend: Any | None = None,
        moves: Any | None = None,
        parallel: bool = False,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> None:
        self._assert_emcee_installation()

        if "objective_function" in kwargs:
            deprecate_args_or_kwargs(
                "objective_function",
                "2.0.0",
                reason="Use the argument objfunction instead",
            )
            objfunction = kwargs.pop("objective_function")

        self.objfunction = objfunction

        super().__init__(name=name, **kwargs)

        # Set sampler properties
        self.sampler: Any | None = None
        self.backend = backend
        self.moves = moves
        self.parallel = parallel
        self.progress_bar = progress_bar
        self.nwalkers = nwalkers
        self.nsteps: int | None = None
        self.priors: list[DataFrame] = []
        self.initial: np.ndarray
        self.vary: np.ndarray
        self.bounds: np.ndarray

        # Set objective function
        self.objfunction = objfunction
        self.set_init_parameters()

    def _assert_emcee_installation(self) -> None:
        try:
            global emcee
            import emcee as emcee  # Import emcee here, so it is no dependency
        except ImportError:
            msg = "emcee not installed. Please install emcee first."
            raise ImportError(msg) from None

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get the initial parameters for the solver.

        Parameters
        ----------
        name: str
            Name of the solver instance.

        Returns
        -------
        parameters: DataFrame
            Initial parameters for the solver.

        """
        parameters = self.objfunction.get_init_parameters(name if name else self.name)
        return parameters

    def solve(
        self,
        noise: bool = False,
        weights: Series | None = None,
        steps: int = 5000,
        callback: CallBack | None = None,
        **kwargs: Any,
    ) -> tuple[bool, DataFrame]:
        # Store initial parameters
        self.initial = self.ml.parameters.initial.to_numpy(dtype=float)
        self.vary = self.ml.parameters.vary.to_numpy(dtype=bool)

        # Set lower and upper bounds
        lb = self.ml.parameters[self.ml.parameters.vary].pmin.to_numpy(dtype=float)
        ub = self.ml.parameters[self.ml.parameters.vary].pmax.to_numpy(dtype=float)
        self.bounds = np.vstack([lb, ub]).T

        # Set priors
        self._set_priors()

        self.nsteps = steps

        # Set initial positions of the walkers
        pinit = self.initial[self.vary]
        ndim = pinit.size
        pinit = pinit + np.abs(pinit) * 1e-2 * np.random.randn(self.nwalkers, ndim)

        # Create sampler and run mcmc
        if self.parallel:
            logger.info("Going into the parallel universe")

            from multiprocessing import Pool

            with Pool() as pool:
                self.sampler = emcee.EnsembleSampler(
                    nwalkers=self.nwalkers,
                    ndim=ndim,
                    log_prob_fn=self.log_probability,
                    moves=self.moves,
                    backend=self.backend,
                    pool=pool,
                    args=(noise, weights, callback),
                )

                self.sampler.run_mcmc(
                    initial_state=pinit,
                    nsteps=steps,
                    progress=self.progress_bar,
                    **kwargs,
                )
        else:
            self.sampler = emcee.EnsembleSampler(
                nwalkers=self.nwalkers,
                ndim=ndim,
                log_prob_fn=self.log_probability,
                moves=self.moves,
                backend=self.backend,
                pool=None,
                args=(noise, weights, callback),
            )

            self.sampler.run_mcmc(
                initial_state=pinit, nsteps=steps, progress=self.progress_bar, **kwargs
            )

        # Get optimal values
        optimal = self.initial.copy()
        chains = self.sampler.get_chain(discard=0, flat=True, thin=1)
        optimal[self.vary] = chains[self.sampler.get_log_prob().argmax()]

        # Set the optimal values for the objective function parameters
        self.parameters.loc[:, "optimal"] = optimal[-self.objfunction.nparam :]

        success = True
        result = DataFrame(
            {
                "optimal": optimal,
                # "Q025": TODO: compute credible intervals
                # "Q975": TODO: compute credible intervals
            },
            index=self.ml.parameters.index,
            dtype=float,
        )
        return success, result

    def log_probability(
        self,
        p: ArrayLike,
        noise: bool | None = False,
        weights: Series | None = None,
        callback: CallBack | None = None,
    ) -> float:
        """Full log-probability called by Emcee.

        Parameters
        ----------
        p: numpy.Array
            Numpy array with the parameters.
        noise: bool, optional
            If True, the noise model is applied to the residuals.
        weights: pandas.Series, optional
            Series with weights for the residuals.
        callback: callable, optional
            Callback function that will be called after each iteration of the solver.

        Returns
        -------
        log_probability: float

        """
        lp = self.log_prior(p)

        # This will occur if the parameters are outside the boundaries
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + self.log_likelihood(
                p, noise=noise, weights=weights, callback=callback
            )

    def log_likelihood(
        self,
        p: ArrayLike,
        noise: bool,
        weights: Series | None = None,
        callback: CallBack | None = None,
    ) -> float:
        """Log-likelihood function.

        Parameters
        ----------
        p: numpy.Array
            Numpy array with the parameters.
        noise: bool
            If True, the noise model is applied to the residuals. This is passed on to
            the misfit function, which will apply the noise model if True.
        weights: pandas.Series, optional
            Series with weights for the residuals. This is passed on to the misfit
            function, which will apply the weights if provided.
        callback: callable, optional
            Callback function that will be called after each iteration of the solver.
            This is passed on to the misfit function, which will call the callback if
            provided.

        Returns
        -------
        lnlike: float
            The log-likelihood for the parameters.

        Notes
        -----
        This method is always called by emcee.

        """
        par = self.initial.copy()

        # Set the parameters that are varied from the model and objective function
        par[self.vary] = p

        rv = misfit(
            p=p,
            noise=noise,
            ml=self.ml,
            weights=weights,
            callback=callback,
            returnseparate=False,
        )

        lnlike = self.objfunction.compute(rv, par[-self.objfunction.nparam :])

        return lnlike

    def log_prior(self, p: ArrayLike) -> float:
        """Probability of parameter set given the priors.

        Parameters
        ----------
        p: numpy.Array
            Numpy array with the parameters

        Returns
        -------
        lp: float
            Probability of parameter set given the priors

        Notes
        -----
        Two cases exist:

        - If any of the parameters touch the boundary, -np.inf is returned. This
          basically tells the algorithm that the parameter set is very unlikely.
        - Otherwise, the probability of each parameter given its prior is computed.

        """
        # Check if parameters are within the boundaries
        if np.any(p < self.bounds[:, 0]) or np.any(p > self.bounds[:, 1]):
            lp = -np.inf
        # If not, compute the probability of each parameter given its prior
        else:
            lp = 0.0
            for param, prior in zip(p, self.priors):
                lp += prior.logpdf(param)
        return lp

    def _set_priors(self) -> None:
        """Set the priors for the parameters."""
        self.priors = []
        cols = ["initial", "pmin", "pmax", "sigma", "dist"]

        # Set the priors for the parameters that are varied from the model
        for _, p in self.ml.parameters.loc[self.ml.parameters.vary, cols].iterrows():
            prior = self._get_prior(
                dist=p["dist"],
                loc=p["initial"],
                scale=p["sigma"],
                pmin=p["pmin"],
                pmax=p["pmax"],
            )
            self.priors.append(prior)

    def _get_prior(
        self, dist: str, loc: float, scale: float, pmin: float, pmax: float
    ) -> Any:
        """Set the prior for a parameter.

        Parameters
        ----------
        dist: str
            Name of the distribution. Must be a scipy.stats distribution.
        loc: float
            Location parameter. For example, the mean for a normal distribution.
        scale: float
            Scale parameter. For example, the standard deviation for a normal distribution.

        Returns
        -------
        dist: scipy.stats distribution

        """
        # Import the distribution
        mod = importlib.import_module("scipy.stats")

        # Return the distribution
        if dist == "uniform":
            loc = pmin
            scale = pmax - pmin

        if np.isnan(loc) or np.isnan(scale):
            msg = "Location and/or scale parameter is NaN."
            logger.error(msg)
            raise ValueError(msg)

        return getattr(mod, dist)(loc=loc, scale=scale)

    def fit_report(
        self,
        full_output: bool = False,
        warnings: bool = True,
        obj_func: float = np.nan,
    ) -> str:
        """Method that reports on the fit after a model is optimized.

        Parameters
        ----------
        full_output : bool, optional
            If True, all options are shown in the fit report. This is a shortcut for
            `warnings=True`.
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
            "nwalkers": self.nwalkers,
            "nsteps": self.nsteps,
            "nobs": self.ml.observations().index.size,
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
            warnings = True

        parameters = self.ml._parameters.loc[
            :, ["optimal", "initial", "vary", "sigma", "dist"]
        ].copy()

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

        warnings_rep = ""
        if warnings:
            msg = self.ml._generate_warnings_report(log=False, solve_success=True)

            # create message
            if len(msg) > 0:
                msg = [
                    f"\n\nWarnings! ({len(msg)})\n"
                    f"{string.format('', fill='=', align='>', width=width)}"
                ] + msg
                warnings_rep += "\n".join(msg)

        report = f"{header}{basic}{params}{warnings_rep}"

        return report

    def to_dict(self) -> dict:
        """This method is not supported for this solver.

        Returns
        -------
        dict
        """
        logger.warning(
            "Note that the EmceeSolve class is not fully reproducible. "
            "The EmceeSolve class has some attributes that are not saved"
            " and cannot be reproduced. To ensure reproducibility, it "
            " is recommended to save the attributes separately."
        )
        return super().to_dict() | {"nwalkers": self.nwalkers}
