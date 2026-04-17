import importlib
from logging import getLogger

import numpy as np
from pandas import DataFrame, Series

from pastas.decorators import deprecate_args_or_kwargs
from pastas.typing import ArrayLike, CallBack

from .base import BaseSolver
from .objective_functions import GaussianLikelihood, GaussianLikelihoodAr1

logger = getLogger(__name__)


class EmceeSolve(BaseSolver):
    """Solver based on MCMC approach in emcee :cite:p:`foreman-mackey_emcee_2013`.

    Parameters
    ----------
    objective_function: func, optional
        An objective function to be minimized. If not provided, the
        GaussianLikelihood is used. See the pastas.objective_functions module for
        more information.
    nwalkers: int, optional
        Number of walkers to use. Default is 20.
    backend: emcee.backend, optional
        One of the Backends from Emcee used to store MCMC results. See Emcee
        for more information.
    moves: emcee.moves, optional
        The moves argument determines how the next step for a walker is chosen in
        the MCMC approach. One of the Moves classes from Emcee has to be provided.
        See Emcee documentation for more information.
    parallel: bool, optional
        Run the sampler in parallel or not.
    progress_bar: bool, optional
        Show the progress bar or not. Requires the `tqdm` package to be installed.
    **kwargs, optional
        All other keyword arguments are passed on to the BaseSolver class.

    Notes
    -----
    The EmceeSolve solver uses the emcee package to perform a Markov Chain Monte Carlo
    (MCMC) approach to find the optimal parameter values. The solver can be used as
    follows::

        solver = ps.EmceeSolve(
            nwalkers=20,
            progress_bar=True,
        )
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
    pastas.objective_functions

    """

    def __init__(
        self,
        objfunction: GaussianLikelihood | GaussianLikelihoodAr1 | None = None,
        nwalkers: int = 20,
        backend=None,
        moves=None,
        parallel: bool = False,
        progress_bar: bool = True,
        **kwargs,
    ) -> None:
        # Check if emcee is installed, if not, return error
        try:
            global emcee
            import emcee as emcee  # Import emcee here, so it is no dependency
        except ImportError:
            msg = "emcee not installed. Please install emcee first."
            raise ImportError(msg) from None

        if "objective_function" in kwargs:
            deprecate_args_or_kwargs(
                "objective_function",
                "2.0.0",
                reason="Use the argument objfunction instead",
            )
            objfunction = kwargs.pop("objective_function")

        super().__init__(**kwargs)

        # Set sampler properties
        self.sampler = None
        self.backend = backend
        self.moves = moves
        self.parallel = parallel
        self.progress_bar = progress_bar
        self.nwalkers = nwalkers
        self.priors: list[DataFrame] = []

        # Set objective function
        self.objfunction = GaussianLikelihood() if objfunction is None else objfunction
        self.parameters = self.objfunction.get_init_parameters("ln")

    def fit_report(self) -> str:
        return ""

    def solve(
        self,
        noise: bool = False,
        weights: Series | None = None,
        steps: int = 5000,
        callback: CallBack | None = None,
        **kwargs,
    ) -> tuple[bool, ArrayLike, ArrayLike]:
        # Store initial parameters
        self.initial = np.append(
            self.ml.parameters.initial.values, self.parameters.initial.values
        )
        self.vary = np.append(
            self.ml.parameters.vary.values, self.parameters.vary.values
        )

        # Set lower and upper bounds
        lb = np.append(
            self.ml.parameters[self.ml.parameters.vary].pmin.values,
            self.parameters[self.parameters.vary].pmin.values,
        )
        ub = np.append(
            self.ml.parameters[self.ml.parameters.vary].pmax.values,
            self.parameters[self.parameters.vary].pmax.values,
        )
        self.bounds = np.vstack([lb, ub]).T

        # Set priors
        self._set_priors()

        # Set initial positions of the walkers
        pinit = np.append(
            self.ml.parameters[self.ml.parameters.vary].initial.values,
            self.parameters[self.parameters.vary].initial.values,
        )
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
                    pinit, steps, progress=self.progress_bar, **kwargs
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

            self.sampler.run_mcmc(pinit, steps, progress=self.progress_bar, **kwargs)

        # Get optimal values
        optimal = self.initial.copy()
        chains = self.sampler.get_chain(discard=0, flat=True, thin=1)
        optimal[self.vary] = chains[self.sampler.get_log_prob().argmax()]

        # Set the optimal values for the objective function parameters
        self.parameters.loc[:, "optimal"] = optimal[-self.objfunction.nparam :]

        # Don't estimate stderr for now
        optimal = optimal[: -self.objfunction.nparam]
        stderr = np.zeros(len(optimal)) * np.nan

        success = True
        return success, optimal, stderr

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

        weights
        callback

        Returns
        -------
        lnlike: float
            The log-likelihood for the parameters.

        Notes
        -----
        This method is always called by emcee.

        """
        par = self.initial

        # Set the parameters that are varied from the model and objective function
        par[self.vary] = p

        rv = self.misfit(
            p=par[: -self.objective_function.nparam],
            noise=noise,
            weights=weights,
            callback=callback,
        )

        lnlike = self.objective_function.compute(
            rv, par[-self.objective_function.nparam :]
        )

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

    def add_parameters_dist(self) -> None:
        """Add the distribution of the parameters to the parameters DataFrame."""
        # TODO: This method needs to be implemented such that the distribution
        # of the parameters is added to the model parameters DataFrame.

        # for i, prior in enumerate(self.priors):
        # self.ml.parameters.loc[self.ml.parameters.vary, "dist"] = prior.dist.name

    def _set_priors(self) -> None:
        """Set the priors for the parameters."""
        self.priors = []

        # Set the priors for the parameters that are varied from the model
        for _, (loc, pmin, pmax, scale, dist) in self.ml.parameters.loc[
            self.ml.parameters.vary, ["initial", "pmin", "pmax", "stderr"]
        ].iterrows():
            self.priors.append(self._get_prior(dist, loc, scale, pmin, pmax))

        # Set the priors for the parameters that are varied from the objective function
        for _, (loc, pmin, pmax, scale, dist) in self.parameters.loc[
            self.parameters.vary, ["initial", "pmin", "pmax", "stderr"]
        ].iterrows():
            self.priors.append(self._get_prior(dist, loc, scale, pmin, pmax))

    def _get_prior(self, dist: str, loc: float, scale: float, pmin: float, pmax: float):
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

    def to_dict(self) -> dict:
        """This method is not supported for this solver.

        Returns
        -------
        dict
        """
        # msg = "The EmceeSolve class does not support to_dict() and cannot be saved."
        # raise NotImplementedError(msg)
        return super().to_dict()
