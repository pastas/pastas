import importlib
from logging import getLogger

import numpy as np
from pandas import DataFrame, Series

from pastas.decorators import deprecate_args_or_kwargs
from pastas.typing import ArrayLike, CallBack

from .base import SolverBase
from .likelihood import GaussianLikelihood, GaussianLikelihoodAr1

logger = getLogger(__name__)


class EmceeSolve(SolverBase):
    """Solver based on MCMC approach in emcee :cite:p:`foreman-mackey_emcee_2013`.

    Parameters
    ----------
    objfunction: func, optional
        An objective function to be minimized. See the pastas.likelihood_functions module for more information.
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
    pastas.objective_functions

    """

    def __init__(
        self,
        name: str = "solver",
        objfunction: GaussianLikelihood
        | GaussianLikelihoodAr1
        | None = GaussianLikelihood(),
        nwalkers: int = 20,
        backend=None,
        moves=None,
        parallel: bool = False,
        progress_bar: bool = True,
        **kwargs,
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
        self.sampler = None
        self.backend = backend
        self.moves = moves
        self.parallel = parallel
        self.progress_bar = progress_bar
        self.nwalkers = nwalkers
        self.priors: list[DataFrame] = []

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

    def get_init_parameters(self, name):
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

    def misfit(
        self,
        p: ArrayLike,
        noise: bool,
        weights: Series | None = None,
        callback: CallBack | None = None,
        returnseparate: bool = False,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike]:
        """This method is called by all LeastSquares solvers to obtain a series that are
        minimized in the optimization process. It handles the application of the
        weights, a noisemodel and other optimization options.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the model
            parameters.
        noise: Boolean
            If True, minimizes the sum of squared noise computed by the NoiseModel.
        weights: pandas.Series, optional
            A pandas Series used to scale the residual or noise (in the case of a
            `NoiseModel`) during optimization. The weights must share the same
            `DateTimeIndex` as the observations (`ml.observations()`) to ensure proper
            alignment. These weights are applied such that the minimized objective
            function in least-squares solvers is ``sum((weights * residuals)**2)``.
            This means that a residual with double the weight has four times as much
            influence. If None, equal weights are used. This can be used to put extra/
            less weight on certain periods (e.g., droughts) or measurements (i.e.
            outliers), and make more complex calibration schemes (see, for example,
            :cite:`colllenteur_analysis_2023`).
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
            rv = self.ml.noise(p) * self.ml._noise_weights(p)
        else:
            rv = self.ml.residuals(p)

        # Determine if weights need to be applied
        if weights is not None:
            weights = weights.reindex(rv.index)
            weights.fillna(1.0, inplace=True)
            rv = rv.multiply(weights)

        if callback is not None:
            callback(p)

        if returnseparate:
            return (
                self.ml.residuals(p).to_numpy(copy=True),
                self.ml.noise(p).to_numpy(copy=True),
                self.ml._noise_weights(p).to_numpy(copy=True),
            )

        return rv.to_numpy(copy=True)

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
        self.initial = self.ml.parameters.initial.to_numpy(dtype=float)
        self.vary = self.ml.parameters.vary.to_numpy(dtype=bool)

        # Set lower and upper bounds
        lb = self.ml.parameters[self.ml.parameters.vary].pmin.to_numpy(dtype=float)
        ub = self.ml.parameters[self.ml.parameters.vary].pmax.to_numpy(dtype=float)
        self.bounds = np.vstack([lb, ub]).T

        # Set priors
        self._set_priors()

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
        # optimal = optimal[: -self.objfunction.nparam]
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
        par = self.initial.copy()

        # Set the parameters that are varied from the model and objective function
        par[self.vary] = p

        rv = self.misfit(
            p=par[: -self.objfunction.nparam],
            noise=noise,
            weights=weights,
            callback=callback,
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
        msg = "The EmceeSolve class does not support to_dict() and cannot be saved."
        raise NotImplementedError(msg)
        # return super().to_dict()
