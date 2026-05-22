"""This module contains all the response functions available in Pastas.

Examples
--------
Use response functions in stress models::

    rfunc = ps.Gamma()
    sm = ps.StressModel(stress, rfunc=rfunc, name="well")
    ml.add_stressmodel(sm)

"""

from abc import ABC, abstractmethod
from logging import getLogger

import numpy as np
from numpy import pi
from pandas import DataFrame, Series
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.signal import unit_impulse
from scipy.special import (
    erfc,
    erfcinv,
    exp1,
    factorial,
    gamma,
    gammainc,
    gammaincinv,
    k0,
    k0e,
    k1,
    kv,
    wrightomega,
)

from .decorators import njit

try:
    from numba import prange
except ImportError:
    prange = range

from typing import Literal

from pastas.decorators import PastasDeprecationWarning
from pastas.stats import moment
from pastas.typing import ArrayLike

logger = getLogger(__name__)

__all__ = [
    "Gamma",
    "Exponential",
    "Hantush",
    "Polder",
    "FourParam",
    "DoubleExponential",
    "One",
    # "Edelman",
    "HantushWellModel",
    "Kraijenhoff",
    "Spline",
]


class RfuncBase(ABC):
    """Base class for response functions."""

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        **kwargs,
    ) -> None:
        """Base class for response functions.

        Parameters
        ----------
        cutoff: float, optional
            Fraction of the step response after which the response is truncated.
            Default is 0.999.
        use_block: bool, optional
            Use the block response (rather than the impulse response) to simulate
            the effect of a stress. The block response approximates the stress
            as uniform during a time interval dt. When False, the impulse response
            is used which means that the the entire stress occurs midway the time
            interval dt. The impulse response is generally quicker to compute.
        kwargs: dict
            Additional keyword arguments.
        """
        self.cutoff = cutoff
        self.use_block = use_block
        if "up" in kwargs:
            raise TypeError(
                "keyword argument 'up' is not supported in init. "
                "Set with update_rfunc_settings()."
            )
        if "gain_scale_factor" in kwargs:
            raise TypeError(
                "keyword argument 'gain_scale_factor' is not supported in "
                "init. Set with update_rfunc_settings()."
            )
        # initialize attributes, these can be set with update_rfunc_settings()
        self.up = None
        self.gain_scale_factor = 1.0

    @property
    @abstractmethod
    def nparam(self) -> int:
        """Number of parameters of the response function."""

    @abstractmethod
    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters and bounds. It is called by the stressmodel.

        Parameters
        ----------
        name: str
            Name of the stressmodel.

        Returns
        -------
        parameters: pandas DataFrame
            The initial parameters and parameter bounds used by the solver.
        """

    @abstractmethod
    def get_tmax(self, p: ArrayLike, cutoff: float | None = None) -> float:
        """Method to get the response time for a certain cutoff.

        For instance, a cutoff of 0.99 returns the time when the step
        response has reached 99% of its upper limit, i.e. the gain.

        Parameters
        ----------
        p: array_like
            Response function parameters.
        cutoff: float, optional
            Fraction of the step response used to determine the response cutoff.
            Defaults to `self.cutoff` if `cutoff is None`.

        Returns
        -------
        tmax: float
        """

    def _resolve_tmax(
        self,
        p: ArrayLike,
        cutoff: float | None = None,
        **kwargs,
    ) -> float:
        """Internal hook to determine `tmax` from :meth:`get_tmax`.

        Subclasses can override this method to support extra keyword arguments.
        """
        return self.get_tmax(p=p, cutoff=cutoff)

    @abstractmethod
    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        """Method to return the step function.

        Parameters
        ----------
        p: array_like
            Response function parameters.
        dt: float
            Time step in days.
        cutoff: float, optional
            Fraction of the step response used to determine the response cutoff.
            Defaults to `self.cutoff` if `cutoff is None`.
        maxtmax: float, optional
            Maximum response time to compute. Not used if None, else, the used
            tmax is the minimum of the tmax determined from the cutoff and maxtmax.
        kwargs: dict
            Additional keyword arguments passed to :meth:`get_t` or used for specific
            response functions.

        Returns
        -------
        s: array_like
            Array with the step response.
        """

    @abstractmethod
    def gain(self, p: ArrayLike) -> float:
        """Method to return the gain for the response function."""

    @abstractmethod
    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        """Compute the raw moment of the response function.

        Parameters
        ----------
        p: array_like
            Response function parameters.
        order : int
            Order of the moment to compute.
        method : {'discrete', 'exact'}, optional
            Method used to compute the moment. `"discrete"` uses the discrete
            block response, while `"exact"` uses an analytical expression when
            available. Default is `"discrete"`.
        dt : float, optional
            Time step in days. Default is 1.0.
        """

    @staticmethod
    @abstractmethod
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        """Method to return the impulse response function.

        Parameters
        ----------
        t: array_like
            array_like object with the times at which to evaluate the impulse
            response, can be obtained with get_t() method
        p: array_like
            array_like object with the values as floats representing the model
            parameters.

        Returns
        -------
        s: array_like
            Array with the impulse response.
        """

    @property
    def _name(self) -> str:
        return self.__class__.__name__

    def update_rfunc_settings(
        self,
        up: bool | Literal["nochange"] = "nochange",
        gain_scale_factor: float | None = None,
        cutoff: float | None = None,
    ) -> None:
        """Internal method to set the settings of the response function.

        Parameters
        ----------
        up: bool or None, optional
            Whether a positive stress causes the head to go up (`True`), down
            (`False`), or either direction (`None`).
        gain_scale_factor: float, optional
            Scale factor used to set the initial value and bounds of the gain
            parameter, computed as `1 / gain_scale_factor`.
        cutoff: float, optional
            Fraction of the step response after which the response is truncated.

        """
        if up != "nochange":
            self.up = up

        if gain_scale_factor is not None:
            if 1e-8 > gain_scale_factor > 0:
                gain_scale_factor = 1e-8  # arbitrary number to prevent division by zero
            elif gain_scale_factor < 0 and up is True:
                gain_scale_factor = gain_scale_factor * -1
            elif gain_scale_factor == 0.0:
                msg = "The gain_scale_factor, the factor to scale the initial value of"
                " the gain parameter, cannot be zero, setting to 1.0. Consider "
                "providing a custom gain_scale_factor."
                logger.warning(msg)
                gain_scale_factor = 1.0
            self.gain_scale_factor = gain_scale_factor

        if cutoff is not None:
            self.cutoff = cutoff

    def get_t(
        self,
        p: ArrayLike,
        dt: float | ArrayLike,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        """Internal method to determine the times at which to evaluate the step
        response, from t=0.

        Parameters
        ----------
        p: array_like
            Response function parameters.
        dt: float | ArrayLike
            Time step in days. If an array_like is provided, it is returned as is
            and the cutoff and maxtmax parameters are ignored. If a float is
            provided, the times are computed from dt until the tmax determined
            from the cutoff and maxtmax parameters.
        cutoff: float | None, optional
            Fraction of the step response used to determine the response cutoff.
        maxtmax: float | None, optional
            Maximum response time to compute,. Not used if None, else, the used
            tmax is the minimum of the tmax determined from the cutoff and maxtmax.
        kwargs: dict
            Additional keyword arguments used by specific response functions.

        Returns
        -------
        t: array_like
            Times at which the response is evaluated.
        """
        if np.ndim(dt) > 0:
            return np.asarray(dt, dtype=float)

        tmax = self._resolve_tmax(p=p, cutoff=cutoff, **kwargs)
        # make sure tmax is at least 3*dt such that len(t) is at least 2
        # make sure tmax does not exceed maxtmax if provided
        tmax = max(min(tmax, maxtmax) if maxtmax is not None else tmax, 3 * dt)
        t = np.arange(dt, stop=tmax, step=dt, dtype=float)
        return t

    def block(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        """Method to return the block function.

        Parameters
        ----------
        p: array_like
            Response function parameters.
        dt: float
            Time step in days.
        cutoff: float, optional
            Fraction of the step response used to determine the response cutoff.
            Defaults to `self.cutoff` if `cutoff is None`.
        maxtmax: float, optional
            Maximum response time to compute. Not used if None, else, the used
            tmax is the minimum of the tmax determined from the cutoff and maxtmax.
        kwargs: dict
            Additional keyword arguments passed to :meth:`block_from_step` or
            :meth:`block_from_impulse`.

        Returns
        -------
        array_like
            Block response values.
        """
        if self.use_block:
            b = self.block_from_step(
                p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs
            )
        else:
            b = self.block_from_impulse(
                p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs
            )
        return b

    def block_from_impulse(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        """Method to return the block function from the impulse response.

        Parameters
        ----------
        p: array_like
            Response function parameters.
        dt: float
            Time step in days.
        cutoff: float, optional
            Fraction of the step response used to determine the response cutoff.
            Defaults to `self.cutoff` if `cutoff is None`.
        maxtmax: float, optional
            Maximum response time to compute. Not used if None, else, the used
            tmax is the minimum of the tmax determined from the cutoff and maxtmax.

        Returns
        -------
        array_like
            Block response values.
        """
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
        t_mid = t - (0.5 * dt)  # compute times at the middle of the interval
        b = self.impulse(t_mid, p) * dt
        return b

    def block_from_step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        """Method to return the block function from the step response.

        Parameters
        ----------
        p: array_like
            Response function parameters.
        dt: float
            Time step in days.
        cutoff: float, optional
            Fraction of the step response used to determine the response cutoff.
            Defaults to `self.cutoff` if `cutoff is None`.
        maxtmax: float, optional
            Maximum response time to compute. Not used if None, else, the used
            tmax is the minimum of the tmax determined from the cutoff and maxtmax.
        kwargs: dict
            Additional keyword arguments passed to :meth:`step`.

        Returns
        -------
        b: array_like
            Block response values.
        """
        s = self.step(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
        b = np.append(s[0], np.subtract(s[1:], s[:-1]))
        return b

    def to_dict(self):
        """Method to export the response function to a dictionary.

        Returns
        -------
        dict[str, Any]
            dictionary with all necessary settings to reconstruct the rfunc object.

        Notes
        -----
        The exported dictionary matches the input arguments of `__init__`.

        """
        settings = {
            "class": self._name,
            "cutoff": self.cutoff,
            "use_block": self.use_block,
            "up": self.up,
            "gain_scale_factor": self.gain_scale_factor,
        }
        return settings


class Gamma(RfuncBase):
    """Gamma response function with 3 parameters A, a, and n.

    Parameters
    ----------
    cutoff: float, optional
        Fraction of the step response after which the response is truncated.
        Default is 0.999.
    use_block: bool, optional
        Use the block response (rather than the impulse response) to simulate
        the effect of a stress. The block response approximates the stress
        as uniform during a time interval dt. When False, the impulse response
        is used which means that the the entire stress occurs midway the time
        interval dt. The impulse response is generally quicker to compute.

    Attributes
    ----------
    up: bool or None, optional
        Whether a positive stress causes the head to go up (`True`), down
        (`False`), or either direction (`None`).
    gain_scale_factor: float, optional
        Mean stress value used to scale the initial value so that the final step
        response times the mean stress equals 1.
    """

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(cutoff=cutoff, use_block=use_block, **kwargs)

    @property
    def nparam(self) -> int:
        return 3

    def get_init_parameters(self, name: str) -> DataFrame:
        if self.up:
            initial_A, pmin_A, pmax_A = (
                1.0 / self.gain_scale_factor,
                1e-5,
                100.0 / self.gain_scale_factor,
            )
        elif self.up is False:
            initial_A, pmin_A, pmax_A = (
                -1.0 / self.gain_scale_factor,
                -100.0 / self.gain_scale_factor,
                -1e-5,
            )
        else:
            initial_A, pmin_A, pmax_A = 1.0 / self.gain_scale_factor, np.nan, np.nan

        parameters = DataFrame(
            [
                (initial_A, pmin_A, pmax_A, True, name, "uniform"),
                (1.0, 0.1, 5.0, True, name, "uniform"),
                (10.0, 1e-2, 1e4, True, name, "uniform"),
            ],
            index=[name + "_A", name + "_n", name + "_a"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: float | None = None) -> float:
        if cutoff is None:
            cutoff = self.cutoff
        return gammaincinv(p[1], cutoff) * p[2]

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
        s = p[0] * gammainc(p[1], t / p[2])
        return s

    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        if method == "discrete":
            t = self.get_t(p=p, dt=dt, cutoff=self.cutoff)
            b = Series(self.block(p=p, dt=dt, cutoff=self.cutoff), index=t)
            return moment(b, order)
        elif method == "exact":
            A, n, a = p
            return A * gamma(n + order) / gamma(n) * a**order
        else:
            raise ValueError(f"Invalid method {method}. Choose 'discrete' or 'exact'.")

    @staticmethod
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, n, a = p
        return A * t ** (n - 1) * np.exp(-t / a) / (a**n * gamma(n))


class Exponential(RfuncBase):
    """Exponential response function with 2 parameters: A and a.

    Parameters
    ----------
    cutoff: float, optional
        Fraction of the step response after which the response is truncated.
        Default is 0.999.
    use_block: bool, optional
        Use the block response (rather than the impulse response) to simulate
        the effect of a stress. The block response approximates the stress
        as uniform during a time interval dt. When False, the impulse response
        is used which means that the the entire stress occurs midway the time
        interval dt. The impulse response is generally quicker to compute.

    Attributes
    ----------
    up: bool or None, optional
        Whether a positive stress causes the head to go up (`True`), down
        (`False`), or either direction (`None`).
    gain_scale_factor: float, optional
        Scale factor used to set the initial value and bounds of the gain
        parameter, computed as `1 / gain_scale_factor`.

    """

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(cutoff=cutoff, use_block=use_block, **kwargs)

    @property
    def nparam(self) -> int:
        return 2

    def get_init_parameters(self, name: str) -> DataFrame:
        # Determine initial, pmin, pmax for parameter A based on self.up
        if self.up:
            initial_A, pmin_A, pmax_A = (
                1.0 / self.gain_scale_factor,
                1e-5,
                1e2 / self.gain_scale_factor,
            )
        elif self.up is False:
            initial_A, pmin_A, pmax_A = (
                -1.0 / self.gain_scale_factor,
                -1e2 / self.gain_scale_factor,
                -1e-5,
            )
        else:
            initial_A, pmin_A, pmax_A = 1.0 / self.gain_scale_factor, np.nan, np.nan

        parameters = DataFrame(
            [
                (initial_A, pmin_A, pmax_A, True, name, "uniform"),
                (10.0, 1e-2, 1e4, True, name, "uniform"),
            ],
            index=[name + "_A", name + "_a"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff=None) -> float:
        if cutoff is None:
            cutoff = self.cutoff
        return -p[1] * np.log(1 - cutoff)

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
        s = p[0] * (1.0 - np.exp(-t / p[1]))
        return s

    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        if method == "discrete":
            t = self.get_t(p=p, dt=dt, cutoff=self.cutoff)
            b = Series(self.block(p=p, dt=dt, cutoff=self.cutoff), index=t)
            return moment(b, order)
        elif method == "exact":
            A, a = p
            return A * factorial(order) * a**order
        else:
            raise ValueError(f"Invalid method {method}. Choose 'discrete' or 'exact'.")

    @staticmethod
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, a = p
        return A / a * np.exp(-t / a)


class Hantush(RfuncBase):
    """The Hantush well function, using the standard A, a, b parameters.

    Parameters
    ----------
    cutoff: float, optional
        Fraction of the step response after which the response is truncated.
        Default is 0.999.
    use_block: bool, optional
        Use the block response (rather than the impulse response) to simulate
        the effect of a stress. The block response approximates the stress
        as uniform during a time interval dt. When False, the impulse response
        is used which means that the the entire stress occurs midway the time
        interval dt. The impulse response is generally quicker to compute.
    quad: bool, optional
        Use `quad_step` to compute the step response using numerical
        integration. Default is False.
    approximate_tmax: bool, optional
        If True, get_tmax will use the fast Lambert W approximation (default). If False,
        it will use the exact numerical root finding method.

    Attributes
    ----------
    up: bool or None, optional
        Whether a positive stress causes the head to go up (`True`), down
        (`False`), or either direction (`None`).
    gain_scale_factor: float, optional
        Scale factor used to set the initial value and bounds of the gain
        parameter, computed as `1 / gain_scale_factor`.

    Notes
    -----
    The implementation used here is explained in :cite:t:`veling_hantush_2010`.

    """

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        quad: bool = False,
        approximate_tmax: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(cutoff=cutoff, use_block=use_block, **kwargs)
        self.quad = quad
        self.approximate_tmax = approximate_tmax
        if self.quad and not self.approximate_tmax:
            logger.warning(
                "Using quad_step with approximate_tmax=False can lead to long "
                "computation times for get_tmax. Consider setting "
                "approximate_tmax=True or quad=False."
            )

    @property
    def nparam(self) -> int:
        return 3

    def get_init_parameters(self, name: str) -> DataFrame:
        if self.up:
            initial_A, pmin_A, pmax_A = (
                1.0 / self.gain_scale_factor,
                0.0,
                100.0 / self.gain_scale_factor,
            )
        elif self.up is False:
            initial_A, pmin_A, pmax_A = (
                -1.0 / self.gain_scale_factor,
                -100.0 / self.gain_scale_factor,
                0.0,
            )
        else:
            initial_A, pmin_A, pmax_A = 1.0 / self.gain_scale_factor, np.nan, np.nan

        parameters = DataFrame(
            [
                (initial_A, pmin_A, pmax_A, True, name, "uniform"),
                (1e2, 1e-3, 1e4, True, name, "uniform"),
                (1.0, 1e-6, 25.0, True, name, "uniform"),
            ],
            index=[name + "_A", name + "_a", name + "_b"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        return parameters

    def get_tmax_approximation(
        self, p: ArrayLike, cutoff: float | None = None
    ) -> float:
        """Approximates the time (tmax) when the step response reaches a specified cutoff.

        This analytical approximation is derived by evaluating the tail of the
        impulse response integral. The derivation relies on the following steps:
        1.  The tail integral is mapped to dimensionless time x = t/a.
        2.  Deep in the tail, the b/t term in the exponent is assumed negligible,
            reducing the integral to the standard Exponential Integral, E1(x).
        3.  E1(x) is approximated by its leading asymptotic term: exp(-x) / x.
        4.  Equating this to the remaining area (1 - cutoff) yields an equation
            of the form x * exp(x) = z, which is classically solved using the
            Lambert W function: x = W(z).

        To prevent crashing the calculation when rho is large (causing k0 to
        underflow and z to overflow), the math is translated into log-space.
        The exponentially scaled Bessel function (k0e) and the Wright Omega
        function (the exact analytical solution to y + ln(y) = log_z)
        are used. This provides a globally continuous, unbreakable equivalent to
        lambertw(z).

        Parameters
        ----------
        p : array_like
            Response function parameters `[A, a, b]`.
        cutoff : float, optional
            The fraction of the total step response area reached at tmax.
            Must be strictly between 0 and 1. Defaults to `self.cutoff` if
            `cutoff is None`.

        Returns
        -------
        float
            The approximated tmax value.
        """
        cutoff = self.cutoff if cutoff is None else cutoff
        a, b = p[1], p[2]
        rho = 2.0 * np.sqrt(b)

        # Compute log-space equivalent of z = 1 / ((1 - cutoff) * k0(rho))
        # to prevent large values of rho returning 0.0 for k0(rho)
        # and thus z = inf, which causes lambertw to return inf.
        # ln(k0(rho)) = ln(k0e(rho)) - rho
        log_z = rho - np.log(1 - cutoff) - np.log(k0e(rho))

        # wrightomega(L) = lambertw(exp(L))
        tmax = wrightomega(log_z).real * a
        return tmax

    def _f_step(self, t: float, A: float, a: float, b: float, cutoff: float) -> float:
        """Objective function for root finding (t varies, other params fixed)."""
        t_arr = np.array([t], dtype=float)
        if self.quad:
            step_val = self.quad_step(A=A, a=a, b=b, t=t_arr)[0]
        else:
            step_val = self.numpy_step(A=A, a=a, b=b, t=t_arr)[0]
        return (step_val / A) - cutoff

    def get_tmax(self, p: ArrayLike, cutoff: float | None = None) -> float:
        """Calculate `tmax` using either the approximation or root finding.

        Parameters
        ----------
        p: array_like
            Response function parameters.
        cutoff: float, optional
            Fraction of the step response used to determine the response cutoff.
            Defaults to `self.cutoff` if `cutoff is None`.

        Returns
        -------
        float
            Response time in days corresponding to the selected cutoff.
        """

        cutoff = self.cutoff if cutoff is None else cutoff

        t0 = self.get_tmax_approximation(p, cutoff)
        if self.approximate_tmax:
            return t0

        A, a, b = p[0], p[1], p[2]

        # Use Brentq's method
        tol = min(10.0 ** np.floor(np.log10(t0)) / 1e2, 0.1)
        root, info = brentq(
            f=self._f_step,
            a=1e-30,  # avoid divide by zero warnings
            b=t0,
            xtol=tol,
            maxiter=100,  # generally converges within 10 iterations
            args=(A, a, b, cutoff),
            full_output=True,
            disp=False,
        )
        # Check the convergence flag directly
        if info.converged:
            logger.debug(
                "Root finding for tmax converged successfully. Brentq RootResults: %s",
                info,
            )
            return root
        else:
            logger.warning(
                "Root finding for tmax did not converge, returning approximate tmax. "
                "Consider setting approximate_tmax=True for the Hantush response. "
                "Brentq RootResults: %s",
                info,
            )
            return t0

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    @staticmethod
    def numpy_step(A: float, a: float, b: float, t: ArrayLike) -> ArrayLike:
        rho = 2.0 * np.sqrt(b)
        k0rho = k0(rho)
        if k0rho == 0.0:
            logger.warning(
                f"K_0(rho) is underflowing to 0.0 for b: {b:.4e}, rho = {rho:.4e}. "
                "The parameter `b` is too high or which means that the observation well "
                "is too far away. Consider lowering the initial value and bounds for b "
                "to prevent this error."
            )
        exp1_rho = exp1(rho)
        w = (exp1_rho - k0rho) / (exp1_rho - exp1(rho / 2.0))
        w_minus_1 = w - 1.0
        tau = t / a
        b_over_tau = b / tau

        F = np.empty_like(tau)
        mask = tau < (rho / 2.0)
        inv_mask = ~mask

        tau1 = tau[mask]
        b_tau1 = b_over_tau[mask]
        F[mask] = w * exp1(b_tau1) - w_minus_1 * exp1(tau1 + b_tau1)

        tau2 = tau[inv_mask]
        b_tau2 = b_over_tau[inv_mask]
        F[inv_mask] = 2.0 * k0rho - w * exp1(tau2) + w_minus_1 * exp1(tau2 + b_tau2)

        return A * F / (2.0 * k0rho)

    @staticmethod
    @njit
    def _integrand_hantush(y: float, b: float) -> float:
        return np.exp(-y - (b / y)) / y

    @staticmethod
    def quad_step(A: float, a: float, b: float, t: ArrayLike) -> ArrayLike:
        F = np.zeros_like(t)
        u = a * b / t
        for i in range(0, len(t)):
            F[i] = quad(Hantush._integrand_hantush, u[i], np.inf, args=(b,))[0]
        return F * A / (2 * k0(2 * np.sqrt(b)))

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        A, a, b = p
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)

        step = (
            self.quad_step(A=A, a=a, b=b, t=t)
            if self.quad
            else self.numpy_step(A=A, a=a, b=b, t=t)
        )
        return step

    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        if method == "discrete":
            t = self.get_t(p=p, dt=dt, cutoff=self.cutoff)
            b = Series(self.block(p=p, dt=dt, cutoff=self.cutoff), index=t)
            return moment(b, order)
        elif method == "exact":
            A, a, b = p
            return (
                (a**2 * b) ** (order / 2)
                * kv(order, 2 * np.sqrt(b))
                / kv(0, 2 * np.sqrt(b))
            )
        else:
            raise ValueError(f"Invalid method {method}. Choose 'discrete' or 'exact'.")

    @staticmethod
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, a, b = p
        return A / (2 * t * k0(2 * np.sqrt(b))) * np.exp(-t / a - a * b / t)

    def to_dict(self):
        settings = super().to_dict() | {
            "quad": self.quad,
            "approximate_tmax": self.approximate_tmax,
        }
        return settings


class HantushWellModel(RfuncBase):
    """An implementation of the Hantush well function for multiple pumping wells.

    Parameters
    ----------
    cutoff: float, optional
        Fraction of the step response after which the response is truncated.
        Default is 0.999.
    use_block: bool, optional
        Use the block response (rather than the impulse response) to simulate
        the effect of a stress. The block response approximates the stress
        as uniform during a time interval dt. When False, the impulse response
        is used which means that the the entire stress occurs midway the time
        interval dt. The impulse response is generally quicker to compute.
    quad: bool, optional
        Use `quad_step` to compute the step response using numerical
        integration. Default is False.
    approximate_tmax: bool, optional
        If True, get_tmax will use the fast Lambert W approximation (default). If
        False, it will use the exact numerical root finding method.
    log_b: bool, optional
        Whether to use log10 scaling for parameter b, Default is True. In this Hantush
        implementation parameter b is multiplied by distances squared meaning values
        of b can get very small under certain conditions. Log scaling can help with
        optimization when the value of b is very small and the range of b spans
        several orders of magnitude.

    Attributes
    ----------
    up: bool, optional
        Whether a positive stress causes the head to go up (`True`) or down
        (`False`).
    gain_scale_factor: float, optional
        Scale factor used to set the initial value and bounds of the gain
        parameter, computed as `1 / gain_scale_factor`.

    Notes
    -----
    where r is the distance from the pumping well to the observation point and must
    be specified. A, a, and b are parameters, which are slightly different from the
    Hantush response function. The gain is defined as:

    :math:`\\text{gain} = A K_0 \\left( 2r \\sqrt(b) \\right)`


    The implementation used here is explained in :cite:t:`veling_hantush_2010`.
    """

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        quad: bool = False,
        approximate_tmax: bool = True,
        log_b: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(cutoff=cutoff, use_block=use_block, **kwargs)
        self.distances: float | ArrayLike | None = None
        self.quad: bool = quad
        self.approximate_tmax: bool = approximate_tmax
        self.log_b: bool = log_b

    @property
    def nparam(self) -> int:
        return 3

    def set_distances(self, distances: float | ArrayLike) -> None:
        """Method to set the distances from the pumping well(s) to the observation well."""
        self.distances: float | ArrayLike = distances

    def get_init_parameters(self, name: str) -> DataFrame:
        if self.distances is None:
            raise (
                ValueError(
                    "distances is None. Set using method set_distances() or use "
                    "Hantush."
                )
            )
        # Calculate initial, min, max for A  (divide by k0(2) to get same initial value as ps.Hantush)
        if self.up:
            initial_A, pmin_A, pmax_A = (
                1.0 / (self.gain_scale_factor * k0(2)),
                0.0,
                np.nan,
            )
        elif self.up is False:
            initial_A, pmin_A, pmax_A = (
                -1.0 / (self.gain_scale_factor * k0(2)),
                np.nan,
                0.0,
            )
        else:
            initial_A, pmin_A, pmax_A = 1.0 / self.gain_scale_factor, np.nan, np.nan

        initial_b, pmin_b, pmax_b = (
            1.0 / np.mean(self.distances) ** 2,
            1e-6 / np.max(self.distances) ** 2,
            25.0 / np.min(self.distances) ** 2,
        )
        if self.log_b:
            initial_b, pmin_b, pmax_b = (
                np.log10(initial_b),
                np.log10(pmin_b),
                np.log10(pmax_b),
            )

        parameters = DataFrame(
            [
                (initial_A, pmin_A, pmax_A, True, name, "uniform"),
                (1e2, 1e-3, 1e4, True, name, "uniform"),
                # set initial and bounds for b taking into account distances
                # note log transform to avoid tiny values for b
                (initial_b, pmin_b, pmax_b, True, name, "uniform"),
            ],
            index=[name + "_A", name + "_a", name + "_b"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        return parameters

    @staticmethod
    def _get_distance_from_params(p: ArrayLike, warn: bool = True) -> float:
        """Internal method to get the distance from the parameters. If the distance is not
        provided, it assumes a distance of 1.0 and raises a warning if warn is True.
        """
        if len(p) == 3:
            r = 1.0
            if warn:
                logger.info("No distance passed to HantushWellModel, assuming r=1.0.")
        else:
            r = p[3]
        return r

    def _get_hantush_params(self, p: ArrayLike, warn: bool = True) -> np.ndarray:
        """Internal method to convert the HantushWellModel to the Hantush parameters"""
        r = self._get_distance_from_params(p, warn=warn)
        A, a, b = p[:3]
        b_scaled = 10 ** (b / 2.0) if self.log_b else np.sqrt(b)
        rho = 2.0 * r * b_scaled
        A_h = A * k0(rho)
        b_h = (r * b_scaled) ** 2
        return np.array([A_h, a, b_h])

    def get_tmax(
        self, p: ArrayLike, cutoff: float | None = None, warn: bool = True
    ) -> float:
        cutoff = self.cutoff if cutoff is None else cutoff
        p_h = self._get_hantush_params(p, warn=warn)
        h = Hantush(
            cutoff=self.cutoff,
            quad=self.quad,
            approximate_tmax=self.approximate_tmax,
        )
        return h.get_tmax(p_h, cutoff=cutoff)

    def _resolve_tmax(
        self,
        p: ArrayLike,
        cutoff: float | None = None,
        **kwargs,
    ) -> float:
        """Internal hook to determine `tmax` from :meth:`get_tmax`,
        with support for extra keyword arguments which is needed for
        the warn argument used in the _get_distance_from_params method.
        """
        warn = kwargs.get("warn", True)
        return self.get_tmax(p, cutoff=cutoff, warn=warn)

    def gain(self, p: ArrayLike, r: float | None = None) -> float:
        if r is None:
            r = self._get_distance_from_params(p)
        b_scaled = 10 ** (p[2] / 2.0) if self.log_b else np.sqrt(p[2])
        rho = 2.0 * r * b_scaled
        return p[0] * k0(rho)

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        warn: bool = True,
        **kwargs,
    ) -> ArrayLike:
        p_h = self._get_hantush_params(p, warn=warn)
        kwargs["warn"] = warn
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)

        if self.quad:
            return Hantush.quad_step(p_h[0], p_h[1], p_h[2], t)
        else:
            return Hantush.numpy_step(p_h[0], p_h[1], p_h[2], t)

    def block_from_impulse(
        self,
        p: ArrayLike,
        dt: float = 1,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
        t_mid = t - (0.5 * dt)  # compute times at the middle of the interval
        p = self._get_hantush_params(p, warn=False)
        return self.impulse(t=t_mid, p=p) * dt

    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        if method == "discrete":
            t = self.get_t(p=p, dt=dt, cutoff=self.cutoff)
            b = Series(self.block(p=p, dt=dt, cutoff=self.cutoff), index=t)
            return moment(b, order)
        else:
            raise ValueError(
                f"Invalid method {method}. Only 'discrete' is supported for "
                f"{self._name}."
            )

    @staticmethod
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        # A, a, b, r = p
        # b = 10**b if log_b else b
        # A / 2 * t * np.exp(-t / a - a * b * r**2 / t)
        return Hantush.impulse(t=t, p=p)

    @staticmethod
    def variance_gain(
        A: float,
        b: float,
        var_A: float,
        var_b: float,
        cov_Ab: float,
        r: float = 1.0,
        log_b: bool = True,
    ) -> float | ArrayLike:
        """Calculate variance of the gain from parameters A and b.

        Variance of the gain is calculated based on propagation of uncertainty using
        optimal values, the variances of A and b and the covariance between A and b.

        Notes
        -----
        Estimated variance can be biased for non-linear functions as it uses
        truncated series expansion.

        Parameters
        ----------
        A : float
            optimal value of parameter A, (e.g. ml.parameters.optimal).
        b : float
            optimal value of parameter b, (e.g. ml.parameters.optimal).
        var_A : float
            variance of parameter A, can be obtained from the diagonal of the
            covariance matrix (e.g. ml.solver.pcov).
        var_b : float
            variance of parameter b, can be obtained from the diagonal of the
            covariance matrix (e.g. ml.solver.pcov).
        cov_Ab : float
            covariance between A and b, can be obtained from the covariance matrix (
            e.g. ml.solver.pcov).
        r : float or array_like, optional
            distance(s) between observation well and stress(es), default value is 1.0.
        log_b: bool, optional
            indicates if parameter b is log10 transformed. Default is True.

        Returns
        -------
        var_gain : float or array_like
            variance of the gain calculated based on propagation of uncertainty of
            parameters A and b.

        See Also
        --------
        ps.WellModel.variance_gain
        """
        if log_b:
            b_scaled = 10 ** (b / 2.0)
            db_scaled = b_scaled * np.log(10) / 2.0
        else:
            b_scaled = np.sqrt(b)
            db_scaled = 0.5 / np.sqrt(b)

        rho = 2.0 * r * b_scaled
        drho_db = 2.0 * r * db_scaled

        dg_dA = k0(rho)
        dg_db = -A * k1(rho) * drho_db

        var_gain = dg_dA**2 * var_A + dg_db**2 * var_b + 2 * dg_dA * dg_db * cov_Ab
        return var_gain

    def to_dict(self):
        settings = super().to_dict() | {
            "quad": self.quad,
            "approximate_tmax": self.approximate_tmax,
            "log_b": self.log_b,
        }
        return settings


class Polder(RfuncBase):
    """The Polder function, using the standard A, a, b parameters.

    Parameters
    ----------
    cutoff: float, optional
        Fraction of the step response after which the response is truncated.
        Default is 0.999.
    use_block: bool, optional
        Use the block response (rather than the impulse response) to simulate
        the effect of a stress. The block response approximates the stress
        as uniform during a time interval dt. When False, the impulse response
        is used which means that the the entire stress occurs midway the time
        interval dt. The impulse response is generally quicker to compute.

    Attributes
    ----------
    up: bool or None, optional
        Whether a positive stress causes the head to go up (`True`), down
        (`False`), or either direction (`None`).
    gain_scale_factor: float, optional
        Scale factor used to set the initial value and bounds of the gain
        parameter, computed as `1 / gain_scale_factor`.

    Notes
    -----
    The function is explained in Eq. 123.32 in :cite:t:`bruggeman_analytical_1999`.

    """

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(cutoff=cutoff, use_block=use_block, **kwargs)

    @property
    def nparam(self) -> int:
        return 3

    def get_init_parameters(self, name) -> DataFrame:
        parameters = DataFrame(
            [
                (
                    1.0 if self.up else -1.0 if self.up is False else 1.0,
                    0.0 if self.up else -2.0 if self.up is False else -2.0,
                    2.0 if self.up else 0.0 if self.up is False else 2.0,
                    True,
                    name,
                    "uniform",
                ),
                (10.0, 1e-2, 1e3, True, name, "uniform"),
                (1.0, 1e-6, 25.0, True, name, "uniform"),
            ],
            index=[name + "_A", name + "_a", name + "_b"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: float | None = None) -> float:
        if cutoff is None:
            cutoff = self.cutoff
        _, a, b = p
        b = a * b
        x = np.sqrt(b / a)
        inverfc = erfcinv(2 * cutoff)
        y = (-inverfc + np.sqrt(inverfc**2 + 4 * x)) / 2
        tmax = a * y**2
        return tmax

    def gain(self, p: ArrayLike) -> float:
        # the steady state solution of Mazure
        g = p[0] * np.exp(-np.sqrt(4 * p[2]))
        return g

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
        A, a, b = p
        s = A * self.polder_function(np.sqrt(b), np.sqrt(t / a))
        # / np.exp(-2 * np.sqrt(b))
        return s

    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        if method == "discrete":
            t = self.get_t(p=p, dt=dt, cutoff=self.cutoff)
            b = Series(self.block(p=p, dt=dt, cutoff=self.cutoff), index=t)
            return moment(b, order)
        elif method == "exact":
            A, a, b = p
            return (
                A
                * 2
                / np.sqrt(pi)
                * (a * a * b) ** (order / 2)
                * kv(order - 0.5, 2 * np.sqrt(b))
            )
        else:
            raise ValueError(f"Invalid method {method}. Choose 'discrete' or 'exact'.")

    @staticmethod
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, a, b = p
        return A * np.sqrt(a * b / pi) * t ** (-1.5) * np.exp(-t / a - a * b / t)

    @staticmethod
    def polder_function(x: float, y: float) -> float:
        return 0.5 * np.exp(2 * x) * erfc(x / y + y) + 0.5 * np.exp(-2 * x) * erfc(
            x / y - y
        )


class One(RfuncBase):
    """Instant response with no lag and one parameter A.

    Parameters
    ----------
    cutoff: float, optional
        Fraction of the step response after which the response is truncated. This
        setting has no effect for this response function.
    use_block: bool, optional
        Use the block response (rather than the impulse response) to simulate
        the effect of a stress. The block response approximates the stress
        as uniform during a time interval dt. When False, the impulse response
        is used which means that the the entire stress occurs midway the time
        interval dt. The impulse response is generally quicker to compute.

    Attributes
    ----------
    up: bool or None, optional
        Whether a positive stress causes the head to go up (`True`), down
        (`False`), or either direction (`None`).
    gain_scale_factor: float, optional
        Scale factor used to set the initial value and bounds of the gain
        parameter, computed as `1 / gain_scale_factor`.

    """

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(cutoff=cutoff, use_block=use_block, **kwargs)

    @property
    def nparam(self) -> int:
        return 1

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            [
                (
                    self.gain_scale_factor
                    if self.up
                    else -self.gain_scale_factor
                    if self.up is False
                    else self.gain_scale_factor,
                    0.0 if self.up else np.nan if self.up is False else np.nan,
                    np.nan if self.up else 0.0 if self.up is False else np.nan,
                    True,
                    name,
                    "uniform",
                )
            ],
            index=[name + "_A"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: float | None = None) -> float:
        return 1.0

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        if isinstance(dt, np.ndarray):
            return np.full(len(dt), p[0], dtype=float)
        else:
            return np.full(1, p[0], dtype=float)

    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        if method == "discrete":
            if order == 0:
                return self.gain(p)
            else:
                return 0.0
        else:
            raise ValueError(
                f"Invalid method {method}. Only 'discrete' is supported for "
                f"{self._name}."
            )

    @staticmethod
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        return unit_impulse(t.shape, idx=0, dtype=float) * p[0]


class FourParam(RfuncBase):
    """Four Parameter response function with 4 parameters A, a, b, and n.

    Parameters
    ----------
    cutoff: float, optional
        Fraction of the step response after which the response is truncated.
        Default is 0.999.
    use_block: bool, optional
        Use the block response (rather than the impulse response) to simulate
        the effect of a stress. The block response approximates the stress
        as uniform during a time interval dt. When False, the impulse response
        is used which means that the the entire stress occurs midway the time
        interval dt. The impulse response is generally quicker to compute.
    quad: bool, optional
        If true, use the 'quad' method from scipy.integrate to integrate the impulse
        response function. This may be more accurate but increases computation times.
    approximate_tmax: bool, optional
        If True, get_tmax will use a fast numerical approximation (default). If False,
        it will use the exact numerical root finding method.

    Attributes
    ----------
    up: bool or None, optional
        Whether a positive stress causes the head to go up (`True`), down
        (`False`), or either direction (`None`).
    gain_scale_factor: float, optional
        Scale factor used to set the initial value and bounds of the gain
        parameter, computed as `1 / gain_scale_factor`.

    Notes
    -----
    The function is explained in :cite:t:`bakker_calibration_2008`.

    """

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        quad: bool = False,
        approximate_tmax: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(cutoff=cutoff, use_block=use_block, **kwargs)
        self.quad = quad
        self.approximate_tmax = approximate_tmax

    @property
    def nparam(self) -> int:
        return 4

    def get_init_parameters(self, name: str) -> DataFrame:
        if self.up:
            initial_A, pmin_A, pmax_A = (
                1.0 / self.gain_scale_factor,
                0.0,
                1e2 / self.gain_scale_factor,
            )
        elif self.up is False:
            initial_A, pmin_A, pmax_A = (
                -1.0 / self.gain_scale_factor,
                -1e2 / self.gain_scale_factor,
                0.0,
            )
        else:
            initial_A, pmin_A, pmax_A = 1.0 / self.gain_scale_factor, np.nan, np.nan

        parameters = DataFrame(
            [
                (initial_A, pmin_A, pmax_A, True, name, "uniform"),
                (1.0, -10.0, 10.0, True, name, "uniform"),
                (10.0, 1e-2, 5e3, True, name, "uniform"),
                (10.0, 1e-6, 25.0, True, name, "uniform"),
            ],
            index=[name + "_A", name + "_n", name + "_a", name + "_b"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        return parameters

    def get_tmax_approximation(
        self, p: ArrayLike, cutoff: float | None = None
    ) -> float:
        """Approximate tmax using adaptive cumulative integration in log-time.

        Parameters
        ----------
        p : array_like
            Response function parameters `[A, n, a, b]`.
        cutoff : float, optional
            Fraction of the total step response reached at tmax.

        Returns
        -------
        float
            Approximated tmax in days.
        """
        cutoff = self.cutoff if cutoff is None else cutoff
        if not 0.0 < cutoff < 1.0:
            raise ValueError("Cutoff must be between 0 and 1.")

        # Because Model.get_response_tmax() provides parameters for the stressmodel,
        # not only the response functions
        if len(p) > 4:
            p = p[:4]

        impulse_integral = self._impulse_integral_for_mode(p)
        if not np.isfinite(impulse_integral) or impulse_integral <= 0.0:
            logger.warning(
                "Unable to compute FourParam tmax due to invalid normalization "
                "integral (value=%s). Returning 1.0 day.",
                impulse_integral,
            )
            return 1.0

        u_min = -40.0
        u_max = 20.0
        max_u = 40.0
        frac_max = 0.0

        while frac_max < cutoff and u_max <= max_u:
            u = np.linspace(u_min, u_max, 4000)
            fu = self._integrand_fourparam(u, p)
            cum = np.zeros_like(u)
            cum[1:] = np.cumsum(0.5 * (fu[1:] + fu[:-1]) * np.diff(u))
            frac = cum / impulse_integral
            frac_max = float(frac[-1])
            if frac_max < cutoff:
                u_max += 5.0

        if frac_max < cutoff:
            logger.warning(
                "FourParam tmax search hit maximum limit (%s days) without "
                "reaching cutoff=%s. Returning search limit.",
                np.exp(max_u),
                cutoff,
            )
            return float(np.exp(max_u))

        frac = np.clip(frac, 0.0, 1.0)
        # Conservative mode returns the first grid point where cutoff is reached.
        # This intentionally overestimates tmax versus interpolation.
        idx = int(np.searchsorted(frac, cutoff, side="left"))
        idx = min(max(idx, 0), len(u) - 1)
        return float(np.exp(u[idx]))

    def _impulse_integral(self, p: ArrayLike) -> float:
        """Integral of FourParam impulse over [0, inf)."""
        if self.quad:
            q = quad(self.impulse, 0, np.inf, args=p)[0]
            return float(q)

        _, n, a, b = p
        scale = a * np.sqrt(b)
        arg = 2.0 * np.sqrt(b)
        integral = 2.0 * (scale**n) * kv(n, arg)
        return float(integral)

    @staticmethod
    def _integrand_fourparam(u: ArrayLike, p: ArrayLike) -> ArrayLike:
        """Impulse integrand transformed to log-time (t = exp(u))."""
        _, n, a, b = p
        t = np.exp(u)
        return np.exp(n * u - t / a - (a * b) / t)

    def _f_step(
        self,
        t: float,
        p: ArrayLike,
        cutoff: float,
        impulse_integral: float,
    ) -> float:
        """Objective function for exact tmax root finding."""
        if t <= 0.0:
            return -cutoff

        if self.quad:
            step_value = quad(self.impulse, 0, t, args=p)[0] / impulse_integral
            return step_value - cutoff

        u_upper = np.log(t)
        # Integrate in log-time for numerical stability at very small t.
        if u_upper <= -80.0:
            step_value = 0.0
        else:
            step_value = (
                quad(self._integrand_fourparam, -80.0, u_upper, args=(p,))[0]
                / impulse_integral
            )
        return step_value - cutoff

    def get_tmax(self, p: ArrayLike, cutoff: float | None = None) -> float:
        """Calculate `tmax` using approximation or root finding.

        Parameters
        ----------
        p: array_like
            Response function parameters.
        cutoff: float, optional
            Fraction of the step response used to determine the response cutoff.

        Returns
        -------
        float
            Response time in days corresponding to the selected cutoff.
        """
        cutoff = self.cutoff if cutoff is None else cutoff

        # Because Model.get_response_tmax() provides parameters for the stressmodel,
        # not only the response functions
        if len(p) > 4:
            p = p[:4]

        t0 = self.get_tmax_approximation(p, cutoff)
        if self.approximate_tmax:
            return t0

        impulse_integral = self._impulse_integral(p)
        if not np.isfinite(impulse_integral) or impulse_integral <= 0.0:
            logger.warning(
                "Unable to compute exact FourParam tmax due to invalid normalization "
                "integral (value=%s). Returning approximate tmax.",
                impulse_integral,
            )
            return t0

        lower = 1e-30
        upper = max(t0, lower * 10.0)

        f_upper = self._f_step(upper, p, cutoff, impulse_integral)
        max_upper = float(np.exp(40.0))
        while f_upper < 0.0 and upper < max_upper:
            upper = min(upper * 2.0, max_upper)
            f_upper = self._f_step(upper, p, cutoff, impulse_integral)

        if f_upper < 0.0:
            logger.warning(
                "Could not bracket exact FourParam tmax up to %s days for cutoff=%s. "
                "Returning approximate tmax.",
                max_upper,
                cutoff,
            )
            return t0

        # Scale xtol with response timescale to keep absolute precision meaningful.
        tol = min(max(upper * 1e-9, 1e-15), 1e-3)

        try:
            root, info = brentq(
                f=self._f_step,
                a=lower,
                b=upper,
                xtol=tol,
                maxiter=200,
                args=(p, cutoff, impulse_integral),
                full_output=True,
                disp=False,
            )
        except ValueError as err:
            logger.warning(
                "Exact FourParam tmax root finding failed (%s). Returning approximate "
                "tmax.",
                err,
            )
            return t0

        if info.converged:
            return float(root)

        logger.warning(
            "Root finding for FourParam tmax did not converge, returning "
            "approximate tmax. Brentq RootResults: %s",
            info,
        )
        return t0

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        # Because Model.get_response_tmax() provides parameters for the stressmodel,
        # not only the response functions
        if len(p) > 4:
            p = p[:4]

        impulse_integral = self._impulse_integral(p)

        if self.quad:
            t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
            s = np.zeros_like(t)
            s[0] = quad(self.impulse, 0, dt, args=p)[0]
            for i in range(1, len(t)):
                s[i] = s[i - 1] + quad(self.impulse, t[i - 1], t[i], args=p)[0]
            s = s * (p[0] / impulse_integral)
            return s

        else:
            t1 = -np.sqrt(3 / 5)
            t2 = 0
            t3 = np.sqrt(3 / 5)
            w1 = 5 / 9
            w2 = 8 / 9
            w3 = 5 / 9

            if dt > 0.1:
                step = 0.1  # step size for numerical integration
                tmax = max(self.get_tmax(p=p, cutoff=cutoff), 3 * dt)
                t = np.arange(step, tmax, step)
                s = np.zeros_like(t)

                # for interval [0,dt] :
                s[0] = (step / 2) * (
                    w1 * self.impulse((step / 2) * t1 + (step / 2), p)
                    + w2 * self.impulse((step / 2) * t2 + (step / 2), p)
                    + w3 * self.impulse((step / 2) * t3 + (step / 2), p)
                )

                # for interval [dt,tmax]:
                func = self.impulse(t, p)
                func_half = self.impulse(t[:-1] + step / 2, p)
                s[1:] = s[0] + np.cumsum(
                    step / 6 * (func[:-1] + 4 * func_half + func[1:])
                )
                s = s * (p[0] / impulse_integral)
                return s[int(dt / step - 1) :: int(dt / step)]
            else:
                t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
                s = np.zeros_like(t)

                # for interval [0,dt] Gaussian quadrate:
                s[0] = (dt / 2) * (
                    w1 * self.impulse((dt / 2) * t1 + (dt / 2), p)
                    + w2 * self.impulse((dt / 2) * t2 + (dt / 2), p)
                    + w3 * self.impulse((dt / 2) * t3 + (dt / 2), p)
                )

                # for interval [dt,tmax] Simpson integration:
                func = self.impulse(t, p)
                func_half = self.impulse(t[:-1] + dt / 2, p)
                s[1:] = s[0] + np.cumsum(
                    dt / 6 * (func[:-1] + 4 * func_half + func[1:])
                )
                s = s * (self.gain(p) / impulse_integral)
                return s

    def block_from_impulse(
        self,
        p: ArrayLike,
        dt: float = 1,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        block = super().block_from_impulse(
            p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs
        )
        block = block / np.sum(block) * self.gain(p)
        return block

    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        if method == "discrete":
            t = self.get_t(p=p, dt=dt, cutoff=self.cutoff)
            b = Series(self.block(p=p, dt=dt, cutoff=self.cutoff), index=t)
            return moment(b, order)
        elif method == "exact":
            A, n, a, b = p
            return (
                A
                * (a * a * b) ** (order / 2)
                * kv(order + n, 2 * np.sqrt(b))
                / kv(n, 2 * np.sqrt(b))
            )
        else:
            raise ValueError(f"Invalid method {method}. Choose 'discrete' or 'exact'.")

    @staticmethod
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        _, n, a, b = p
        return (t ** (n - 1)) * np.exp(-t / a - a * b / t)

    def to_dict(self):
        settings = super().to_dict() | {
            "quad": self.quad,
            "approximate_tmax": self.approximate_tmax,
        }
        return settings


class DoubleExponential(RfuncBase):
    """Double Exponential response function with 4 parameters A, alpha, a1 and a2.

    Parameters
    ----------
    cutoff: float, optional
        Fraction of the step response after which the response is truncated.
        Default is 0.999.
    use_block: bool, optional
        Use the block response (rather than the impulse response) to simulate
        the effect of a stress. The block response approximates the stress
        as uniform during a time interval dt. When False, the impulse response
        is used which means that the the entire stress occurs midway the time
        interval dt. The impulse response is generally quicker to compute.

    Attributes
    ----------
    up: bool or None, optional
        Whether a positive stress causes the head to go up (`True`), down
        (`False`), or either direction (`None`).
    gain_scale_factor: float, optional
        Scale factor used to set the initial value and bounds of the gain
        parameter, computed as `1 / gain_scale_factor`.

    """

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(cutoff=cutoff, use_block=use_block, **kwargs)

    @property
    def nparam(self) -> int:
        return 4

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            [
                (
                    1.0 / self.gain_scale_factor
                    if self.up
                    else -1.0 / self.gain_scale_factor
                    if self.up is False
                    else 1.0 / self.gain_scale_factor,
                    0.0
                    if self.up
                    else -1e2 / self.gain_scale_factor
                    if self.up is False
                    else np.nan,
                    1e2 / self.gain_scale_factor
                    if self.up
                    else 0.0
                    if self.up is False
                    else np.nan,
                    True,
                    name,
                    "uniform",
                ),
                (0.1, 1e-2, 0.99, True, name, "uniform"),
                (10.0, 1e-2, 5e3, True, name, "uniform"),
                (10.0, 1e-2, 5e3, True, name, "uniform"),
            ],
            index=[name + "_A", name + "_alpha", name + "_a1", name + "_a2"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: float | None = None) -> float:
        if cutoff is None:
            cutoff = self.cutoff
        if p[2] > p[3]:  # a1 > a2
            return -p[2] * np.log(1 - cutoff)
        else:  # a1 < a2
            return -p[3] * np.log(1 - cutoff)

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
        s = p[0] * (1 - ((1 - p[1]) * np.exp(-t / p[2]) + p[1] * np.exp(-t / p[3])))
        return s

    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        if method == "discrete":
            t = self.get_t(p=p, dt=dt, cutoff=self.cutoff)
            b = Series(self.block(p=p, dt=dt, cutoff=self.cutoff), index=t)
            return moment(b, order)
        elif method == "exact":
            A, alpha, a1, a2 = p
            return A * factorial(order) * ((1 - alpha) * a1**order + alpha * a2**order)
        else:
            raise ValueError(f"Invalid method {method}. Choose 'discrete' or 'exact'.")

    @staticmethod
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, alpha, a_1, a_2 = p
        return A * (
            (1 - alpha) / a_1 * np.exp(-t / a_1) + alpha / a_2 * np.exp(-t / a_2)
        )


class Kraijenhoff(RfuncBase):
    """The response function of :cite:t:`van_de_leur_study_1958`.

    Parameters
    ----------
    cutoff: float, optional
        Fraction of the step response after which the response is truncated.
        Default is 0.999.
    use_block: bool, optional
        Use the block response (rather than the impulse response) to simulate
        the effect of a stress. The block response approximates the stress
        as uniform during a time interval dt. When False, the impulse response
        is used which means that the the entire stress occurs midway the time
        interval dt. The impulse response is generally quicker to compute.
    n_terms: int, optional
        Number of terms used in the truncated series expansion.

    Attributes
    ----------
    up: bool or None, optional
        Whether a positive stress causes the head to go up (`True`), down
        (`False`), or either direction (`None`).
    gain_scale_factor: float, optional
        Scale factor used to set the initial value and bounds of the gain
        parameter, computed as `1 / gain_scale_factor`.

    Notes
    -----
    The Kraijenhoff van de Leur function is explained in
    :cite:t:`van_de_leur_study_1958`.

    The function describes the response of a domain between two drainage channels.
    The function gives the same outcome as equation 133.15 in
    :cite:t:`bruggeman_analytical_1999`. This is the response that is actually
    calculated with this function.

    The response function has three parameters A, a and b:

    - A is the gain (scaled),
    - a is the reservoir coefficient (j in :cite:t:`van_de_leur_study_1958`),
    - b is the location in the domain with the origin in the middle. This means that
      b=0 is in the middle and b=1/2 is at the drainage channel. At b=1/4 the
      response function is most similar to the exponential response function.

    """

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        n_terms: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(cutoff=cutoff, use_block=use_block, **kwargs)
        self.n_terms = n_terms

    @property
    def nparam(self) -> int:
        return 3

    def get_init_parameters(self, name: str) -> DataFrame:
        if self.up:
            initial_A, pmin_A, pmax_A = (
                1.0 / self.gain_scale_factor,
                1e-5,
                1e2 / self.gain_scale_factor,
            )
        elif self.up is False:
            initial_A, pmin_A, pmax_A = (
                -1.0 / self.gain_scale_factor,
                -1e2 / self.gain_scale_factor,
                -1e-5,
            )
        else:
            initial_A, pmin_A, pmax_A = 1.0 / self.gain_scale_factor, np.nan, np.nan

        parameters = DataFrame(
            [
                (initial_A, pmin_A, pmax_A, True, name, "uniform"),
                (1e2, 1e-2, 1e4, True, name, "uniform"),
                (0.0, 0.0, 0.499999, True, name, "uniform"),
            ],
            index=[name + "_A", name + "_a", name + "_b"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: float | None = None) -> float:
        if cutoff is None:
            cutoff = self.cutoff
        return -p[1] * np.log(1 - cutoff)

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
        h = 0
        for n in range(self.n_terms):
            h += (
                (-1) ** n
                / (2 * n + 1) ** 3
                * np.cos((2 * n + 1) * pi * p[2])
                * np.exp(-((2 * n + 1) ** 2) * t / p[1])
            )
        s = p[0] * (1 - (8 / (pi**3 * (1 / 4 - p[2] ** 2)) * h))
        return s

    def block_from_impulse(
        self,
        p: ArrayLike,
        dt: float = 1,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
        t_mid = t - (0.5 * dt)  # compute times at the middle of the interval
        return self.impulse(t=t_mid, p=p, n_terms=self.n_terms) * dt

    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        if method == "discrete":
            t = self.get_t(p=p, dt=dt, cutoff=self.cutoff)
            b = Series(self.block(p=p, dt=dt, cutoff=self.cutoff), index=t)
            return moment(b, order)
        elif method == "exact":
            A, a, b = p
            const = A * 8 * gamma(order + 1) * a**order / (pi**3 * (0.25 - b**2))
            n = np.arange(self.n_terms)
            m = 2 * n + 1
            S = np.sum((-1) ** n * np.cos(m * pi * b) / (m ** (2 * order + 3)))
            return const * S
        else:
            raise ValueError(f"Invalid method {method}. Choose 'discrete' or 'exact'.")

    @staticmethod
    def impulse(t: ArrayLike, p: ArrayLike, n_terms: int = 10) -> ArrayLike:
        A, a, b = p
        leading_term = A * 8 / (pi**3 * ((1 / 4) - b**2))

        h = 0.0
        for n in range(n_terms):
            k = 2 * n + 1
            oscillation_term = (-1) ** n / (a * k) * np.cos(k * pi * b)
            decay_term = np.exp(-(k**2 * t) / a)
            h += oscillation_term * decay_term

        return leading_term * h

    def to_dict(self):
        settings = super().to_dict() | {"n_terms": self.n_terms}
        return settings


class Spline(RfuncBase):
    """Spline response function with parameters: A and a factor for every t.

    Parameters
    ----------
    cutoff: float, optional
        Fraction of the step response after which the response is truncated. This
        setting has no effect for this response function.
    use_block: bool
        Must be True as Spline does not have an impulse response function
    kind: str, optional
        Interpolation kind passed to :func:`scipy.interpolate.interp1d`.
        Common choices are `"quadratic"` and `"cubic"`.
    t: list[int], optional
        Times at which the response function is defined. Defaults to
        `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]`.

    Attributes
    ----------
    up: bool or None, optional
        Whether a positive stress causes the head to go up (`True`), down
        (`False`), or either direction (`None`).
    gain_scale_factor: float, optional
        Scale factor used to set the initial value and bounds of the gain
        parameter, computed as `1 / gain_scale_factor`.


    Notes
    -----
    The spline response function generates a response function from factors at t = 1,
    2, 4, 8, 16, 32, 64, 128, 256, 512 and 1024 days by default. This response
    function is more data-driven than existing response functions and has no physical
    background. Therefore, it can primarily be used to compare to other more physical
    response functions, that probably describe the groundwater system better.
    """

    def __init__(
        self,
        cutoff: float = 0.999,
        use_block: bool = True,
        kind: Literal["quadratic", "cubic"] = "quadratic",
        t: list[int] | None = None,
        **kwargs,
    ) -> None:
        if not use_block:
            logger.error(
                "The Spline response function does not have an impulse response function, "
                "so use_block cannot be False. Please set use_block to True."
            )

        super().__init__(cutoff=cutoff, use_block=True, **kwargs)
        self.kind = kind
        self.t = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] if t is None else t

    @property
    def nparam(self) -> int:
        return len(self.t) + 1

    def get_init_parameters(self, name: str) -> DataFrame:
        if self.up:
            initial_A, pmin_A, pmax_A = (
                1.0 / self.gain_scale_factor,
                1e-5,
                1e2 / self.gain_scale_factor,
            )
        elif self.up is False:
            initial_A, pmin_A, pmax_A = (
                -1.0 / self.gain_scale_factor,
                -1e2 / self.gain_scale_factor,
                -1e-5,
            )
        else:
            initial_A, pmin_A, pmax_A = 1 / self.gain_scale_factor, np.nan, np.nan
        parameters = DataFrame(
            ([initial_A, pmin_A, pmax_A, True, name, "uniform"],),
            index=[name + "_A"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        initial = np.linspace(0.0, 1.0, len(self.t) + 1)[1:]
        for i in range(len(self.t)):
            index = name + "_" + str(self.t[i])
            vary = True
            # fix the value of the factor at the last timestep to 1.0
            if i == len(self.t) - 1:
                vary = False
            parameters.loc[index] = (initial[i], 0.0, 1.0, vary, name, "uniform")

        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: float | None = None) -> float:
        return self.t[-1]

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: float | None = None,
        maxtmax: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        f = interp1d(self.t, p[1 : len(self.t) + 1], kind=self.kind)
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, **kwargs)
        s = p[0] * f(t)
        return s

    def moment(
        self,
        p: ArrayLike,
        order: int,
        method: Literal["discrete", "exact"] = "discrete",
        dt: float = 1.0,
    ) -> float:
        if method == "discrete":
            t = self.get_t(p=p, dt=dt, cutoff=self.cutoff)
            s = Series(self.block(p=p, dt=dt, cutoff=self.cutoff), index=t)
            return moment(s, order)
        else:
            raise ValueError(
                f"Invalid method {method}. Only 'discrete' is supported for "
                f"{self._name}."
            )

    @staticmethod
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        """Raise a NotImplementedError because Spline does not define an impulse response."""
        raise NotImplementedError(
            "Spline does not define an impulse response. Use step() or block()."
        )

    def to_dict(self):
        settings = super().to_dict() | {
            "kind": self.kind,
            "t": self.t,
        }
        return settings


@PastasDeprecationWarning(
    version="2.0.0",
    reason=(
        "Please use the pastas-plugins library if you want to keep using this "
        "response function (https://github.com/pastas/pastas/issues/475)."
    ),
)
class Edelman(RfuncBase):
    """Moved to pastas-plugins: `pastas_plugins.responses.Edelman`"""

    pass
