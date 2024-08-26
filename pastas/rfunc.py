# coding=utf-8
"""This module contains all the response functions available in Pastas."""

from logging import getLogger

import numpy as np
from numpy import pi
from pandas import DataFrame
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import (
    erfc,
    erfcinv,
    exp1,
    gamma,
    gammainc,
    gammaincinv,
    k0,
    k1,
    lambertw,
)

from .decorators import latexfun, njit
from .version import check_numba_scipy

try:
    from numba import prange
except ImportError:
    prange = range

# Type Hinting
from typing import Optional, Union

from pastas.decorators import PastasDeprecationWarning
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
    "Edelman",
    "HantushWellModel",
    "Kraijenhoff",
    "Spline",
]


class RfuncBase:
    _name = "RfuncBase"

    def __init__(
        self,
        cutoff: float = 0.999,
        **kwargs,
    ) -> None:
        self.cutoff = cutoff
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

    def update_rfunc_settings(
        self,
        up: Optional[bool] = "nochange",
        gain_scale_factor: Optional[float] = None,
        cutoff: Optional[float] = None,
    ) -> None:
        """Internal method to set the settings of the response function.

        Parameters
        ----------
        up: bool or None, optional
            indicates whether a positive stress will cause the head to go up (True,
            default) or down (False), if None the head can go both ways.
        gain_scale_factor: float, optional
            the scale factor is used to set the initial value and the bounds of the gain
            parameter, computed as 1 / gain_scale_factor.
        cutoff: float, optional
            proportion after which the step function is cut off.

        Notes
        -----
        Only change the settings if values are provided!

        """
        if up != "nochange":
            self.up = up

        if gain_scale_factor is not None:
            if 1e-8 > gain_scale_factor > 0:
                gain_scale_factor = 1e-8  # arbitrary number to prevent division by zero
            elif gain_scale_factor < 0 and up is True:
                gain_scale_factor = gain_scale_factor * -1
            self.gain_scale_factor = gain_scale_factor

        if cutoff is not None:
            self.cutoff = cutoff

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

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
        """Method to get the response time for a certain cutoff.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the model
            parameters.
        cutoff: float, optional
            proportion after which the step function is cut off. default is 0.999.

        Returns
        -------
        tmax: float
            Number of days when 99.9% of the response has effectuated, when the
            cutoff is chosen at 0.999.
        """

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        """Method to return the step function.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the model
            parameters.
        dt: float
            timestep as a multiple of one day.
        cutoff: float, optional
            proportion after which the step function is cut off. default is 0.999.
        maxtmax: int, optional
            Maximum timestep to compute the block response for.

        Returns
        -------
        s: array_like
            Array with the step response.
        """
        return

    def block(self, p: ArrayLike, dt: float = 1.0, **kwargs) -> ArrayLike:
        """Method to return the block function.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the model
            parameters.
        dt: float
            timestep as a multiple of one day.
        kwargs: dict
            kwargs are passed onto self.step()

        Returns
        -------
        s: array_like
            Array with the block response.
        """
        s = self.step(p=p, dt=dt, **kwargs)
        return np.append(s[0], np.subtract(s[1:], s[:-1]))

    @staticmethod
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

        Notes
        -----
        The impulse response function for each response function class can be viewed on
        the Documentation website or using `latexify` by running the following code in a
        Jupyter notebook environment::

            ps.RfuncName.impulse

        Only used for internal consistency checks
        """

    def get_t(
        self,
        p: ArrayLike,
        dt: float,
        cutoff: float,
        maxtmax: Optional[int] = None,
        warn: bool = True,
    ) -> ArrayLike:
        """Internal method to determine the times at which to evaluate the step
        response, from t=0.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the model
            parameters.
        dt: float
            timestep as a multiple of one day.
        cutoff: float
            proportion after which the step function is cut off.
        maxtmax: float, optional
            The maximum time of the response, usually set to the simulation length.
        warn : bool, optional
            only used for HantushWellModel, whether to warn when r is set to 1.0
            for calculations.

        Returns
        -------
        t: array_like
            Array with the times.
        """
        if isinstance(dt, np.ndarray):
            return dt
        else:
            if isinstance(self, HantushWellModel):
                tmax = self.get_tmax(p, cutoff, warn=warn)
            else:
                tmax = self.get_tmax(p, cutoff)
            if maxtmax is not None:
                tmax = min(tmax, maxtmax)
            tmax = max(tmax, 3 * dt)
            return np.arange(dt, tmax, dt)

    def to_dict(self):
        """Method to export the response function to a dictionary.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the rfunc object.

        Notes
        -----
        The exported dictionary should exactly match the input arguments of __init__.

        """
        data = {
            "class": self._name,
            "up": self.up,
            "gain_scale_factor": self.gain_scale_factor,
            "cutoff": self.cutoff,
        }
        return data


class Gamma(RfuncBase):
    """Gamma response function with 3 parameters A, a, and n.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up (True,
        default) or down (False), if None the head can go both ways.
    gain_scale_factor: float, optional
        mean value of the stress, used to set the initial value such that the final
        step times the mean stress equals 1.
    cutoff: float, optional
        proportion after which the step function is cut off.

    Notes
    -----
    The impulse response function for this class can be viewed on the
    Documentation website or using `latexify` by running the following code in a
    Jupyter notebook environment::

        ps.Gamma.impulse

    The Gamma function is equal to the Exponential function when n=1.
    """

    _name = "Gamma"

    def __init__(
        self,
        cutoff: float = 0.999,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.nparam = 3

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.up:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                1e-5,
                100 / self.gain_scale_factor,
                True,
                name,
                "uniform",
            )
        elif self.up is False:
            parameters.loc[name + "_A"] = (
                -1 / self.gain_scale_factor,
                -100 / self.gain_scale_factor,
                -1e-5,
                True,
                name,
                "uniform",
            )
        else:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                np.nan,
                np.nan,
                True,
                name,
                "uniform",
            )

        # if n is too small, the length of response function is close to zero
        parameters.loc[name + "_n"] = (1, 0.01, 100, True, name, "uniform")
        parameters.loc[name + "_a"] = (10, 0.01, 1e4, True, name, "uniform")
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
        if cutoff is None:
            cutoff = self.cutoff
        return gammaincinv(p[1], cutoff) * p[2]

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
        s = p[0] * gammainc(p[1], t / p[2])
        return s

    @staticmethod
    @latexfun(identifiers={"impulse": "theta", "gamma": "Gamma"})
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, n, a = p
        return A * t ** (n - 1) * np.exp(-t / a) / (a**n * gamma(n))


class Exponential(RfuncBase):
    """Exponential response function with 2 parameters: A and a.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up (True,
        default) or down (False), if None the head can go both ways.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    cutoff: float, optional
        proportion after which the step function is cut off.

    Notes
    -----
    The impulse response function for this class can be viewed on the Documentation
    website or using `latexify` by running the following code in a Jupyter notebook
    environment::

        ps.Exponential.impulse

    """

    _name = "Exponential"

    def __init__(
        self,
        cutoff: float = 0.999,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.nparam = 2

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.up:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                1e-5,
                100 / self.gain_scale_factor,
                True,
                name,
                "uniform",
            )
        elif self.up is False:
            parameters.loc[name + "_A"] = (
                -1 / self.gain_scale_factor,
                -100 / self.gain_scale_factor,
                -1e-5,
                True,
                name,
                "uniform",
            )
        else:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                np.nan,
                np.nan,
                True,
                name,
                "uniform",
            )

        parameters.loc[name + "_a"] = (10, 0.01, 1000, True, name, "uniform")
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
        cutoff: Optional[float] = None,
        maxtmax: Optional[float] = None,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
        s = p[0] * (1.0 - np.exp(-t / p[1]))
        return s

    @staticmethod
    @latexfun(identifiers={"impulse": "theta"})
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, a = p
        return A / a * np.exp(-t / a)


class HantushWellModel(RfuncBase):
    """An implementation of the Hantush well function for multiple pumping wells.

    Parameters
    ----------
    up: bool, optional
        indicates whether a positive stress will cause the head to go up (True,
        default) or down (False).
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    cutoff: float, optional
        proportion after which the step function is cut off.
    use_numba: bool, optional
        Use the method 'numba_step' to compute the step_response.
    quad: bool, optional
        Use the method 'numba_quad' to compute the step_response.

    Notes
    -----
    The impulse response function for this class can be viewed on the Documentation
    website or using `latexify` by running the following code in a Jupyter notebook
    environment::

        ps.HantushWellModel.impulse

    where r is the distance from the pumping well to the observation point and must
    be specified. A, a, and b are parameters, which are slightly different from the
    Hantush response function. The gain is defined as:

    :math:`\\text{gain} = A K_0 \\left( 2r \\sqrt(b) \\right)`

    The implementation used here is explained in :cite:t:`veling_hantush_2010`.
    """

    _name = "HantushWellModel"

    def __init__(
        self,
        cutoff: float = 0.999,
        use_numba: bool = False,
        quad: bool = False,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.distances = None
        self.nparam = 3
        self.use_numba = use_numba  # requires numba_scipy for real speedups
        self.quad = quad  # if quad=True, implicitly uses numba
        # check numba and numba_scipy installation
        if self.quad or self.use_numba:
            # turn off use_numba if numba_scipy is not available
            # or there is a version conflict
            if self.use_numba:
                self.use_numba = check_numba_scipy()

    def set_distances(self, distances) -> None:
        self.distances = distances

    def get_init_parameters(self, name: str) -> DataFrame:
        if self.distances is None:
            raise (
                Exception(
                    "distances is None. Set using method set_distances() or use "
                    "Hantush."
                )
            )
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.up:
            # divide by k0(2) to get same initial value as ps.Hantush
            parameters.loc[name + "_A"] = (
                1 / (self.gain_scale_factor * k0(2)),
                0,
                np.nan,
                True,
                name,
                "uniform",
            )
        elif self.up is False:
            # divide by k0(2) to get same initial value as ps.Hantush
            parameters.loc[name + "_A"] = (
                -1 / (self.gain_scale_factor * k0(2)),
                np.nan,
                0,
                True,
                name,
                "uniform",
            )
        else:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                np.nan,
                np.nan,
                True,
                name,
                "uniform",
            )
        parameters.loc[name + "_a"] = (100, 1e-3, 1e4, True, name, "uniform")
        # set initial and bounds for b taking into account distances
        # note log transform to avoid tiny values for b
        binit = np.log(1.0 / np.mean(self.distances) ** 2)
        bmin = np.log(1e-6 / np.max(self.distances) ** 2)
        bmax = np.log(25.0 / np.min(self.distances) ** 2)
        parameters.loc[name + "_b"] = (binit, bmin, bmax, True, name, "uniform")
        return parameters

    @staticmethod
    def _get_distance_from_params(p: ArrayLike, warn: bool = True) -> float:
        if len(p) == 3:
            r = 1.0
            if warn:
                logger.info("No distance passed to HantushWellModel, assuming r=1.0.")
        else:
            r = p[3]
        return r

    def get_tmax(
        self, p: ArrayLike, cutoff: Optional[float] = None, warn: bool = True
    ) -> float:
        r = self._get_distance_from_params(p, warn=warn)
        # approximate formula for tmax
        if cutoff is None:
            cutoff = self.cutoff
        a, b = p[1:3]
        rho = 2 * r * np.exp(b / 2)
        k0rho = k0(rho)
        if k0rho == 0.0:
            return 50 * 365.0  # 50 years, need to set some tmax if k0rho==0.0
        else:
            return lambertw(1 / ((1 - cutoff) * k0rho)).real * a

    def gain(self, p: ArrayLike, r: Optional[float] = None) -> float:
        if r is None:
            r = self._get_distance_from_params(p)
        rho = 2 * r * np.exp(p[2] / 2)
        return p[0] * k0(rho)

    @staticmethod
    @njit
    def _integrand_hantush(y: float, b: float) -> float:
        return np.exp(-y - (b / y)) / y

    @staticmethod
    @njit(parallel=True)
    def numba_step(A: float, a: float, b: float, r: float, t: ArrayLike) -> ArrayLike:
        rho = 2 * r * np.exp(b / 2)
        rhosq = rho**2
        k0rho = k0(rho)
        tau = t / a
        w = (exp1(rho) - k0rho) / (exp1(rho) - exp1(rho / 2))
        F = np.zeros((tau.size,), dtype=np.float64)
        for i in prange(tau.size):
            tau_i = tau[i]
            if tau_i < rho / 2:
                F[i] = w * exp1(rhosq / (4 * tau_i)) - (w - 1) * exp1(
                    tau_i + rhosq / (4 * tau_i)
                )
            elif tau_i >= rho / 2:
                F[i] = (
                    2 * k0rho
                    - w * exp1(tau_i)
                    + (w - 1) * exp1(tau_i + rhosq / (4 * tau_i))
                )
        return A * F / 2

    @staticmethod
    def numpy_step(A: float, a: float, b: float, r: float, t: ArrayLike) -> ArrayLike:
        rho = 2 * r * np.exp(b / 2)
        rhosq = rho**2
        k0rho = k0(rho)
        tau = t / a
        tau1 = tau[tau < rho / 2]
        tau2 = tau[tau >= rho / 2]
        w = (exp1(rho) - k0rho) / (exp1(rho) - exp1(rho / 2))
        F = np.zeros_like(tau)
        F[tau < rho / 2] = w * exp1(rhosq / (4 * tau1)) - (w - 1) * exp1(
            tau1 + rhosq / (4 * tau1)
        )
        F[tau >= rho / 2] = (
            2 * k0rho - w * exp1(tau2) + (w - 1) * exp1(tau2 + rhosq / (4 * tau2))
        )
        return A * F / 2

    def quad_step(
        self, A: float, a: float, b: float, r: float, t: ArrayLike
    ) -> ArrayLike:
        F = np.zeros_like(t)
        brsq = np.exp(b) * r**2
        u = a * brsq / t
        for i in range(0, len(t)):
            F[i] = quad(self._integrand_hantush, u[i], np.inf, args=(brsq,))[0]
        return F * A / 2

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
        warn: bool = True,
    ) -> ArrayLike:
        A, a, b = p[:3]
        r = self._get_distance_from_params(p, warn=warn)
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax, warn=warn)

        if self.quad:
            return self.quad_step(A, a, b, r, t)
        else:
            # if numba_scipy is available and param a >= ~30, numba is faster
            if a >= 30.0 and self.use_numba:
                return self.numba_step(A, a, b, r, t)
            else:  # otherwise numpy is faster
                return self.numpy_step(A, a, b, r, t)

    @staticmethod
    def variance_gain(
        A: float,
        b: float,
        var_A: float,
        var_b: float,
        cov_Ab: float,
        r: float = 1.0,
    ) -> Union[float, ArrayLike]:
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
            variance of parameter A, can be obtained from the diagonal of the
            covariance matrix (e.g. ml.solver.pcov).
        cov_Ab : float
            covariance between A and b, can be obtained from the covariance matrix (
            e.g. ml.solver.pcov).
        r : float or array_like, optional
            distance(s) between observation well and stress(es), default value is 1.0.

        Returns
        -------
        var_gain : float or array_like
            variance of the gain calculated based on propagation of uncertainty of
            parameters A and b.

        See Also
        --------
        ps.WellModel.variance_gain
        """
        var_gain = (
            (k0(2 * r * np.exp(b / 2))) ** 2 * var_A
            + (A * r * k1(2 * r * np.exp(b / 2))) ** 2 * np.exp(b) * var_b
            - 2
            * A
            * r
            * k0(2 * r * np.exp(b / 2))
            * k1(2 * r * np.exp(b / 2))
            * np.exp(b / 2)
            * cov_Ab
        )
        return var_gain

    def to_dict(self):
        """Method to export the response function to a dictionary.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the rfunc object.

        Notes
        -----
        The exported dictionary should exactly match the input arguments of __init__.

        """
        data = {
            "class": self._name,
            "up": self.up,
            "gain_scale_factor": self.gain_scale_factor,
            "cutoff": self.cutoff,
            "use_numba": self.use_numba,
            "quad": self.quad,
        }
        return data


class Hantush(RfuncBase):
    """The Hantush well function, using the standard A, a, b parameters.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up (True,
        default) or down (False), if None the head can go both ways.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    cutoff: float, optional
        proportion after which the step function is cut off.
    use_numba: bool, optional
        Use the method 'numba_step' to compute the step_response.
    quad: bool, optional
        Use the method 'numba_quad' to compute the step_response.

    Notes
    -----
    Notes
    -----
    The impulse response function for this class can be viewed on the Documentation
    website or using `latexify` by running the following code in a Jupyter notebook
    environment::

        ps.Hantush.impulse

    The implementation used here is explained in :cite:t:`veling_hantush_2010`.

    """

    _name = "Hantush"

    def __init__(
        self,
        cutoff: float = 0.999,
        use_numba: bool = False,
        quad: bool = False,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.nparam = 3
        self.use_numba = use_numba
        self.quad = quad
        # check numba and numba_scipy installation
        if self.quad or self.use_numba:
            # turn off use_numba if numba_scipy is not available
            # or there is a version conflict
            if self.use_numba:
                self.use_numba = check_numba_scipy()

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.up:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                0,
                np.nan,
                True,
                name,
                "uniform",
            )
        elif self.up is False:
            parameters.loc[name + "_A"] = (
                -1 / self.gain_scale_factor,
                np.nan,
                0,
                True,
                name,
                "uniform",
            )
        else:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                np.nan,
                np.nan,
                True,
                name,
                "uniform",
            )
        parameters.loc[name + "_a"] = (100, 1e-3, 1e4, True, name, "uniform")
        parameters.loc[name + "_b"] = (1, 1e-6, 25, True, name, "uniform")
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
        # approximate formula for tmax
        if cutoff is None:
            cutoff = self.cutoff
        a, b = p[1:]
        rho = 2 * np.sqrt(b)
        return lambertw(1 / ((1 - cutoff) * k0(rho))).real * a

    @staticmethod
    def gain(p: ArrayLike) -> float:
        return p[0]

    @staticmethod
    @njit
    def _integrand_hantush(y: float, b: float) -> float:
        return np.exp(-y - (b / y)) / y

    @staticmethod
    @njit(parallel=True)
    def numba_step(A: float, a: float, b: float, t: ArrayLike) -> ArrayLike:
        rho = 2 * np.sqrt(b)
        rhosq = rho**2
        k0rho = k0(rho)
        tau = t / a
        w = (exp1(rho) - k0rho) / (exp1(rho) - exp1(rho / 2))
        F = np.zeros((tau.size,), dtype=np.float64)
        for i in prange(tau.size):
            tau_i = tau[i]
            if tau_i < rho / 2:
                F[i] = w * exp1(rhosq / (4 * tau_i)) - (w - 1) * exp1(
                    tau_i + rhosq / (4 * tau_i)
                )
            elif tau_i >= rho / 2:
                F[i] = (
                    2 * k0rho
                    - w * exp1(tau_i)
                    + (w - 1) * exp1(tau_i + rhosq / (4 * tau_i))
                )
        return A * F / (2 * k0rho)

    @staticmethod
    def numpy_step(A: float, a: float, b: float, t: ArrayLike) -> ArrayLike:
        rho = 2 * np.sqrt(b)
        rhosq = rho**2
        k0rho = k0(rho)
        tau = t / a
        tau_mask = tau < rho / 2
        tau1 = tau[tau_mask]
        tau2 = tau[~tau_mask]
        w = (exp1(rho) - k0rho) / (exp1(rho) - exp1(rho / 2))
        F = np.zeros_like(tau)
        F[tau_mask] = w * exp1(rhosq / (4 * tau1)) - (w - 1) * exp1(
            tau1 + rhosq / (4 * tau1)
        )
        F[~tau_mask] = (
            2 * k0rho - w * exp1(tau2) + (w - 1) * exp1(tau2 + rhosq / (4 * tau2))
        )
        return A * F / (2 * k0rho)

    def quad_step(self, A: float, a: float, b: float, t: ArrayLike) -> ArrayLike:
        F = np.zeros_like(t)
        u = a * b / t
        for i in range(0, len(t)):
            F[i] = quad(self._integrand_hantush, u[i], np.inf, args=(b,))[0]
        return F * A / (2 * k0(2 * np.sqrt(b)))

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        A, a, b = p
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)

        if self.quad:
            return self.quad_step(A, a, b, t)
        else:
            # if numba_scipy is available and param a >= ~30, numba is faster
            if a >= 30.0 and self.use_numba:
                return self.numba_step(A, a, b, t)
            else:  # otherwise numpy is faster
                return self.numpy_step(A, a, b, t)

    @staticmethod
    @latexfun(identifiers={"impulse": "theta", "k0": "K_0"})
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, a, b = p
        return A / (2 * t * k0(2 * np.sqrt(b))) * np.exp(-t / a - a * b / t)

    def to_dict(self):
        """Method to export the response function to a dictionary.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the rfunc object.

        Notes
        -----
        The exported dictionary should exactly match the input arguments of __init__.

        """
        data = {
            "class": self._name,
            "up": self.up,
            "gain_scale_factor": self.gain_scale_factor,
            "cutoff": self.cutoff,
            "use_numba": self.use_numba,
            "quad": self.quad,
        }
        return data


class Polder(RfuncBase):
    """The Polder function, using the standard A, a, b parameters.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up (True,
        default) or down (False), if None the head can go both ways.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    cutoff: float, optional
        proportion after which the step function is cut off.

    Notes
    -----
    The impulse response function for this class can be viewed on the Documentation
    website or using `latexify` by running the following code in a Jupyter notebook
    environment::

        ps.Polder.impulse

    The function is explained in Eq. 123.32 in:cite:t:`bruggeman_analytical_1999`.

    """

    _name = "Polder"

    def __init__(
        self,
        cutoff: float = 0.999,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.nparam = 3

    def get_init_parameters(self, name) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        parameters.loc[name + "_A"] = (1, 0, 2, True, name, "uniform")
        parameters.loc[name + "_a"] = (10, 0.01, 1000, True, name, "uniform")
        parameters.loc[name + "_b"] = (1, 1e-6, 25, True, name, "uniform")
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
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
        if not self.up:
            g = -g
        return g

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
        A, a, b = p
        s = A * self.polder_function(np.sqrt(b), np.sqrt(t / a))
        # / np.exp(-2 * np.sqrt(b))
        if not self.up:
            s = -s
        return s

    @staticmethod
    @latexfun(identifiers={"impulse": "theta"})
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, a, b = p
        return A * np.sqrt(a * b / pi) * t ** (-1.5) * np.exp(-t / a - a * b / t)

    @staticmethod
    @latexfun(use_raw_function_name=True)
    def polder_function(x: float, y: float) -> float:
        return 0.5 * np.exp(2 * x) * erfc(x / y + y) + 0.5 * np.exp(-2 * x) * erfc(
            x / y - y
        )


class One(RfuncBase):
    """Instant response with no lag and one parameter d.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up (True) or
        down (False), if None (default) the head can go both ways.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    cutoff: float, optional
        proportion after which the step function is cut off. Has no influence for
        this response function.
    """

    _name = "One"

    def __init__(
        self,
        cutoff: float = 0.999,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.nparam = 1

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.up:
            parameters.loc[name + "_d"] = (
                self.gain_scale_factor,
                0,
                np.nan,
                True,
                name,
                "uniform",
            )
        elif self.up is False:
            parameters.loc[name + "_d"] = (
                -self.gain_scale_factor,
                np.nan,
                0,
                True,
                name,
                "uniform",
            )
        else:
            parameters.loc[name + "_d"] = (
                self.gain_scale_factor,
                np.nan,
                np.nan,
                True,
                name,
                "uniform",
            )
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
        return 0.0

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        if isinstance(dt, np.ndarray):
            return p[0] * np.ones(len(dt))
        else:
            return p[0] * np.ones(1)

    def block(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        return p[0] * np.ones(1)


class FourParam(RfuncBase):
    """Four Parameter response function with 4 parameters A, a, b, and n.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up (True,
        default) or down (False), if None the head can go both ways.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    cutoff: float, optional
        proportion after which the step function is cut off.
    quad: bool, optional
        If true, use the 'quad' method from scipy.integrate to integrate the impulse
        response function. This may be more accurate but increases computation times.

    Notes
    -----
    The impulse response function for this class can be viewed on the Documentation
    website or using `latexify` by running the following code in a Jupyter notebook
    environment::

        ps.FourParam.impulse

    """

    _name = "FourParam"

    def __init__(
        self,
        cutoff: float = 0.999,
        quad: bool = False,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.nparam = 4
        self.quad = quad

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.up:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                0,
                100 / self.gain_scale_factor,
                True,
                name,
                "uniform",
            )
        elif self.up is False:
            parameters.loc[name + "_A"] = (
                -1 / self.gain_scale_factor,
                -100 / self.gain_scale_factor,
                0,
                True,
                name,
                "uniform",
            )
        else:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                np.nan,
                np.nan,
                True,
                name,
                "uniform",
            )

        parameters.loc[name + "_n"] = (1, -10, 10, True, name, "uniform")
        parameters.loc[name + "_a"] = (10, 0.01, 5000, True, name, "uniform")
        parameters.loc[name + "_b"] = (10, 1e-6, 25, True, name, "uniform")
        return parameters

    @staticmethod
    @latexfun(identifiers={"impulse": "theta"})
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        _, n, a, b = p
        return (t ** (n - 1)) * np.exp(-t / a - a * b / t)

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
        if cutoff is None:
            cutoff = self.cutoff

        # Because Model.get_response_tmax() provides parameters for the stressmodel,
        # not only the response functions
        if len(p) > 4:
            p = p[:4]

        if self.quad:
            x = np.arange(1, 10000, 1)
            y = np.zeros_like(x)
            func = self.impulse(x, p)
            func_half = self.impulse(x[:-1] + 1 / 2, p)
            y[1:] = y[0] + np.cumsum(1 / 6 * (func[:-1] + 4 * func_half + func[1:]))
            y = y / quad(self.impulse, 0, np.inf, args=p)[0]
            return np.searchsorted(y, cutoff)

        else:
            t1 = -np.sqrt(3 / 5)
            t2 = 0
            t3 = np.sqrt(3 / 5)
            w1 = 5 / 9
            w2 = 8 / 9
            w3 = 5 / 9

            x = np.arange(1, 10000, 1)
            y = np.zeros_like(x)
            func = self.impulse(x, p)
            func_half = self.impulse(x[:-1] + 1 / 2, p)
            y[0] = 0.5 * (
                w1 * self.impulse(0.5 * t1 + 0.5, p)
                + w2 * self.impulse(0.5 * t2 + 0.5, p)
                + w3 * self.impulse(0.5 * t3 + 0.5, p)
            )
            y[1:] = y[0] + np.cumsum(1 / 6 * (func[:-1] + 4 * func_half + func[1:]))
            y = y / quad(self.impulse, 0, np.inf, args=p)[0]
            return np.searchsorted(y, cutoff)

    @staticmethod
    def gain(p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        # Because Model.get_response_tmax() provides parameters for the stressmodel,
        # not only the response functions
        if len(p) > 4:
            p = p[:4]

        if self.quad:
            t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
            s = np.zeros_like(t)
            s[0] = quad(self.impulse, 0, dt, args=p)[0]
            for i in range(1, len(t)):
                s[i] = s[i - 1] + quad(self.impulse, t[i - 1], t[i], args=p)[0]
            s = s * (p[0] / (quad(self.impulse, 0, np.inf, args=p))[0])
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
                s = s * (p[0] / quad(self.impulse, 0, np.inf, args=p)[0])
                return s[int(dt / step - 1) :: int(dt / step)]
            else:
                t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
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
                s = s * (p[0] / quad(self.impulse, 0, np.inf, args=p)[0])
                return s

    def to_dict(self):
        """Method to export the response function to a dictionary.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the rfunc object.

        Notes
        -----
        The exported dictionary should exactly match the input arguments of __init__.

        """
        data = {
            "class": self._name,
            "up": self.up,
            "gain_scale_factor": self.gain_scale_factor,
            "cutoff": self.cutoff,
            "quad": self.quad,
        }
        return data


class DoubleExponential(RfuncBase):
    """Double Exponential response function with 4 parameters A, alpha, a1 and a2.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up (True,
        default) or down (False), if None the head can go both ways.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    cutoff: float, optional
        proportion after which the step function is cut off.

    Notes
    -----
    The impulse response function for this class can be viewed on the Documentation
    website or using `latexify` by running the following code in a Jupyter notebook
    environment::

        ps.DoubleExponential.impulse

    """

    _name = "DoubleExponential"

    def __init__(
        self,
        cutoff: float = 0.999,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.nparam = 4

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.up:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                0,
                100 / self.gain_scale_factor,
                True,
                name,
                "uniform",
            )
        elif self.up is False:
            parameters.loc[name + "_A"] = (
                -1 / self.gain_scale_factor,
                -100 / self.gain_scale_factor,
                0,
                True,
                name,
                "uniform",
            )
        else:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                np.nan,
                np.nan,
                True,
                name,
                "uniform",
            )

        parameters.loc[name + "_alpha"] = (0.1, 0.01, 0.99, True, name, "uniform")
        parameters.loc[name + "_a1"] = (10, 0.01, 5000, True, name, "uniform")
        parameters.loc[name + "_a2"] = (10, 0.01, 5000, True, name, "uniform")
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
        if cutoff is None:
            cutoff = self.cutoff
        if p[2] > p[3]:  # a1 > a2
            return -p[2] * np.log(1 - cutoff)
        else:  # a1 < a2
            return -p[3] * np.log(1 - cutoff)

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    @staticmethod
    @latexfun(identifiers={"impulse": "theta"})
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, alpha, a_1, a_2 = p
        return A * (
            (1 - alpha) / a_1 * np.exp(-t / a_1) + alpha / a_2 * np.exp(-t / a_2)
        )

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
        s = p[0] * (1 - ((1 - p[1]) * np.exp(-t / p[2]) + p[1] * np.exp(-t / p[3])))
        return s


@PastasDeprecationWarning(
    remove_version="2.0.0",
    reason=(
        "Please use the pastas-plugins library if you want to keep using this "
        "response function (https://github.com/pastas/pastas/issues/475)."
    ),
)
class Edelman(RfuncBase):
    """The function of Edelman, describing the propagation of an instantaneous
    water level change into an adjacent half-infinite aquifer.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up (True,
        default) or down (False), if None the head can go both ways.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    cutoff: float, optional
        proportion after which the step function is cut off.

    Notes
    -----
    The Edelman function is explained in :cite:t:`edelman_over_1947`.

    The impulse response function for this class can be viewed on the Documentation
    website or using `latexify` by running the following code in a Jupyter notebook
    environment::

        ps.Edelman.impulse

    """

    _name = "Edelman"

    def __init__(
        self,
        cutoff: float = 0.999,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.nparam = 1

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        beta_init = 1.0
        parameters.loc[name + "_beta"] = (beta_init, 0, 1000, True, name, "uniform")
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
        if cutoff is None:
            cutoff = self.cutoff
        return 1.0 / (p[0] * erfcinv(cutoff)) ** 2

    @staticmethod
    def gain(p: ArrayLike) -> float:
        return 1.0

    @staticmethod
    @latexfun(identifiers={"impulse": "theta"})
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        (a,) = p
        return 1 / (np.sqrt(pi) * a * t**1.5) * np.exp(-1 / (a**2 * t))

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
        s = erfc(1 / (p[0] * np.sqrt(t)))
        return s


class Kraijenhoff(RfuncBase):
    """The response function of :cite:t:`van_de_leur_study_1958`.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up (True,
        default) or down (False), if None the head can go both ways.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    cutoff: float, optional
        proportion after which the step function is cut off.
    n_terms: int, optional
        Number of terms.

    Notes
    -----
    The Kraijenhoff van de Leur function is explained in
    :cite:t:`van_de_leur_study_1958`.

    The impulse response function for this class can be viewed on the Documentation
    website or using `latexify` by running the following code in a Jupyter notebook
    environment::

        ps.Kraijenhoff.impulse

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

    _name = "Kraijenhoff"

    def __init__(
        self,
        cutoff: float = 0.999,
        n_terms: int = 10,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.nparam = 3
        self.n_terms = n_terms

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.up:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                1e-5,
                100 / self.gain_scale_factor,
                True,
                name,
                "uniform",
            )
        elif self.up is False:
            parameters.loc[name + "_A"] = (
                -1 / self.gain_scale_factor,
                -100 / self.gain_scale_factor,
                -1e-5,
                True,
                name,
                "uniform",
            )
        else:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                np.nan,
                np.nan,
                True,
                name,
                "uniform",
            )

        parameters.loc[name + "_a"] = (1e2, 0.01, 1e5, True, name, "uniform")
        parameters.loc[name + "_b"] = (0, 0, 0.499999, True, name, "uniform")
        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
        if cutoff is None:
            cutoff = self.cutoff
        return -p[1] * np.log(1 - cutoff)

    @staticmethod
    def gain(p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
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

    @staticmethod
    @latexfun(identifiers={"impulse": "theta"})
    def impulse(t: ArrayLike, p: ArrayLike) -> ArrayLike:
        A, a, b = p
        nterms = 10
        return (
            A
            * 8
            / (pi**3 * ((1 / 4) - b**2))
            * sum(
                (-1) ** n
                / (a * (2 * n + 1))
                * np.cos((2 * n + 1) * pi * b)
                * np.exp(-((2 * n + 1) ** 2 * t) / a)
                for n in range(nterms)
            )
        )

    def to_dict(self):
        """Method to export the response function to a dictionary.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the rfunc object.

        Notes
        -----
        The exported dictionary should exactly match the input arguments of __init__.

        """
        data = {
            "class": self._name,
            "up": self.up,
            "gain_scale_factor": self.gain_scale_factor,
            "cutoff": self.cutoff,
            "n_terms": self.n_terms,
        }
        return data


class Spline(RfuncBase):
    """Spline response function with parameters: A and a factor for every t.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up (True,
        default) or down (False), if None the head can go both ways.
    gain_scale_factor: float, optional
        the scale factor is used to set the initial value and the bounds of the gain
        parameter, computed as 1 / gain_scale_factor.
    cutoff: float, optional
        proportion after which the step function is cut off. default is 0.999. this
        parameter has no influence for this response function.
    kind: string, optional
        see scipy.interpolate.interp1d. Most useful for a smooth response function
        are 'quadratic' and 'cubic'.
    t: list, optional
        times at which the response function is defined.

    Notes
    -----
    The spline response function generates a response function from factors at t = 1,
    2, 4, 8, 16, 32, 64, 128, 256, 512 and 1024 days by default. This response
    function is more data-driven than existing response functions and has no physical
    background. Therefore, it can primarily be used to compare to other more physical
    response functions, that probably describe the groundwater system better.
    """

    _name = "Spline"

    def __init__(
        self,
        cutoff: float = 0.999,
        kind: str = "quadratic",
        t: Optional[list] = None,
        **kwargs,
    ) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.kind = kind
        if t is None:
            t = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.t = t
        self.nparam = len(t) + 1

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.up:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                1e-5,
                100 / self.gain_scale_factor,
                True,
                name,
                "uniform",
            )
        elif self.up is False:
            parameters.loc[name + "_A"] = (
                -1 / self.gain_scale_factor,
                -100 / self.gain_scale_factor,
                -1e-5,
                True,
                name,
                "uniform",
            )
        else:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                np.nan,
                np.nan,
                True,
                name,
                "uniform",
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

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
        return self.t[-1]

    def gain(self, p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        f = interp1d(self.t, p[1 : len(self.t) + 1], kind=self.kind)
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
        s = p[0] * f(t)
        return s

    def to_dict(self):
        """Method to export the response function to a dictionary.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the rfunc object.

        Notes
        -----
        The exported dictionary should exactly match the input arguments of __init__.

        """
        data = {
            "class": self._name,
            "up": self.up,
            "gain_scale_factor": self.gain_scale_factor,
            "cutoff": self.cutoff,
            "kind": self.kind,
            "t": self.t,
        }
        return data
