# coding=utf-8
"""This module contains all the response functions available in Pastas."""

from logging import getLogger

import numpy as np
from pandas import DataFrame
from scipy.integrate import quad
from scipy.special import (erfc, erfcinv, exp1, gamma, gammainc, gammaincinv,
                           k0, k1, lambertw)
from scipy.interpolate import interp1d
from .decorators import njit
from .utils import check_numba, check_numba_scipy

try:
    from numba import prange
except ModuleNotFoundError:
    prange = range

logger = getLogger(__name__)

__all__ = ["Gamma", "Exponential", "Hantush", "Polder", "FourParam",
           "DoubleExponential", "One", "Edelman", "HantushWellModel",
           "Kraijenhoff", "Spline"]


class RfuncBase:
    _name = "RfuncBase"

    def __init__(self, **kwargs):
        self.up = True
        self.meanstress = 1
        self.cutoff = 0.999
        self.kwargs = kwargs

    def _set_init_parameter_settings(self, up=True, meanstress=1,
                                     cutoff=0.999):
        self.up = up
        # Completely arbitrary number to prevent division by zero
        if 1e-8 > meanstress > 0:
            meanstress = 1e-8
        elif meanstress < 0 and up is True:
            meanstress = meanstress * -1
        self.meanstress = meanstress
        self.cutoff = cutoff

    def get_init_parameters(self, name):
        """Get initial parameters and bounds. It is called by the stressmodel.

        Parameters
        ----------
        name :  str
            Name of the stressmodel

        Returns
        -------
        parameters : pandas DataFrame
            The initial parameters and parameter bounds used by the solver
        """
        pass

    def get_tmax(self, p, cutoff=None):
        """Method to get the response time for a certain cutoff.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the
            model parameters.
        cutoff: float, optional
            float between 0 and 1.

        Returns
        -------
        tmax: float
            Number of days when 99.9% of the response has effectuated, when the
            cutoff is chosen at 0.999.
        """
        pass

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        """Method to return the step function.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the
            model parameters.
        dt: float
            timestep as a multiple of of day.
        cutoff: float, optional
            float between 0 and 1.
        maxtmax: int, optional
            Maximum timestep to compute the block response for.

        Returns
        -------
        s: numpy.array
            Array with the step response.
        """
        pass

    def block(self, p, dt=1, cutoff=None, maxtmax=None):
        """Method to return the block function.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the
            model parameters.
        dt: float
            timestep as a multiple of of day.
        cutoff: float, optional
            float between 0 and 1.
        maxtmax: int, optional
            Maximum timestep to compute the block response for.

        Returns
        -------
        s: numpy.array
            Array with the block response.
        """
        s = self.step(p, dt, cutoff, maxtmax)
        return np.append(s[0], np.subtract(s[1:], s[:-1]))

    def impulse(self, t, p):
        """Method to return the impulse response function.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the
            model parameters.
        dt: float
            timestep as a multiple of of day.
        cutoff: float, optional
            float between 0 and 1.
        maxtmax: int, optional
            Maximum timestep to compute the block response for.

        Returns
        -------
        s: numpy.array
            Array with the impulse response.

        Note
        ----
        Only used for internal consistency checks
        """
        pass

    def get_t(self, p, dt, cutoff, maxtmax=None):
        """Internal method to determine the times at which to evaluate the
        step-response, from t=0.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the
            model parameters.
        dt: float
            timestep as a multiple of of day.
        cutoff: float
            float between 0 and 1, that determines which part of the step-
            response is taken into account.
        maxtmax: float, optional
            The maximum time of the response, usually set to the simulation
            length.

        Returns
        -------
        t: numpy.array
            Array with the times
        """
        if isinstance(dt, np.ndarray):
            return dt
        else:
            tmax = self.get_tmax(p, cutoff)
            if maxtmax is not None:
                tmax = min(tmax, maxtmax)
            tmax = max(tmax, 3 * dt)
            return np.arange(dt, tmax, dt)


class Gamma(RfuncBase):
    """Gamma response function with 3 parameters A, a, and n.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False), if None the head can go both ways.
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. default is 0.999.

    Notes
    -----
    The impulse response function is:

    .. math:: \\theta(t) = At^{n-1} e^{-t/a} / (a^n Gamma(n))

    where A, a, and n are parameters. The Gamma function is equal to the
    Exponential function when n=1.
    """
    _name = "Gamma"

    def __init__(self):
        RfuncBase.__init__(self)
        self.nparam = 3

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (1 / self.meanstress, 1e-5,
                                           100 / self.meanstress, True, name)
        elif self.up is False:
            parameters.loc[name + '_A'] = (-1 / self.meanstress,
                                           -100 / self.meanstress,
                                           -1e-5, True, name)
        else:
            parameters.loc[name + '_A'] = (1 / self.meanstress,
                                           np.nan, np.nan, True, name)

        # if n is too small, the length of response function is close to zero
        parameters.loc[name + '_n'] = (1, 0.01, 100, True, name)
        parameters.loc[name + '_a'] = (10, 0.01, 1e4, True, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        return gammaincinv(p[1], cutoff) * p[2]

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        t = self.get_t(p, dt, cutoff, maxtmax)
        s = p[0] * gammainc(p[1], t / p[2])
        return s

    def impulse(self, t, p):
        A, n, a = p
        ir = A * t ** (n - 1) * np.exp(-t / a) / (a ** n * gamma(n))
        return ir


class Exponential(RfuncBase):
    """Exponential response function with 2 parameters: A and a.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False), if None the head can go both ways.
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. default is 0.999.

    Notes
    -----
    The impulse response function is:

    .. math:: \\theta(t) = A / a * e^{-t/a}

    where A and a are parameters.
    """
    _name = "Exponential"

    def __init__(self):
        RfuncBase.__init__(self)
        self.nparam = 2

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (1 / self.meanstress, 1e-5,
                                           100 / self.meanstress, True, name)
        elif self.up is False:
            parameters.loc[name + '_A'] = (-1 / self.meanstress,
                                           -100 / self.meanstress,
                                           -1e-5, True, name)
        else:
            parameters.loc[name + '_A'] = (1 / self.meanstress,
                                           np.nan, np.nan, True, name)

        parameters.loc[name + '_a'] = (10, 0.01, 1000, True, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        return -p[1] * np.log(1 - cutoff)

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        t = self.get_t(p, dt, cutoff, maxtmax)
        s = p[0] * (1.0 - np.exp(-t / p[1]))
        return s

    def impulse(self, t, p):
        A, a = p
        ir = A / a * np.exp(-t / a)
        return ir


class HantushWellModel(RfuncBase):
    """An implementation of the Hantush well function for multiple pumping
    wells.

    Parameters
    ----------
    up: bool, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False)
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. Default is 0.999.

    Notes
    -----
    The impulse response function is:

    .. math:: \\theta(r, t) = \\frac{A}{2t} \\exp(-t/a - abr^2/t)

    where r is the distance from the pumping well to the observation point
    and must be specified. A, a, and b are parameters, which are slightly
    different from the Hantush response function. The gain is defined as:

    :math:`\\text{gain} = A K_0 \\left( 2r \\sqrt(b) \\right)`

    The implementation used here is explained in  :cite:t:`veling_hantush_2010`.

    """
    _name = "HantushWellModel"

    def __init__(self, use_numba=False, quad=False):
        RfuncBase.__init__(self, use_numba=use_numba, quad=quad)
        self.distances = None
        self.nparam = 3
        self.use_numba = use_numba  # requires numba_scipy for real speedups
        self.quad = quad  # if quad=True, implicitly uses numba
        # check numba and numba_scipy installation
        if self.quad or self.use_numba:
            check_numba()
            # turn off use_numba if numba_scipy is not available
            # or there is a version conflict
            if self.use_numba:
                self.use_numba = check_numba_scipy()

    def set_distances(self, distances):
        self.distances = distances

    def get_init_parameters(self, name):
        if self.distances is None:
            raise(Exception('distances is None. Set using method'
                            ' set_distances() or use Hantush.'))
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            # divide by k0(2) to get same initial value as ps.Hantush
            parameters.loc[name + '_A'] = (1 / (self.meanstress * k0(2)),
                                           0, np.nan, True, name)
        elif self.up is False:
            # divide by k0(2) to get same initial value as ps.Hantush
            parameters.loc[name + '_A'] = (-1 / (self.meanstress * k0(2)),
                                           np.nan, 0, True, name)
        else:
            parameters.loc[name + '_A'] = (1 / self.meanstress, np.nan,
                                           np.nan, True, name)
        parameters.loc[name + '_a'] = (100, 1e-3, 1e4, True, name)
        # set initial and bounds for b taking into account distances
        # note log transform to avoid extremely small values for b
        binit = np.log(1.0 / np.mean(self.distances) ** 2)
        bmin = np.log(1e-6 / np.max(self.distances) ** 2)
        bmax = np.log(25. / np.min(self.distances) ** 2)
        parameters.loc[name + '_b'] = (binit, bmin, bmax, True, name)
        return parameters

    @staticmethod
    def _get_distance_from_params(p):
        if len(p) == 3:
            r = 1.0
            logger.info("No distance passed to HantushWellModel, "
                        "assuming r=1.0.")
        else:
            r = p[3]
        return r

    def get_tmax(self, p, cutoff=None):
        r = self._get_distance_from_params(p)
        # approximate formula for tmax
        if cutoff is None:
            cutoff = self.cutoff
        a, b = p[1:3]
        rho = 2 * r * np.exp(b / 2)
        k0rho = k0(rho)
        if k0rho == 0.0:
            return 50 * 365.  # 50 years, need to set some tmax if k0rho==0.0
        else:
            return lambertw(1 / ((1 - cutoff) * k0rho)).real * a

    def gain(self, p, r=None):
        if r is None:
            r = self._get_distance_from_params(p)
        rho = 2 * r * np.exp(p[2] / 2)
        return p[0] * k0(rho)

    @staticmethod
    @njit
    def _integrand_hantush(y, b):
        return np.exp(-y - (b / y)) / y

    @staticmethod
    @njit(parallel=True)
    def numba_step(A, a, b, r, t):
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
                    tau_i + rhosq / (4 * tau_i))
            elif tau_i >= rho / 2:
                F[i] = 2 * k0rho - w * exp1(tau_i) + (w - 1) * exp1(
                    tau_i + rhosq / (4 * tau_i))
        return A * F / 2

    @staticmethod
    def numpy_step(A, a, b, r, t):
        rho = 2 * r * np.exp(b / 2)
        rhosq = rho**2
        k0rho = k0(rho)
        tau = t / a
        tau1 = tau[tau < rho / 2]
        tau2 = tau[tau >= rho / 2]
        w = (exp1(rho) - k0rho) / (exp1(rho) - exp1(rho / 2))
        F = np.zeros_like(tau)
        F[tau < rho / 2] = w * exp1(rhosq / (4 * tau1)) - (w - 1) * exp1(
            tau1 + rhosq / (4 * tau1))
        F[tau >= rho / 2] = 2 * k0rho - w * exp1(tau2) + (w - 1) * exp1(
            tau2 + rhosq / (4 * tau2))
        return A * F / 2

    def quad_step(self, A, a, b, r, t):
        F = np.zeros_like(t)
        brsq = np.exp(b) * r**2
        u = a * brsq / t
        for i in range(0, len(t)):
            F[i] = quad(self._integrand_hantush,
                        u[i], np.inf, args=(brsq,))[0]
        return F * A / 2

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        A, a, b = p[:3]
        r = self._get_distance_from_params(p)
        t = self.get_t(p, dt, cutoff, maxtmax)

        if self.quad:
            return self.quad_step(A, a, b, r, t)
        else:
            # if numba_scipy is available and param a >= ~30, numba is faster
            if a >= 30. and self.use_numba:
                return self.numba_step(A, a, b, r, t)
            else:  # otherwise numpy is faster
                return self.numpy_step(A, a, b, r, t)

    @staticmethod
    def variance_gain(A, b, var_A, var_b, cov_Ab, r=1.0):
        """Calculate variance of the gain from parameters A and b.

        Variance of the gain is calculated based on propagation of
        uncertainty using optimal values, the variances of A and b
        and the covariance between A and b.

        Note
        ----
        Estimated variance can be biased for non-linear functions as it uses
        truncated series expansion.

        Parameters
        ----------
        A : float
            optimal value of parameter A, (e.g. ml.parameters.optimal)
        b : float
            optimal value of parameter b, (e.g. ml.parameters.optimal)
        var_A : float
            variance of parameter A, can be obtained from the diagonal of
            the covariance matrix (e.g. ml.fit.pcov)
        var_b : float
            variance of parameter A, can be obtained from the diagonal of
            the covariance matrix (e.g. ml.fit.pcov)
        cov_Ab : float
            covariance between A and b, can be obtained from the covariance
            matrix (e.g. ml.fit.pcov)
        r : float or np.array, optional
            distance(s) between observation well and stress(es),
            default value is 1.0

        Returns
        -------
        var_gain : float or np.array
            variance of the gain calculated based on propagation of uncertainty
            of parameters A and b.

        See Also
        --------
        ps.WellModel.variance_gain
        """
        var_gain = (
            (k0(2 * r * np.exp(b / 2))) ** 2 * var_A +
            (A * r * k1(2 * r * np.exp(b / 2)))**2 * np.exp(b) * var_b
            - 2 * A * r * k0(2 * r * np.exp(b / 2)) *
            k1(2 * r * np.exp(b / 2)) * np.exp(b / 2) * cov_Ab
        )
        return var_gain


class Hantush(RfuncBase):
    """The Hantush well function, using the standard A, a, b parameters.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False), if None the head can go both ways.
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. default is 0.999.

    Notes
    -----
    The impulse response function is:

    .. math:: \\theta(t) = \\frac{A}{2t \\text{K}_0\\left(2\\sqrt{b} \\right)}
              \\exp(-t/a - ab/t)

    where A, a, and b are parameters.

    The implementation used here is explained in  :cite:t:`veling_hantush_2010`.

    References
    ----------

    .. [veling_2010] Veling, E. J. M., & Maas, C. (2010). Hantush well function
       revisited. Journal of hydrology, 393(3), 381-388.
    """
    _name = "Hantush"

    def __init__(self, use_numba=False, quad=False):
        RfuncBase.__init__(self, use_numba=use_numba, quad=quad)
        self.nparam = 3
        self.use_numba = use_numba
        self.quad = quad
        # check numba and numba_scipy installation
        if self.quad or self.use_numba:
            check_numba()
            # turn off use_numba if numba_scipy is not available
            # or there is a version conflict
            if self.use_numba:
                self.use_numba = check_numba_scipy()

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (1 / self.meanstress,
                                           0, np.nan, True, name)
        elif self.up is False:
            parameters.loc[name + '_A'] = (-1 / self.meanstress,
                                           np.nan, 0, True, name)
        else:
            parameters.loc[name + '_A'] = (1 / self.meanstress,
                                           np.nan, np.nan, True, name)
        parameters.loc[name + '_a'] = (100, 1e-3, 1e4, True, name)
        parameters.loc[name + '_b'] = (1, 1e-6, 25, True, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        # approximate formula for tmax
        if cutoff is None:
            cutoff = self.cutoff
        a, b = p[1:]
        rho = 2 * np.sqrt(b)
        return lambertw(1 / ((1 - cutoff) * k0(rho))).real * a

    @staticmethod
    def gain(p):
        return p[0]

    @staticmethod
    @njit
    def _integrand_hantush(y, b):
        return np.exp(-y - (b / y)) / y

    @staticmethod
    @njit(parallel=True)
    def numba_step(A, a, b, t):
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
                    tau_i + rhosq / (4 * tau_i))
            elif tau_i >= rho / 2:
                F[i] = 2 * k0rho - w * exp1(tau_i) + (w - 1) * exp1(
                    tau_i + rhosq / (4 * tau_i))
        return A * F / (2 * k0rho)

    @staticmethod
    def numpy_step(A, a, b, t):
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
            tau1 + rhosq / (4 * tau1))
        F[~tau_mask] = 2 * k0rho - w * exp1(tau2) + (w - 1) * exp1(
            tau2 + rhosq / (4 * tau2))
        return A * F / (2 * k0rho)

    def quad_step(self, A, a, b, t):
        F = np.zeros_like(t)
        u = a * b / t
        for i in range(0, len(t)):
            F[i] = quad(self._integrand_hantush,
                        u[i], np.inf, args=(b,))[0]
        return F * A / (2 * k0(2 * np.sqrt(b)))

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        A, a, b = p
        t = self.get_t(p, dt, cutoff, maxtmax)

        if self.quad:
            return self.quad_step(A, a, b, t)
        else:
            # if numba_scipy is available and param a >= ~30, numba is faster
            if a >= 30. and self.use_numba:
                return self.numba_step(A, a, b, t)
            else:  # otherwise numpy is faster
                return self.numpy_step(A, a, b, t)

    def impulse(self, t, p):
        A, a, b = p
        ir = A / (2 * t * k0(2 * np.sqrt(b))) * np.exp(-t / a - a * b / t)
        return ir


class Polder(RfuncBase):
    """The Polder function, using the standard A, a, b parameters.

    Notes
    -----
    The Polder function is explained in Eq. 123.32 in
    :cite:t:`bruggeman_analytical_1999`. The impulse response function may be
    written as:

    .. math:: \\theta(t) = \\exp(-\\sqrt(4b)) \\frac{A}{t^{-3/2}}
       \\exp(-t/a -b/t)
    .. math:: p[0] = A = \\exp(-x/\\lambda)
    .. math:: p[1] = a = \\sqrt{\\frac{1}{cS}}
    .. math:: p[2] = b = x^2 / (4 \\lambda^2)

    where :math:`\\lambda = \\sqrt{kDc}`

    """
    _name = "Polder"

    def __init__(self):
        RfuncBase.__init__(self)
        self.nparam = 3

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_A'] = (1, 0, 2, True, name)
        parameters.loc[name + '_a'] = (10, 0.01, 1000, True, name)
        parameters.loc[name + '_b'] = (1, 1e-6, 25, True, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        _, a, b = p
        b = a * b
        x = np.sqrt(b / a)
        inverfc = erfcinv(2 * cutoff)
        y = (-inverfc + np.sqrt(inverfc ** 2 + 4 * x)) / 2
        tmax = a * y ** 2
        return tmax

    def gain(self, p):
        # the steady state solution of Mazure
        g = p[0] * np.exp(-np.sqrt(4 * p[2]))
        if not self.up:
            g = -g
        return g

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        t = self.get_t(p, dt, cutoff, maxtmax)
        A, a, b = p
        s = A * self.polder_function(np.sqrt(b), np.sqrt(t / a))
        # / np.exp(-2 * np.sqrt(b))
        if not self.up:
            s = -s
        return s

    def impulse(self, t, p):
        A, a, b = p
        ir = A * t ** (-1.5) * np.exp(-t / a - b / t)
        return ir

    @staticmethod
    def polder_function(x, y):
        s = 0.5 * np.exp(2 * x) * erfc(x / y + y) + \
            0.5 * np.exp(-2 * x) * erfc(x / y - y)
        return s


class One(RfuncBase):
    """Instant response with no lag and one parameter d.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up
        (True) or down (False), if None (default) the head can go both ways.
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. default is 0.999.
    """
    _name = "One"

    def __init__(self):
        RfuncBase.__init__(self)
        self.nparam = 1

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_d'] = (
                self.meanstress, 0, np.nan, True, name)
        elif self.up is False:
            parameters.loc[name + '_d'] = (
                -self.meanstress, np.nan, 0, True, name)
        else:
            parameters.loc[name + '_d'] = (
                self.meanstress, np.nan, np.nan, True, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        return 0.

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        if isinstance(dt, np.ndarray):
            return p[0] * np.ones(len(dt))
        else:
            return p[0] * np.ones(1)

    def block(self, p, dt=1, cutoff=None, maxtmax=None):
        return p[0] * np.ones(1)


class FourParam(RfuncBase):
    """Four Parameter response function with 4 parameters A, a, b, and n.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False), if None the head can go both ways.
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. default is 0.999.

    Notes
    -----
    The impulse response function may be written as:

    .. math:: \\theta(t) = At^{n-1} e^{-t/a -ab/t}

    If Fourparam.quad is set to True, this response function uses np.quad to
    integrate the Four Parameter response function, which requires more
    calculation time.
    """
    _name = "FourParam"

    def __init__(self, quad=False):
        RfuncBase.__init__(self, quad=quad)
        self.nparam = 4
        self.quad = quad

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (1 / self.meanstress, 0,
                                           100 / self.meanstress, True, name)
        elif self.up is False:
            parameters.loc[name + '_A'] = (-1 / self.meanstress,
                                           -100 / self.meanstress, 0, True,
                                           name)
        else:
            parameters.loc[name + '_A'] = (1 / self.meanstress,
                                           np.nan, np.nan, True, name)

        parameters.loc[name + '_n'] = (1, -10, 10, True, name)
        parameters.loc[name + '_a'] = (10, 0.01, 5000, True, name)
        parameters.loc[name + '_b'] = (10, 1e-6, 25, True, name)
        return parameters

    @staticmethod
    def function(t, p):
        return (t ** (p[1] - 1)) * np.exp(-t / p[2] - p[2] * p[3] / t)

    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff

        if self.quad:
            x = np.arange(1, 10000, 1)
            y = np.zeros_like(x)
            func = self.function(x, p)
            func_half = self.function(x[:-1] + 1 / 2, p)
            y[1:] = y[0] + np.cumsum(1 / 6 *
                                     (func[:-1] + 4 * func_half + func[1:]))
            y = y / quad(self.function, 0, np.inf, args=p)[0]
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
            func = self.function(x, p)
            func_half = self.function(x[:-1] + 1 / 2, p)
            y[0] = 0.5 * (w1 * self.function(0.5 * t1 + 0.5, p) +
                          w2 * self.function(0.5 * t2 + 0.5, p) +
                          w3 * self.function(0.5 * t3 + 0.5, p))
            y[1:] = y[0] + np.cumsum(1 / 6 *
                                     (func[:-1] + 4 * func_half + func[1:]))
            y = y / quad(self.function, 0, np.inf, args=p)[0]
            return np.searchsorted(y, cutoff)

    @staticmethod
    def gain(p):
        return p[0]

    def step(self, p, dt=1, cutoff=None, maxtmax=None):

        if self.quad:
            t = self.get_t(p, dt, cutoff, maxtmax)
            s = np.zeros_like(t)
            s[0] = quad(self.function, 0, dt, args=p)[0]
            for i in range(1, len(t)):
                s[i] = s[i - 1] + quad(self.function, t[i - 1], t[i], args=p)[
                    0]
            s = s * (p[0] / (quad(self.function, 0, np.inf, args=p))[0])
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
                tmax = max(self.get_tmax(p, cutoff), 3 * dt)
                t = np.arange(step, tmax, step)
                s = np.zeros_like(t)

                # for interval [0,dt] :
                s[0] = (step / 2) * \
                       (w1 * self.function((step / 2) * t1 + (step / 2), p) +
                        w2 * self.function((step / 2) * t2 + (step / 2), p) +
                        w3 * self.function((step / 2) * t3 + (step / 2), p))

                # for interval [dt,tmax]:
                func = self.function(t, p)
                func_half = self.function(t[:-1] + step / 2, p)
                s[1:] = s[0] + np.cumsum(
                    step / 6 * (func[:-1] + 4 * func_half + func[1:]))
                s = s * (p[0] / quad(self.function, 0, np.inf, args=p)[0])
                return s[int(dt / step - 1)::int(dt / step)]
            else:
                t = self.get_t(p, dt, cutoff, maxtmax)
                s = np.zeros_like(t)

                # for interval [0,dt] Gaussian quadrate:
                s[0] = (dt / 2) * \
                       (w1 * self.function((dt / 2) * t1 + (dt / 2), p) +
                        w2 * self.function((dt / 2) * t2 + (dt / 2), p) +
                        w3 * self.function((dt / 2) * t3 + (dt / 2), p))

                # for interval [dt,tmax] Simpson integration:
                func = self.function(t, p)
                func_half = self.function(t[:-1] + dt / 2, p)
                s[1:] = s[0] + np.cumsum(
                    dt / 6 * (func[:-1] + 4 * func_half + func[1:]))
                s = s * (p[0] / quad(self.function, 0, np.inf, args=p)[0])
                return s


class DoubleExponential(RfuncBase):
    """Double Exponential response function with 4 parameters A, alpha, a1 and
    a2.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False), if None the head can go both ways.
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. default is 0.999.

    Notes
    -----
    The impulse response function may be written as:

    .. math:: \\theta(t) = A (1 - \\alpha) e^{-t/a_1} + A \\alpha e^{-t/a_2}
    """
    _name = "DoubleExponential"

    def __init__(self):
        RfuncBase.__init__(self)
        self.nparam = 4

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (1 / self.meanstress, 0,
                                           100 / self.meanstress, True, name)
        elif self.up is False:
            parameters.loc[name + '_A'] = (-1 / self.meanstress,
                                           -100 / self.meanstress, 0, True,
                                           name)
        else:
            parameters.loc[name + '_A'] = (1 / self.meanstress,
                                           np.nan, np.nan, True, name)

        parameters.loc[name + '_alpha'] = (0.1, 0.01, 0.99, True, name)
        parameters.loc[name + '_a1'] = (10, 0.01, 5000, True, name)
        parameters.loc[name + '_a2'] = (10, 0.01, 5000, True, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        if p[2] > p[3]:  # a1 > a2
            return -p[2] * np.log(1 - cutoff)
        else:  # a1 < a2
            return -p[3] * np.log(1 - cutoff)

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        t = self.get_t(p, dt, cutoff, maxtmax)
        s = p[0] * (1 - ((1 - p[1]) * np.exp(-t / p[2]) +
                         p[1] * np.exp(-t / p[3])))
        return s


class Edelman(RfuncBase):
    """The function of Edelman, describing the propagation of an instantaneous
    water level change into an adjacent half-infinite aquifer.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False), if None the head can go both ways.
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. default is 0.999.

    Notes
    -----
    The Edelman function is explained in :cite:t:`edelman_over_1947`. The
    impulse response function may be written as:

    .. math:: \\text{unknown}

    It's parameters are:

    .. math:: p[0] = \\beta = \\frac{\\sqrt{\\frac{4kD}{S}}}{x}

    """
    _name = "Edelman"

    def __init__(self):
        RfuncBase.__init__(self)
        self.nparam = 1

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        beta_init = 1.0
        parameters.loc[name + '_beta'] = (beta_init, 0, 1000, True, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        return 1. / (p[0] * erfcinv(cutoff * erfc(0))) ** 2

    @staticmethod
    def gain(p):
        return 1.

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        t = self.get_t(p, dt, cutoff, maxtmax)
        s = erfc(1 / (p[0] * np.sqrt(t)))
        return s


class Kraijenhoff(RfuncBase):
    """The response function of :cite:t:`van_de_leur_study_1958`.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False), if None the head can go both ways.
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. default is 0.999.

    Notes
    -----
    The Kraijenhoff van de Leur function is explained in
    :cite:t:`van_de_leur_study_1958`. The impulse response function may be
    written as:

    .. math:: \\theta(t) = \\frac{4}{\pi S} \sum_{n=1,3,5...}^\infty \\frac{1}{n} e^{-n^2\\frac{t}{j}} \sin (\\frac{n\pi x}{L})

    The function describes the response of a domain between two drainage
    channels. The function gives the same outcome as equation 133.15 in
    :cite:t:`bruggeman_analytical_1999`. This is the response that
    is actually calculated with this function.

    The response function has three parameters: A, a and b.
    A is the gain (scaled),
    a is the reservoir coefficient (j in :cite:t:`van_de_leur_study_1958`),
    b is the location in the domain with the origin in the middle. This means
    that b=0 is in the middle and b=1/2 is at the drainage channel. At b=1/4
    the response function is most similar to the exponential response function.

    """
    _name = "Kraijenhoff"

    def __init__(self, n_terms=10):
        RfuncBase.__init__(self, n_terms=n_terms)
        self.nparam = 3
        self.n_terms = n_terms

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (1 / self.meanstress, 1e-5,
                                           100 / self.meanstress, True, name)
        elif self.up is False:
            parameters.loc[name + '_A'] = (-1 / self.meanstress,
                                           -100 / self.meanstress,
                                           -1e-5, True, name)
        else:
            parameters.loc[name + '_A'] = (1 / self.meanstress,
                                           np.nan, np.nan, True, name)

        parameters.loc[name + '_a'] = (1e2, 0.01, 1e5, True, name)
        parameters.loc[name + '_b'] = (0, 0, 0.499999, True, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        return - p[1] * np.log(1 - cutoff)

    @staticmethod
    def gain(p):
        return p[0]

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        t = self.get_t(p, dt, cutoff, maxtmax)
        h = 0
        for n in range(self.n_terms):
            h += (-1) ** n / (2 * n + 1) ** 3 * \
                np.cos((2 * n + 1) * np.pi * p[2]) * \
                np.exp(-(2 * n + 1) ** 2 * t / p[1])
        s = p[0] * (1 - (8 / (np.pi ** 3 * (1 / 4 - p[2] ** 2)) * h))
        return s


class Spline(RfuncBase):
    """Spline response function with parameters: A and a factor for every t.

    Parameters
    ----------
    up: bool or None, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False), if None the head can go both ways.
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. default is 0.999.
        this parameter is ignored by Points
    t: list
        times at which the response function is defined
    kind: string
        see scipy.interpolate.interp1d. Most useful for a smooth response
        function are ‘quadratic’ and ‘cubic’.

    Notes
    -----
    The spline response function generates a response function from factors at
    t = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 and 1024 days by default. This
    response function is more data-driven than existing response functions and
    has no physical background. Therefore it can primarily be used to compare
    to other more physical response functions, that probably describe the
    groundwater system better.
    """
    _name = "Spline"

    def __init__(self, kind='quadratic',
                 t=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
        RfuncBase.__init__(self, kind=kind, t=t)
        self.kind = kind
        self.t = t
        self.nparam = len(t) + 1

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (1 / self.meanstress, 1e-5,
                                           100 / self.meanstress, True, name)
        elif self.up is False:
            parameters.loc[name + '_A'] = (-1 / self.meanstress,
                                           -100 / self.meanstress,
                                           -1e-5, True, name)
        else:
            parameters.loc[name + '_A'] = (1 / self.meanstress,
                                           np.nan, np.nan, True, name)
        initial = np.linspace(0.0, 1.0, len(self.t) + 1)[1:]
        for i in range(len(self.t)):
            index = name + '_' + str(self.t[i])
            vary = True
            # fix the value of the factor at the last timestep to 1.0
            if i == len(self.t) - 1:
                vary = False
            parameters.loc[index] = (initial[i], 0.0, 1.0, vary, name)

        return parameters

    def get_tmax(self, p, cutoff=None):
        return self.t[-1]

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        f = interp1d(self.t, p[1:len(self.t) + 1], kind=self.kind)
        t = self.get_t(p, dt, cutoff, maxtmax)
        s = p[0] * f(t)
        return s
