# coding=utf-8
"""This module contains all the response functions available in Pastas.

More information on how to write a response class can be found `here
 <http://pastas.readthedocs.io/en/latest/developers.html>_`.

Routines in Module
------------------
Fully supported and tested routines in this module are:

- .. class:: Gamma
- .. class:: Exponential
- .. class:: Hantush
- .. class:: One

TODO
----
- Test Polder response function

"""

import numpy as np
from pandas import DataFrame
from scipy.special import gammainc, gammaincinv, k0, exp1, erfc, lambertw

__all__ = ["Gamma", "Exponential", "Hantush", "One"]


class RfuncBase:
    _name = "RfuncBase"

    def __init__(self, up, meanstress, cutoff):
        self.up = up
        # Completely arbitrary number to prevent divsion by zero
        if meanstress < 1e-8 and meanstress > 0:
            meanstress = 1e-8
        elif meanstress < 0 and up is True:
            meanstress = meanstress * -1
        self.meanstress = meanstress
        self.cutoff = cutoff
        self.tmax = 0

    def set_parameters(self, name):
        pass

    def get_tmax(self, p, cutoff=None):
        """Method to get the response time for a certain cutoff

        Parameters
        ----------
        p:  numpy.array
            numpy array with the parameters.
        cutoff: float, optional
            float between 0 and 1. Default is 0.99.

        Returns
        -------
        tmax: float
            Number of days when 99% of the response has passen, when the
            cutoff is chosen at 0.99.

        """
        pass

    def step(self, p, dt=1, cutoff=None):
        """Method to return the step funtion.

        Parameters
        ----------
        p: numpy.array
            numpy array with the parameters.
        dt: float
            timestep as a multiple of of day.
        cutoff: float, optional
            float between 0 and 1. Default is 0.99.

        Returns
        -------
        s: numpy.array
            Array with the step response.
        """
        pass

    def block(self, p, dt=1, cutoff=None):
        """Method to return the block funtion.

        Parameters
        ----------
        p: numpy.array
            numpy array with the parameters.
        dt: float
            timestep as a multiple of of day.
        cutoff: float, optional
            float between 0 and 1. Default is 0.99.

        Returns
        -------
        s: numpy.array
            Array with the block response.
        """
        s = self.step(p, dt, cutoff)
        return np.append(s[0], s[1:] - s[:-1])


class Gamma(RfuncBase):
    """Gamma response function with 3 parameters A, a, and n.

    Parameters
    ----------
    up: bool, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False)
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        percentage after which the step function is cut off. default=0.99.

    Notes
    -----

    .. math::
        step(t) = A * Gammainc(n, t / a)

    """
    _name = "Gamma"

    def __init__(self, up=True, meanstress=1, cutoff=0.99):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

    def set_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (1 / self.meanstress, 0,
                                           100 / self.meanstress, 1, name)
        else:
            parameters.loc[name + '_A'] = (-1 / self.meanstress,
                                           -100 / self.meanstress, 0, 1, name)
        # if n is too small, the length of the response function is close to zero
        parameters.loc[name + '_n'] = (1, 0.1, 10, 1, name)
        parameters.loc[name + '_a'] = (10, 0.01, 5000, 1, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        return gammaincinv(p[1], cutoff) * p[2]

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1, cutoff=None):
        if isinstance(dt, np.ndarray):
            t = dt
        else:
            self.tmax = max(self.get_tmax(p, cutoff), 3 * dt)
            t = np.arange(dt, self.tmax, dt)

        s = p[0] * gammainc(p[1], t / p[2])
        return s


class Exponential(RfuncBase):
    """Exponential response function with 2 parameters: A and a.

        Parameters
        ----------
        up: bool, optional
            indicates whether a positive stress will cause the head to go up
            (True, default) or down (False)
        meanstress: float
            mean value of the stress, used to set the initial value such that
            the final step times the mean stress equals 1
        cutoff: float
            percentage after which the step function is cut off. default=0.99.

        Notes
        -----
        .. math::
            step(t) = A * (1 - exp(-t / a))

        """
    _name = "Exponential"

    def __init__(self, up=True, meanstress=1, cutoff=0.99):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 2

    def set_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (
                1 / self.meanstress, 0, 100 / self.meanstress, 1, name)
        else:
            parameters.loc[name + '_A'] = (
                -1 / self.meanstress, -100 / self.meanstress, 0, 1, name)
        parameters.loc[name + '_a'] = (10, 0.01, 5000, 1, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        return -p[1] * np.log(1 - cutoff)

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1, cutoff=None):
        if isinstance(dt, np.ndarray):
            t = dt
        else:
            self.tmax = max(self.get_tmax(p, cutoff), 10 * dt)
            t = np.arange(dt, self.tmax, dt)
        s = p[0] * (1.0 - np.exp(-t / p[1]))
        return s


class Hantush(RfuncBase):
    """ The Hantush well function.

    Parameters
    ----------
    up: bool, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False)
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        percentage after which the step function is cut off. default=0.99.

    Notes
    -----
    Parameters are ..math::
        rho = r / lambda and cS

    References
    ----------
    .. [1] Hantush, M. S., & Jacob, C. E. (1955). Non‐steady radial flow in an
        infinite leaky aquifer. Eos, Transactions American Geophysical Union,
        36(1), 95-100.

    .. [2] Veling, E. J. M., & Maas, C. (2010). Hantush well function
    revisited. Journal of hydrology, 393(3), 381-388.

    .. [3] Von Asmuth, J. R., Maas, K., Bakker, M., & Petersen, J. (2008).
    Modeling time series of ground water head fluctuations subjected to
    multiple stresses. Ground Water, 46(1), 30-40.

    """
    _name = "Hantush"

    def __init__(self, up=False, meanstress=1, cutoff=0.99):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

    def set_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (
                1 / self.meanstress, 0, 100 / self.meanstress, 1, name)
        else:
            parameters.loc[name + '_A'] = (
                -1 / self.meanstress, -100 / self.meanstress, 0, 1, name)
        parameters.loc[name + '_rho'] = (1, 1e-4, 10, 1, name)
        parameters.loc[name + '_cS'] = (100, 1e-3, 1e4, 1, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        # approximate formula for tmax
        if cutoff is None:
            cutoff = self.cutoff
        rho = p[1]
        cS = p[2]
        k0rho = k0(rho)
        return lambertw(1 / ((1 - cutoff) * k0rho)).real * cS

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1, cutoff=None):
        rho = p[1]
        cS = p[2]
        k0rho = k0(rho)
        if isinstance(dt, np.ndarray):
            t = dt
        else:
            self.tmax = max(self.get_tmax(p, cutoff), 10 * dt)
            t = np.arange(dt, self.tmax, dt)
        tau = t / cS
        tau1 = tau[tau < rho / 2]
        tau2 = tau[tau >= rho / 2]
        w = (exp1(rho) - k0rho) / (exp1(rho) - exp1(rho / 2))
        F = np.zeros_like(tau)
        F[tau < rho / 2] = w * exp1(rho ** 2 / (4 * tau1)) - (w - 1) * exp1(
            tau1 + rho ** 2 / (4 * tau1))
        F[tau >= rho / 2] = 2 * k0rho - w * exp1(tau2) + (w - 1) * exp1(
            tau2 + rho ** 2 / (4 * tau2))
        return p[0] * F / (2 * k0rho)


class Polder(RfuncBase):
    """The function of Polder, for a river in a confined aquifer,
    overlain by an aquitard with aquiferous ditches.

    References
    ----------
    .. [2] http://grondwaterformules.nl/index.php/formules/waterloop/deklaag
    -met-sloten

    """
    _name = "Polder"

    def __init__(self, up=True, meanstress=1, cutoff=0.99):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

    def set_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        a_init = 1
        b_init = 0.1
        c_init = 1 / np.exp(-2 * a_init) / self.meanstress
        parameters.loc[name + '_a'] = (a_init, 0, 100, 1, name)
        parameters.loc[name + '_b'] = (b_init, 0, 10, 1, name)
        if self.up:
            parameters.loc[name + '_c'] = (c_init, 0, c_init * 100, 1, name)
        else:
            parameters.loc[name + '_c'] = (-c_init, -c_init * 100, 0, 1, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff

        # TODO: find tmax from cutoff, below is just an approximation
        return 4 * p[0] / p[1] ** 2

    def gain(self, p):
        # TODO: check line below
        return p[2]

    def step(self, p, dt=1, cutoff=None):
        if isinstance(dt, np.ndarray):
            t = dt
        else:
            self.tmax = max(self.get_tmax(p, cutoff), 3 * dt)
            t = np.arange(dt, self.tmax, dt)
        s = p[2] * self.polder_function(p[0], p[1] * np.sqrt(t))
        return s

    def polder_function(self, x, y):
        s = .5 * np.exp(2 * x) * erfc(x / y + y) + \
            .5 * np.exp(-2 * x) * erfc(x / y - y)
        return s


class One(RfuncBase):
    """Dummy class for Constant. Returns 1

    """
    _name = "One"

    def __init__(self, up, meanstress, cutoff):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 1

    def set_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_d'] = (1, 0, 100, 1, name)
        else:
            parameters.loc[name + '_d'] = (-1, -100, 0, 1, name)
        return parameters

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1):
        if isinstance(dt, np.ndarray):
            return p[0] * np.ones(len(dt))
        else:
            return p[0] * np.ones(1)

    def block(self, p, dt=1):
        return p[0] * np.ones(1)
