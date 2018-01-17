# coding=utf-8
"""This module contains the classes for the response functions.

More information on how to write a response class can be found :
:ref:` here <http://pastas.github.io/pastas/developers.html>`
"""

from __future__ import print_function, division

import numpy as np
import pandas as pd
from scipy.special import gammainc, gammaincinv, k0, exp1, erfc, lambertw

_class_doc = """
Parameters
----------
nparam: int
    number of parameters.
up: bool
    indicates whether a positive stress will cause the head to go up
    (True) or down (False)
meanstress: float
    mean value of the stress, used to set the initial value such that
    the final step times the mean stress equals 1
cutoff: float
    percentage after which the step function is cut off.
tmax: float
    time corresponding to the cutoff

"""


class RfuncBase:
    _name = "RfuncBase"

    def __init__(self, up, meanstress, cutoff):
        if up:
            self.up = 1
        else:
            self.up = -1
        self.meanstress = meanstress
        self.cutoff = cutoff
        self.tmax = 0

    def set_parameters(self, name):
        pass

    def step(self, p, dt=1):
        pass

    def block(self, p, dt=1):
        pass


class Gamma(RfuncBase):
    _name = "Gamma"

    __doc__ = """Gamma response function with 3 parameters A, a, and n.

    .. math::
        step(t) = A * Gammainc(n, t / a)

    %(doc)s
    """ % {'doc': _class_doc}

    def __init__(self, up=True, meanstress=1, cutoff=0.99):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

    def set_parameters(self, name):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_A'] = (1 / self.meanstress, 0,
                                       100 / self.meanstress, 1, name)
        # if n is too small, the length of the response function is close to zero
        parameters.loc[name + '_n'] = (1, 0.1, 5, 1, name)
        parameters.loc[name + '_a'] = (100, 1, 5000, 1, name)
        return parameters

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1):
        if isinstance(dt, np.ndarray):
            t = dt
        else:
            self.tmax = gammaincinv(p[1], self.cutoff) * p[2]
            self.tmax = max(self.tmax, 3 * dt)
            t = np.arange(dt, self.tmax, dt)

        s = self.up * p[0] * gammainc(p[1], t / p[2])
        return s

    def block(self, p, dt=1):
        s = self.step(p, dt)
        return np.append(s[0], s[1:] - s[:-1])


class Exponential(RfuncBase):
    _name = "Exponential"

    __doc__ = """Exponential response function with 2 parameters: A and a.

    .. math::
        step(t) = A * (1 - exp(-t / a))

    %(doc)s
    """ % {'doc': _class_doc}

    def __init__(self, up=True, meanstress=1, cutoff=0.99):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 2

    def set_parameters(self, name):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_A'] = (
            1 / self.meanstress, 0, 100 / self.meanstress, 1, name)
        parameters.loc[name + '_a'] = (100, 1, 5000, 1, name)
        return parameters

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1):
        if isinstance(dt, np.ndarray):
            t = dt
        else:
            self.tmax = -np.log(1.0 / p[1]) * p[1]
            self.tmax = max(self.tmax, 3 * dt)
            t = np.arange(dt, self.tmax, dt)
        s = self.up * p[0] * (1.0 - np.exp(-t / p[1]))
        return s

    def block(self, p, dt=1):
        s = self.step(p, dt)
        return np.append(s[0], s[1:] - s[:-1])


class Hantush(RfuncBase):
    _name = "Hantush"
    __doc__ = """ The Hantush well function.

    Notes
    -----
    Parameters are ..math::
        rho = r / lambda and cS

    References
    ----------
    Hantush, M. S., & Jacob, C. E. (1955). Non‐steady radial flow in an infinite leaky aquifer. Eos, Transactions American Geophysical Union, 36(1), 95-100.

    Veling, E. J. M., & Maas, C. (2010). Hantush well function revisited. Journal of hydrology, 393(3), 381-388.

    Von Asmuth, J. R., Maas, K., Bakker, M., & Petersen, J. (2008). Modeling time series of ground water head fluctuations subjected to multiple stresses. Ground Water, 46(1), 30-40.

    """

    def __init__(self, up=False, meanstress=1, cutoff=0.99):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

    def set_parameters(self, name):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_A'] = (
            1 / self.meanstress, 0, 100 / self.meanstress, 1, name)
        parameters.loc[name + '_rho'] = (1, 0.0001, 10, 1, name)
        parameters.loc[name + '_cS'] = (100, 1e-3, 1e3, 1, name)
        return parameters

    def calc_tmax(self, p):
        rho = p[1]
        cS = p[2]
        k0rho = k0(rho)
        return lambertw(1 / ((1 - self.cutoff) * k0rho)).real * cS

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1):
        rho = p[1]
        cS = p[2]
        k0rho = k0(rho)
        if isinstance(dt, np.ndarray):
            t = dt
        else:
            # approximate formula for tmax
            self.tmax = lambertw(1 / ((1 - self.cutoff) * k0rho)).real * cS
            self.tmax = max(self.tmax, 3 * dt)
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
        return self.up * p[0] * F / (2 * k0rho)

    def block(self, p, dt=1):
        s = self.step(p, dt)
        return np.append(s[0], s[1:] - s[:-1])


class Theis(RfuncBase):
    _name = "Theis"
    __doc__ = """The Theis well function.

    Notes
    -----
    Theis may not be very appropiate, as the drawdown will continue
    indefinitely.

    References
    ----------
    Theis, C. V. (1935). The relation between the lowering of the Piezometric surface and the rate and duration of discharge of a well using groundwater storage. Eos, Transactions American Geophysical Union, 16(2), 519-524.

    """

    def __init__(self, up=True, meanstress=1, cutoff=0.99):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

    def set_parameters(self, name):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_S'] = (0.25, 1e-3, 1.0, 1, name)
        parameters.loc[name + '_T'] = (100.0, 0.0, 10000.0, 1, name)
        parameters.loc[name + '_r'] = (1000.0, 0.0, 100000.0, 0, name)
        return parameters

    def gain(self, p):
        return np.inf

    def step(self, p, dt=1):
        if isinstance(dt, np.ndarray):
            t = dt
        else:
            self.tmax = 10000  # This should be changed with some analytical expression
            self.tmax = max(self.tmax, 3 * dt)
            t = np.arange(dt, self.tmax, dt)
        r = p[2]
        u = r ** 2.0 * p[0] / (4.0 * p[1] * t)
        s = self.up * exp1(u)
        return s

    def block(self, p, dt=1):
        s = self.step(p, dt)
        return np.append(s[0], s[1:] - s[:-1])


class Bruggeman(RfuncBase):
    _name = "Bruggeman"
    __doc__ = """The function of Bruggeman, for a river in a confined aquifer,
    overlain
    by an aquitard with aquiferous ditches.

    References
    ----------
    http://grondwaterformules.nl/index.php/formules/waterloop/deklaag-met-sloten

    """

    def __init__(self, up=True, meanstress=1, cutoff=0.99):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

    def set_parameters(self, name):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        a_init = 1
        b_init = 0.1
        c_init = 1 / np.exp(-2 * a_init) / self.meanstress
        parameters.loc[name + '_a'] = (a_init, 0, 100, 1, name)
        parameters.loc[name + '_b'] = (b_init, 0, 10, 1, name)
        parameters.loc[name + '_c'] = (c_init, 0, c_init * 100, 1, name)
        return parameters

    def gain(self, p):
        # TODO: check line below
        return p[2]

    def step(self, p, dt=1):
        # TODO: find tmax from cutoff, below is just an opproximation
        if isinstance(dt, np.ndarray):
            t = dt
        else:
            self.tmax = 4 * p[0] / p[1] ** 2
            self.tmax = max(self.tmax, 3 * dt)
            t = np.arange(dt, self.tmax, dt)
        s = self.up * p[2] * self.polder_function(p[0], p[1] * np.sqrt(t))
        return s

    def polder_function(self, x, y):
        s = .5 * np.exp(2 * x) * erfc(x / y + y) + \
            .5 * np.exp(-2 * x) * erfc(x / y - y)
        return s

    def block(self, p, dt=1):
        s = self.step(p, dt)
        return np.append(s[0], s[1:] - s[:-1])


class One(RfuncBase):
    _name = "One"
    __doc__ = """Dummy class for Constant. Returns 1

    """

    def __init__(self, up, meanstress, cutoff):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 1

    def set_parameters(self, name):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_d'] = (1, 0, 100, 1, name)
        return parameters

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1):
        if isinstance(dt, np.ndarray):
            return p[0] * np.ones(len(dt))
        else:
            return p[0] * np.ones(2)

    def block(self, p, dt=1):
        return p[0] * np.ones(2)
