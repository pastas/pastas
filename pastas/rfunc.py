# coding=utf-8
"""This module contains all the response functions available in Pastas.

Supported Response Functions
----------------------------
.. autosummary::
    :nosignatures:
    :toctree: generated/

    Gamma
    Exponential
    Hantush
    HantushWellModel
    FourParam
    One

.. warning::
    The above list contains the supported response function. All other
    methods are for research purposes only and may change without notice.

"""

import numpy as np
from pandas import DataFrame
from scipy.integrate import quad
from scipy.special import gammainc, gammaincinv, k0, exp1, erfc, lambertw, \
    erfcinv

__all__ = ["Gamma", "Exponential", "Hantush", "Polder", "FourParam",
           "DoubleExponential", "One", "Edelman", "HantushWellModel"]


class RfuncBase:
    _name = "RfuncBase"

    def __init__(self, up, meanstress, cutoff):
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
        """Method to get the response time for a certain cutoff

        Parameters
        ----------
        p:  numpy.array
            numpy array with the parameters.
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
        p: numpy.array
            numpy array with the parameters.
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
        """Method to return the block funtion.

        Parameters
        ----------
        p: numpy.array
            numpy array with the parameters.
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

    def get_t(self, p, dt, cutoff, maxtmax=None):
        """Internal method to detemine the times at which to evaluate the step-
        response, from t=0

        Parameters
        ----------
        p: numpy.array
            numpy array with the parameters.
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

    .. math::
        step(t) = A * Gammainc(n, t / a)

    """
    _name = "Gamma"

    def __init__(self, up=True, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
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
        parameters.loc[name + '_n'] = (1, 0.1, 10, True, name)
        parameters.loc[name + '_a'] = (10, 0.01, 1000, True, name)
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
    .. math::
        step(t) = A * (1 - e^{-\\frac{t}{a}})

    """
    _name = "Exponential"

    def __init__(self, up=True, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
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


class Hantush(RfuncBase):
    """
    The Hantush well function.

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
    The Hantush well function is explained in [hantush_1955]_, [veling_2010]_
    and [asmuth_2008]_. It's parameters are:

    .. math:: p[0] = A = \\frac{1}{4 \\pi kD}
    .. math:: p[1] = \\rho = \\frac{r}{\\lambda}
    .. math:: p[2] = cS

    where :math:`\\lambda = \\sqrt{kDc}`

    References
    ----------
    .. [hantush_1955] Hantush, M. S., & Jacob, C. E. (1955). Non‐steady
       radial flow in an infinite leaky aquifer. Eos, Transactions American
       Geophysical Union, 36(1), 95-100.

    .. [veling_2010] Veling, E. J. M., & Maas, C. (2010). Hantush well function
       revisited. Journal of hydrology, 393(3), 381-388.

    .. [asmuth_2008] Von Asmuth, J. R., Maas, K., Bakker, M., & Petersen,
       J. (2008). Modeling time series of ground water head fluctuations
       subjected to multiple stresses. Ground Water, 46(1), 30-40.

    """
    _name = "Hantush"

    def __init__(self, up=False, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

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

        parameters.loc[name + '_rho'] = (1, 1e-4, 10, True, name)
        parameters.loc[name + '_cS'] = (100, 1e-3, 1e4, True, name)
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

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        rho = p[1]
        cS = p[2]
        k0rho = k0(rho)
        t = self.get_t(p, dt, cutoff, maxtmax)
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


class HantushWellModel(RfuncBase):
    """
    A special implementation of the Hantush well function for
    multiple wells.

    Parameters
    ----------
    up: bool, optional
        indicates whether a positive stress will cause the head to go up
        (True, default) or down (False)
    meanstress: float
        mean value of the stress, used to set the initial value such that
        the final step times the mean stress equals 1
    cutoff: float
        proportion after which the step function is cut off. default is 0.999.

    Notes
    -----
    The HantushWellModel well function is explained in [hantush_1955]_,
    [veling_2010]_ and [asmuth_2008]_. It's parameters are (note the
    addition of the r parameter in this implementation):

    .. math:: p[0] = A = \\frac{1}{4 \\pi kD} \\cdot 2 k_0 \\left( \\frac{r}{\\lambda} \\right)
    .. math:: p[1] = lab = \\lambda
    .. math:: p[2] = cS
    .. math:: p[3] = r \\text{(not optimized)}

    where :math:`\\lambda = \\sqrt{\\frac{kD}{c}}`

    Notes
    -----
    The parameter r (distance from the well to the observation point)
    is passed as a known value, and is used to scale the response function.
    The optimized parameters are slightly different from the original
    Hantush implementation:

    - A: To get the same A as the original Hantush:
        A_orig = A * 2 * k0(r / lambda) or use the gain() method
    - lab: lambda, the r parameter is passed separately to calculate
        rho = r / lambda internally
    - cS: stays the same
    - r: distance, used to calculate rho, see lab.

    """
    _name = "HantushWellModel"

    def __init__(self, up=False, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_A'] = (
                1 / self.meanstress, 0, 100 / self.meanstress, True, name)
        else:
            parameters.loc[name + '_A'] = (
                -1 / self.meanstress, -100 / self.meanstress, 0, True, name)
        parameters.loc[name + '_lab'] = (1000, 1, 1e6, True, name)
        parameters.loc[name + '_cS'] = (100, 1e-3, 1e4, True, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        r = 1 if len(p) == 3 else p[3]
        # approximate formula for tmax
        if cutoff is None:
            cutoff = self.cutoff
        rho = r / p[1]
        cS = p[2]
        k0rho = k0(rho)
        return lambertw(1 / ((1 - cutoff) * k0rho)).real * cS

    def gain(self, p):
        r = 1 if len(p) == 3 else p[3]
        return p[0] * 2 * k0(r / p[1])

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        r = 1 if len(p) == 3 else p[3]
        rho = r / p[1]
        cS = p[2]
        k0rho = k0(rho)
        t = self.get_t(p, dt, cutoff, maxtmax)
        tau = t / cS
        tau1 = tau[tau < rho / 2]
        tau2 = tau[tau >= rho / 2]
        w = (exp1(rho) - k0rho) / (exp1(rho) - exp1(rho / 2))
        F = np.zeros_like(tau)
        F[tau < rho / 2] = w * exp1(rho ** 2 / (4 * tau1)) - (w - 1) * exp1(
            tau1 + rho ** 2 / (4 * tau1))
        F[tau >= rho / 2] = 2 * k0rho - w * exp1(tau2) + (w - 1) * exp1(
            tau2 + rho ** 2 / (4 * tau2))
        return p[0] * F
    
class Hantush2(RfuncBase):
    """
    The Hantush well function, using the standard A, a, b parameters

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
    The Hantush well function is explained in [hantush_1955]_, [veling_2010]_
    and [asmuth_2008]_. The impulse response function may be written as
    
    .. math:: \\theta(t) = \\frac{A}{t} \\exp(-t/a -b/t)

    .. math:: p[0] = A # check \\frac{1}{4 \\pi kD}
    .. math:: p[1] = a = cS
    .. math:: p[2] = b = r^2 cS / (4 \\lambda^2)

    where :math:`\\lambda = \\sqrt{kDc}`

    References
    ----------
    .. [hantush_1955] Hantush, M. S., & Jacob, C. E. (1955). Non‐steady
       radial flow in an infinite leaky aquifer. Eos, Transactions American
       Geophysical Union, 36(1), 95-100.

    .. [veling_2010] Veling, E. J. M., & Maas, C. (2010). Hantush well function
       revisited. Journal of hydrology, 393(3), 381-388.

    .. [asmuth_2008] Von Asmuth, J. R., Maas, K., Bakker, M., & Petersen,
       J. (2008). Modeling time series of ground water head fluctuations
       subjected to multiple stresses. Ground Water, 46(1), 30-40.

    """
    _name = "Hantush2"

    def __init__(self, up=False, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

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
        parameters.loc[name + '_a'] = (100, 1e-3, 1e4, True, name)
        parameters.loc[name + '_b'] = (1, 1e-4, 1e4, True, name)
        return parameters

    def get_tmax(self, p, cutoff=None):
        # approximate formula for tmax
        if cutoff is None:
            cutoff = self.cutoff
        cS = p[1]
        rho = np.sqrt(4 * p[2] / p[1])
        k0rho = k0(rho)
        return lambertw(1 / ((1 - cutoff) * k0rho)).real * cS

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        cS = p[1]
        rho = np.sqrt(4 * p[2] / p[1])
        k0rho = k0(rho)
        t = self.get_t(p, dt, cutoff, maxtmax)
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
    """The Polder function, for a river in a confined aquifer,
    overlain by an aquitard with aquiferous ditches.

    Notes
    -----
    The Polder function is explained in [polder]_. It's parameters are:

    .. math:: p[0] = \\frac{x}{2\\lambda}
    .. math:: p[1] = \\sqrt{\\frac{1}{cS}}

    where :math:`\\lambda = \\sqrt{\\frac{kD}{c}}`

    References
    ----------
    .. [polder] http://grondwaterformules.nl/index.php/formules/waterloop
    /deklaag-met-sloten

    """
    _name = "Polder"

    def __init__(self, up=True, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        A_init = 1
        a_init = 1
        b_init = 1
        parameters.loc[name + '_A'] = (A_init, 0, 2, False, name)
        parameters.loc[name + '_a'] = (a_init, 0, 100, True, name)
        parameters.loc[name + '_b'] = (b_init, 0, 10, True, name)
        return parameters

#     def get_tmax(self, p, cutoff=None):
#         if cutoff is None:
#             cutoff = self.cutoff
#         a = p[1]
#         b = erfcinv(2 * cutoff)
#         c = -p[1] / p[2]
#         sqrttmax = (-b + np.sqrt(b ** 2 - 4 * a * c) / (2 * a))
#         return sqrttmax ** 2
    
    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        A, p0, sqrtp1 = p
        p1 = sqrtp1 ** 2
        inverfc = erfcinv(2 * cutoff) 
        y = ((-inverfc + np.sqrt(inverfc ** 2 + 4 * p0)) / 2)
        tmax = (y / np.sqrt(p1)) ** 2
        return tmax

    def gain(self, p):
        # the steady state solution of Mazure
        g = p[0] * np.exp(-2 * p[1])
        if not self.up:
            g = -g
        return g

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        t = self.get_t(p, dt, cutoff, maxtmax)
        s = p[0] * self.polder_function(p[1], p[2] * np.sqrt(t))
        if not self.up:
            s = -s
        return s

    @staticmethod
    def polder_function(x, y):
        s = 0.5 * np.exp(2 * x) * erfc(x / y + y) + \
            0.5 * np.exp(-2 * x) * erfc(x / y - y)
        return s

    
class Polder2(RfuncBase):
    """The Polder function, using the standard A, a, b parameters

    Notes
    -----
    The Polder function is explained in [polder]_. 
    The impulse response function may be written as
    
    .. math:: \\theta(t) = \\frac{A}{t^{-3/2}} \\exp(-t/a -b/t)

    .. math:: p[0] = A = \\exp(-x/\\lambda)
    .. math:: p[1] = a = \\sqrt{\\frac{1}{cS}}
    .. math:: p[2] = b = x^2 cS / (4 \\lambda^2)

    where :math:`\\lambda = \\sqrt{kDc}`

    References
    ----------
    .. [polder] [1] G.A. Bruggeman (1999). Analytical solutions of 
    geohydrological problems. Elsevier Science. Amsterdam, Eq. 123.32

    """
    _name = "Polder2"

    def __init__(self, up=True, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 3

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        A_init = 1
        a_init = 10
        b_init = 1
        parameters.loc[name + '_A'] = (A_init, 0, 2, True, name)
        parameters.loc[name + '_a'] = (a_init, 0.01, 1000, True, name)
        parameters.loc[name + '_b'] = (b_init, 1e-4, 1e4, True, name)
        return parameters
    
    def get_tmax(self, p, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        A, a, b = p
        x = np.sqrt(b / a)
        inverfc = erfcinv(2 * cutoff) 
        y = (-inverfc + np.sqrt(inverfc ** 2 + 4 * x)) / 2
        tmax = a * y ** 2
        return tmax

    def gain(self, p):
        # the steady state solution of Mazure
        g = p[0]
        if not self.up:
            g = -g
        return g

    def step(self, p, dt=1, cutoff=None, maxtmax=None):
        t = self.get_t(p, dt, cutoff, maxtmax)
        A, a, b = p
        s = p[0] * self.polder_function(np.sqrt(b / a), np.sqrt(t / a)) / \
            np.exp(-2 * np.sqrt(b / a))
        if not self.up:
            s = -s
        return s

    @staticmethod
    def polder_function(x, y):
        s = 0.5 * np.exp(2 * x) * erfc(x / y + y) + \
            0.5 * np.exp(-2 * x) * erfc(x / y - y)
        return s
    

class One(RfuncBase):
    """Dummy class for Constant. Returns 1

    """
    _name = "One"

    def __init__(self, up=None, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 1

    def get_init_parameters(self, name):
        parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        if self.up:
            parameters.loc[name + '_d'] = (
                self.meanstress, 0, np.nan, True, name)
        elif self.up is False:
            parameters.loc[name + '_d'] = (
                self.meanstress, np.nan, 0, True, name)
        else:
            parameters.loc[name + '_d'] = (
                self.meanstress, np.nan, np.nan, True, name)
        return parameters

    def gain(self, p):
        return p[0]

    def step(self, p, dt=1, cutoff=None):
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

    .. math::
        step(t) = \\frac{A}{quad(t^n*e^{-\\frac{t}{a} - \\frac{b}{t}},0,inf)} *
                            quad(t^n*e^{-\\frac{t}{a} - \\frac{b}{t}},0,t)

    """
    _name = "FourParam"

    def __init__(self, up=True, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
        self.nparam = 4
        self.quad = False

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
        parameters.loc[name + '_b'] = (10, 0.01, 5000, True, name)
        return parameters

    @staticmethod
    def function(t, p):
        return (t ** (p[1] - 1)) * np.exp(-t / p[2] - p[3] / t)

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
                s[1:] = s[0] + np.cumsum(step / 6 *
                                         (func[:-1] + 4 * func_half + func[
                                                                      1:]))
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
                s[1:] = s[0] + np.cumsum(dt / 6 *
                                         (func[:-1] + 4 * func_half + func[
                                                                      1:]))
                s = s * (p[0] / quad(self.function, 0, np.inf, args=p)[0])
                return s


class FourParamQuad(FourParam):
    """"Four Parameter response function with 4 parameters A, a, b, and n.

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
    This response function uses np.quad to integrate the Four Parameter
    response function, which requires more calculation time. This response
    function can be used for testing purposes.

    .. math::
        step(t) = \\frac{A}{quad(t^n*e^{-\\frac{t}{a} - \\frac{b}{t}},0,inf)} *
                            quad(t^n*e^{-\\frac{t}{a} - \\frac{b}{t}},0,t)

    """
    _name = "FourParamQuad"

    def __init__(self, up=True, meanstress=1, cutoff=0.999):
        FourParam.__init__(self, up, meanstress, cutoff)
        self.nparam = 4
        self.quad = True


class DoubleExponential(RfuncBase):
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

    .. math::
        step(t) = A * (1 - ( (1 - \\alpha)* e^{-\\frac{t}{a1}} +
                                  \\alpha * e^{-\\frac{t}{a2}}))

    """
    _name = "DoubleExponential"

    def __init__(self, up=True, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
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
    The Edelman function is emplained in [5]_. It's parameters are:

    .. math:: p[0] = \\beta = \\frac{\\sqrt{\\frac{4kD}{S}}}{x}

    References
    ----------
    .. [5] http://grondwaterformules.nl/index.php/formules/waterloop/peilverandering

    """
    _name = "Edelman"

    def __init__(self, up=True, meanstress=1, cutoff=0.999):
        RfuncBase.__init__(self, up, meanstress, cutoff)
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
