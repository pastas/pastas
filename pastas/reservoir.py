"""This module contains the classes for reservoir models.

.. codeauthor:: M. Bakker, Delft University of Technology

See Also
--------
pastas.stressmodels.ReservoirModel
    The reservoir models are provided to a ReservoirModel

Examples
--------
To be added
"""

import numpy as np
from numpy import float64
from scipy.integrate import solve_ivp # only used in simulateold
from pandas import DataFrame
from pastas.decorators import njit

class ReservoirBase:
    """Base class for reservoir classes."""

    def __init__(self):
        self.temp = False
        self.nparam = 0

    @staticmethod
    def get_init_parameters(name="recharge"):
        """Method to obtain the initial parameters.

        Parameters
        ----------
        name: str, optional
            String with the name that is used as prefix for the parameters.

        Returns
        -------
        parameters: pandas.DataFrame
            Pandas DataFrame with the parameters.
        """
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        return parameters

    def simulate(self, prec, evap, p, dt=1.0, **kwargs):
        pass


class Reservoir1(ReservoirBase):
    """Single reservoir.

    Notes
    -----
    Should be the same as the exponential function

    References
    ----------
    None
    """
    _name = "Reservoir1"

    def __init__(self, initialhead):
        ReservoirBase.__init__(self)
        self.nparam = 4
        self.initialhead = initialhead

    def get_init_parameters(self, name="reservoir"):
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_S"] = (0.1, 0.001, 1, True, name)
        parameters.loc[name + "_c"] = (100, 1, 5000, True, name)
        dmean = self.initialhead
        parameters.loc[name + "_d"] = (dmean, dmean - 10, 
                                       dmean + 10, True, name)
        parameters.loc[name + "_f"] = (-1.0, -2.0, 0.0, True, name)
        return parameters
    
    def simulate(self, prec, evap, p):
        return self.simulatehead(prec.values, evap.values, p)

    @staticmethod
    @njit
    def simulatehead(prec, evap, p):
        """Simulate the head in the reservoir

        Parameters
        ----------
        prec, evap: array_like
            array with the precipitation and potential evaporation values. These
            arrays must be of the same length and at the same time steps.
        p: array_like
            array_like object with the values as floats representing the
            model parameters.

        Returns
        -------
        head in reservoir: array_like
            array with the head in the reservoir
        """
        S, c, d, f = p
        delt = 1
        nsteps = len(prec)
        h = np.empty(nsteps + 1, dtype=float64)
        h[0] = d
        for i in range(1, nsteps + 1):
            rech = prec[i - 1] + f * evap[i - 1]
            h[i] = h[i - 1] + delt * (rech / S - (h[i - 1] - d) / (c * S))
        return h[1:]

#     def simulateold(self, prec, evap, p, **kwargs):
#         """Implementation using solve_ivp - too slow and 

#         Parameters
#         ----------
#         prec, evap: array_like
#             array with the precipitation and potential evaporation values. These
#             arrays must be of the same length and at the same time steps.
#         p: array_like
#             array_like object with the values as floats representing the
#             model parameters.

#         Returns
#         -------
#         head in reservoir: array_like
#             array with the head in the reservoir
#         """
#         S, c, d, f = p
#         tmax = len(prec)
#         eps = 1e-6
#         t = np.linspace(1 + eps, tmax - eps, tmax)
#         path2 = solve_ivp(self.dhdt, (0, t[-1]), y0=[d], t_eval=t, rtol=1e-4, 
#                           max_step=1, method='RK23', args=(prec, evap, p))
#         h = path2.y[0]
#         return h
    
#     def dhdt(self, t, h, prec, evap, p):
#         S, c, d, f = p
#         #print(t, int(t), int(t))
#         R = prec[int(t - 1)] + f * evap[int(t - 1)]
#         rv = R / S - (h[0] - d) / (c * S)
#         #rv += -expit(100 * (h[0] - 19.6)) * (h[0] - 19.6) / 20
#         return rv

class Reservoir2(ReservoirBase):
    """Single reservoir with outflow at two heights

    Notes
    -----
    None

    References
    ----------
    None
    """
    _name = "Reservoir2"

    def __init__(self, initialhead):
        ReservoirBase.__init__(self)
        self.nparam = 6
        self.initialhead = initialhead

    def get_init_parameters(self, name="reservoir"):
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_S"] = (0.1, 0.001, 1, True, name)
        parameters.loc[name + "_c"] = (100, 1, 5000, True, name)
        dmean = self.initialhead
        parameters.loc[name + "_d"] = (dmean, dmean - 10, 
                                       dmean + 10, True, name)
        parameters.loc[name + "_f"] = (-1.0, -2.0, 0.0, True, name)
        parameters.loc[name + "_c2"] = (100, 1, 1000, True, name)
        parameters.loc[name + "_deld"] = (0.01, 0.001, 10, True, name)
        return parameters
    
    def simulate(self, prec, evap, p):
        return self.simulatehead(prec.values, evap.values, p)

    @staticmethod
    @njit
    def simulatehead(prec, evap, p):
        """Simulate the head in the reservoir

        Parameters
        ----------
        prec, evap: array_like
            array with the precipitation and potential evaporation values. These
            arrays must be of the same length and at the same time steps.
        p: array_like
            array_like object with the values as floats representing the
            model parameters.

        Returns
        -------
        head in reservoir: array_like
            array with the head in the reservoir
        """
        S, c, d, f, c2, deld = p
        d2 = d + deld
        delt = 1
        nsteps = len(prec)
        h = np.empty(nsteps + 1, dtype=float64)
        h[0] = d
        for i in range(1, nsteps + 1):
            rech = prec[i - 1] + f * evap[i - 1]
            h[i] = h[i - 1] + delt * (rech / S - (h[i - 1] - d) / (c * S))
            if h[i - 1] > d2:
                h[i] -= delt * (h[i - 1] - d2) / (c2 * S)
        return h[1:]