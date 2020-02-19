"""recharge_func module

Author: R.A. Collenteur, University of Graz

Contains the classes for the different models that are available to calculate
the recharge from precipitation and evaporation data.

Each Recharge class contains at least the following:

Attributes
----------
nparam: int
    Number of parameters needed for this model.

Functions
---------
get_init_parameters(self, name)
    A function that returns a Pandas DataFrame of the parameters of the
    recharge function. Columns of the dataframe need to be ["value", "pmin",
    "pmax", "vary"]. Rows of the DataFrame have names of the parameters. Input
    name is used as a prefix. This function is called by a stressmodel object.
simulate(self, evap, prec, p=None)
    A function that returns an array of the simulated recharge series.

"""

import numpy as np
import pandas as pd

from ..decorators import njit


class RechargeBase:
    """Base class for classes that calculate the recharge.

    """

    def __init__(self):
        self.temp = False
        self.nparam = 0

    @staticmethod
    def get_init_parameters(name="recharge"):
        """

        Parameters
        ----------
        name: str, optional
            String with the name that is used as prefix for the parameters.

        Returns
        -------
        parameters: pandas.DataFrame
            Pandas DataFrame with the parameters.

        """
        parameters = pd.DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        return parameters

    def simulate(self, prec, evap, p, temp=None):
        pass


class Linear(RechargeBase):
    """Linear recharge model.

    The recharge to the groundwater is calculated as:
    R = P - f * E

    """
    _name = "Linear"

    def __init__(self):
        RechargeBase.__init__(self)
        self.nparam = 1

    def get_init_parameters(self, name="rch"):
        parameters = pd.DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_f"] = (-1.0, -2.0, 0.0, True, name)
        return parameters

    def simulate(self, prec, evap, p, **kwargs):
        """

        Parameters
        ----------
        prec, evap: array_like
            array with the precipitation and evaporation values. These
            arrays must be of the same length and at the same time steps.
        p: float
            parameter value used in recharge calculation.

        Returns
        -------
        recharge: array_like
            array with the recharge series.

        """
        return np.add(prec, np.multiply(evap, p))


class FlexModel:
    """
    Simple recharge model with minimum number of calibration parameters.

    Other water balance for the root zone s calculated as:

    dS/dt = Pe - Kp * (Sr / Srmax)**Gamma - Epu * min(1, Sr / (0.5 * Srmax))

    """
    _name = "FlexModel"

    def __init__(self):
        self.nparam = 5
        self.dt = 1.0
        self.solver = 0
        self.temp = False

    def get_init_parameters(self, name="rch"):
        parameters = pd.DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_sr"] = (150.0, 1e-5, 1e3, True, name)
        parameters.loc[name + "_lp"] = (0.5, 1e-5, 1, False, name)
        parameters.loc[name + "_ks"] = (50.0, 1, 1e3, True, name)
        parameters.loc[name + "_gamma"] = (4.0, 1e-5, 50.0, True, name)
        parameters.loc[name + "_si"] = (1.0, 1e-5, 10.0, False, name)
        return parameters

    def simulate(self, prec, evap, p=None, **kwargs):
        r = self.get_recharge(prec, evap, sr=p[0], lp=p[1], ks=p[2],
                              gamma=p[3], si=p[4], dt=self.dt)[0]
        # self.check_waterbalance(s, fluxes=[-r, -ea, pe])
        return r

    def check_waterbalance(self, s, fluxes):
        wb = np.sum(fluxes, axis=0)
        ds = s[1:] - s[0:-1]
        # print(np.sum(wb[0:-1] - ds))

    @staticmethod
    @njit
    def get_recharge(p, e, sr=250.0, lp=0.5, ks=50.0, gamma=4.0, si=2.0,
                     dt=1.0):
        n = p.size
        # Create an empty array to store the soil state in
        s = np.zeros(n, dtype=np.float64)
        s[0] = 0.5 * sr  # Set the initial system state
        ea = np.zeros(n, dtype=np.float64)
        r = np.zeros(n, dtype=np.float64)
        i = np.zeros(n, dtype=np.float64)
        pe = np.zeros(n, dtype=np.float64)
        ei = np.zeros(n, dtype=np.float64)
        ep = np.zeros(n, dtype=np.float64)
        lp = lp * sr  # Do this here outside the for-loop for efficiency

        for t in range(n - 1):
            # Interception bucket.
            pe[t] = max(p[t] - si + i[t], 0.0)
            ei[t] = min(e[t], i[t])
            ep[t] = e[t] - ei[t]
            i[t + 1] = i[t] + dt * (p[t] - pe[t] - ei[t])

            # Make sure the solution is larger then 0.0 and smaller than sr
            if s[t] > sr:
                s[t] = sr
            elif s[t] < 0.0:
                s[t] = 0.0

            # Calculate actual ET
            if s[t] / lp < 1.0:
                ea[t] = ep[t] * s[t] / lp
            else:
                ea[t] = ep[t]

            r[t] = ks * (s[t] / sr) ** gamma
            # Make sure the solution is larger then 0.0 and smaller than sr
            s[t + 1] = s[t] + dt * (pe[t] - r[t] - ea[t])

        return r, s, ea, p


class Berendrecht:
    """
    Percolation and preferential flow recharge model

    Other water balance for the root zone s calculated as:

    dS/dt = Pe - Kp * (Sr / Srmax)**Gamma - Epu * min(1, Sr / (0.5 * Srmax))

    References
    ----------

    """
    _name = "Berendrecht"

    def __init__(self):
        self.nparam = 7
        self.dt = 1
        self.solver = 0
        self.temp = False

    def get_init_parameters(self, name="recharge"):
        parameters = pd.DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_fi"] = (0.9, 0.7, 1.3, False, name)
        parameters.loc[name + "_fc"] = (1.0, 0.7, 1.3, False, name)
        parameters.loc[name + "_sr"] = (0.5, 1e-5, 1.0, False, name)
        parameters.loc[name + "_de"] = (250, 20, 1e3, True, name)
        parameters.loc[name + "_l"] = (2, -4, 50, True, name)
        parameters.loc[name + "_m"] = (0.5, 1e-5, 0.5, False, name)
        parameters.loc[name + "_ks"] = (50, 1, 1e3, True, name)
        return parameters

    def simulate(self, prec, evap, p=None, **kwargs):
        r = self.get_recharge(prec, evap, fi=p[0], fc=p[1], sr=p[2], de=p[3],
                              l=p[4], m=p[5], ks=p[6], dt=self.dt)[0]
        return np.nan_to_num(r)

    @staticmethod
    @njit
    def get_recharge(prec, evap, fi=1.0, fc=1.0, sr=0.5, de=250.0, l=-2.0,
                     m=0.5, ks=50.0, dt=1.0):
        n = prec.size
        pe = fi * prec
        ep = fc * evap
        s = np.zeros(n, dtype=np.float64)
        s[0] = 0.5  # Set the initial system state
        r = np.zeros(n, dtype=np.float64)
        ea = np.zeros(n, dtype=np.float64)

        # Use explicit Euler scheme
        for t in range(n - 1):
            if s[t] < 0.05:
                s[t] = 0.05 * np.exp(20.0 * s[t] - 1.0)
            elif s[t] > 0.95:
                s[t] = 1 - (0.05 * np.exp(19.0 - 20.0 * s[t]))

            ea[t] = (1.0 - np.exp(-3 * s[t] / sr)) * ep[t]
            r[t] = ks * s[t] ** l * (1.0 - (1.0 - s[t] ** (1.0 / m)) ** m) ** 2
            s[t + 1] = s[t] + dt / de * (pe[t] - ea[t] - r[t])
        return r, s, ea, pe
