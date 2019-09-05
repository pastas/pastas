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

import numba
import numpy as np
import pandas as pd


class RechargeBase:
    """Base class for classes that calculate the recharge.

    """

    def __init__(self):
        self.temp = False
        self.nparam = 0

    def get_init_parameters(self, name="recharge"):
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

    def get_init_parameters(self, name="recharge"):
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


class Percolation:
    """
    Percolation flow recharge model

    Other water balance for the root zone s calculated as:

    dS/dt = Pe - Kp * (Sr / Srmax)**Gamma - Epu * min(1, Sr / (0.5 * Srmax))

    """
    _name = "Percolation"

    def __init__(self):
        self.nparam = 4
        self.dt = 1
        self.solver = 0
        self.temp = False

    def get_init_parameters(self, name="recharge"):
        parameters = pd.DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_srmax"] = (0.1, 0.0, np.nan, True, name)
        parameters.loc[name + "_kp"] = (0.05, 0.0, np.nan, True, name)
        parameters.loc[name + "_gamma"] = (2.0, 0.0, np.nan, True, name)
        parameters.loc[name + "_imax"] = (0.001, 0.0, np.nan, False, name)
        return parameters

    def simulate(self, prec, evap, p=None, **kwargs):
        recharge = perc(prec, evap, p[0], p[1], p[2], p[3], self.dt)[0]
        return recharge


class Preferential:
    """Preferential flow recharge model

    Other water balance for the root zone s calculated as:

    dS/dt = Pe - Kp * (Sr / Srmax)**Gamma - Epu * min(1, Sr / (0.5 * Srmax))

    """
    _name = "Preferential"

    def __init__(self):
        self.nparam = 3
        self.dt = 1
        self.solver = 0
        self.temp = False

    def get_init_parameters(self, name="recharge"):
        parameters = pd.DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_srmax"] = (0.1, 0.0, np.nan, True, name)
        parameters.loc[name + "_beta"] = (2.0, 0.0, np.nan, True, name)
        parameters.loc[name + "_imax"] = (0.001, 0.0, np.nan, False, name)
        return parameters

    def simulate(self, prec, evap, p=None, **kwargs):
        recharge = pref(prec, evap, p[0], p[1], p[2], self.dt)[0]
        return recharge


class Combination:
    """
    Percolation and preferential flow recharge model

    Other water balance for the root zone s calculated as:

    dS/dt = Pe - Kp * (Sr / Srmax)**Gamma - Epu * min(1, Sr / (0.5 * Srmax))

    """
    _name = "Combination"

    def __init__(self):
        self.nparam = 5
        self.dt = 1
        self.solver = 0
        self.temp = False

    def get_init_parameters(self, name="recharge"):
        parameters = pd.DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_srmax"] = (0.05, 0.0, np.nan, True, name)
        parameters.loc[name + "_kp"] = (0.05, 0.0, np.nan, True, name)
        parameters.loc[name + "_beta"] = (4.0, 0.0, np.nan, True, name)
        parameters.loc[name + "_gamma"] = (1.0, 0.0, np.nan, True, name)
        parameters.loc[name + "_imax"] = (0.001, 0.0, np.nan, False, name)
        return parameters

    def simulate(self, prec, evap, p=None, **kwargs):
        rf, rs = comb(prec, evap, p[0], p[1], p[2], p[3], p[4], self.dt)[0:2]
        return rf + rs


@numba.jit(nopython=True)
def perc(prec, evap, srmax=0.1, kp=0.05, gamma=2.0, imax=0.001, dt=1.0):
    n = prec.size
    # Create an empty array to store the soil state in
    S = np.zeros(n, dtype=np.float64)
    S[0] = 0.5 * srmax  # Set the initial system state
    Si = np.zeros(n, dtype=np.float64)
    Pe = np.zeros(n, dtype=np.float64)
    Ei = np.zeros(n, dtype=np.float64)
    Epu = np.zeros(n, dtype=np.float64)
    Ea = np.zeros(n, dtype=np.float64)

    for t in range(1, n):
        Si[t] = Si[t - 1] + prec[t]  # Fill interception bucket with new rain
        Pe[t] = max(0.0, Si[t] - imax)  # Calculate effective precipitation
        Si[t] = Si[t] - Pe[t]
        Ei[t] = min(Si[t], evap[t])  # Evaporation from interception
        Si[t] = Si[t] - Ei[t]  # Update interception state
        Epu[t] = evap[t] - Ei[t]  # Update potential evapotranspiration

        # Use explicit Euler scheme
        S[t] = max(0.0, S[t - 1] + dt * (
                Pe[t - 1] - kp * (S[t - 1] / srmax) ** gamma - Epu[
            t - 1] * min(1.0, S[t - 1] / (0.5 * srmax))))

        # Make sure the solution is larger then 0.0 and smaller than srmax
        S[t] = min(srmax, max(0.0, S[t]))
        Ea[t] = Epu[t] * min(1.0, (S[t] / (0.5 * srmax)))

    R = np.zeros(n, dtype=np.float64)
    R[1:] = kp * dt * 0.5 * \
            (S[:-1] ** gamma + S[1:] ** gamma) / srmax ** gamma

    return R, S, Ea, Ei


@numba.jit()
def pref(prec, evap, srmax=0.1, beta=2.0, imax=0.001, dt=1.0):
    """
    In this section the preferential flow model is defined.
    dS/ Dt = Pe[t] * (1 - (Sr[t] / Srmax)**Beta)- Epu * min(1, Sr/0.5Srmax)
    """
    n = prec.size

    # Create an empty array to store the soil state in
    S = np.zeros(n, dtype=np.float64)
    S[0] = 1 * srmax  # Set the initial system state
    Si = np.zeros(n, dtype=np.float64)
    Pe = np.zeros(n, dtype=np.float64)
    Ei = np.zeros(n, dtype=np.float64)
    Epu = np.zeros(n, dtype=np.float64)
    Ea = np.zeros(n, dtype=np.float64)
    R = np.zeros(n, dtype=np.float64)

    for t in range(1, n):
        Si[t] = Si[t - 1] + prec[t]  # Fill interception bucket with new rain
        Pe[t] = max(0.0, Si[t] - imax)  # Calculate effective precipitation
        Si[t] = Si[t] - Pe[t]
        Ei[t] = min(Si[t], evap[t])  # Evaporation from interception
        Si[t] = Si[t] - Ei[t]  # Update interception state
        Epu[t] = evap[t] - Ei[t]  # Update potential evapotranspiration

        # Use explicit Euler scheme 
        S[t] = S[t - 1] + dt * (
                Pe[t - 1] * (1 - (S[t - 1] / srmax) ** beta) - Epu[t - 1]
                * min(1.0, S[t - 1] / (0.5 * srmax)))

        # Make sure the solution is larger then 0.0 and smaller than Srmax
        S[t] = min(srmax, max(0.0, S[t]))
        Ea[t] = Epu[t] * min(1.0, S[t] / (0.5 * srmax))

    R[1:] = Pe[1:] * dt * 0.5 * (
            (S[:-1] ** beta + S[1:] ** beta) / srmax ** beta)

    return R, S, Ea, Ei


@numba.jit()
def comb(prec, evap, srmax=0.05, kp=0.05, beta=2.0, gamma=2.0, imax=0.001,
         dt=1.0):
    n = prec.size

    # Create an empty array to store the soil state in
    S = np.zeros(n, dtype=np.float64)
    S[0] = 0.5 * srmax  # Set the initial system state
    Si = np.zeros(n, dtype=np.float64)
    Pe = np.zeros(n, dtype=np.float64)
    Ei = np.zeros(n, dtype=np.float64)
    Epu = np.zeros(n, dtype=np.float64)
    Ea = np.zeros(n, dtype=np.float64)

    for t in range(1, n):
        Si[t] = Si[t - 1] + prec[t]  # Fill interception bucket with new rain
        Pe[t] = max(0.0, Si[t] - imax)  # Calculate effective precipitation
        Si[t] = Si[t] - Pe[t]
        Ei[t] = min(Si[t], evap[t])  # Evaporation from interception
        Si[t] = Si[t] - Ei[t]  # Update interception state
        Epu[t] = evap[t] - Ei[t]  # Update potential evapotranspiration

        # Use explicit Euler scheme 
        S[t] = max(0.0,
                   S[t - 1] + dt * (Pe[t - 1] *
                                    (1 - (S[t - 1] / srmax) ** beta) - kp *
                                    (S[t - 1] / srmax) ** gamma - Epu[t - 1]
                                    * min(1.0, S[t - 1] / (0.5 * srmax))))

        # Make sure the solution is larger then 0.0 and smaller than Srmax
        S[t] = min(srmax, max(0.0, S[t]))
        Ea[t] = Epu[t] * min(1.0, S[t] / (0.5 * srmax))

    # Percolation
    Rs = np.zeros(n, dtype=np.float64)
    Rs[1:] = kp * dt * 0.5 * \
             ((S[:-1] ** gamma + S[1:] ** gamma) / srmax ** gamma)

    # Preferential
    Rf = np.zeros(n, dtype=np.float64)
    Rf[1:] = Pe[1:] * dt * 0.5 * \
             ((S[:-1] ** beta + S[1:] ** beta) / srmax ** beta)

    return Rs, Rf, S, Ea, Ei
