"""Contains the classes for the different models that are available to
calculate the recharge from precipitation and evaporation data.

This module contains the different classes that can be used to simulate the
effect of precipitation and evapotranspiration on groundwater levels.
Depending on the mathematical formulation this effect may be interpreted as:
1) seepage to the groundwater 2) precipitation excess, 3) groundwater
recharge. For the implementation of each model we refer to the references
listed in the documentation of each recharge model.

The classes defined here are designed to be used in conjunction with the
stressmodel "RechargeModel", which requires an instance of one of the
classes defined here.

.. codeauthor:: R.A. Collenteur, University of Graz

Supported Recharge models
-------------------------
The following recharge models are currently supported and tested:

.. autosummary::
    :nosignatures:
    :toctree: ./generated

    Linear

"""

from numpy import add, float64, multiply, exp, zeros, nan_to_num
from pandas import DataFrame

from ..decorators import njit


class RechargeBase:
    """Base class for classes that calculate the recharge.

    """

    def __init__(self):
        self.temp = False
        self.nparam = 0

    @staticmethod
    def get_init_parameters(name="rch"):
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
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        return parameters

    def simulate(self, prec, evap, p, **kwargs):
        pass


class Linear(RechargeBase):
    """Linear model for precipitation excess according to [1]_.

    Notes
    -----
    The precipitation excess is calculated as:

    .. math::

        R = P - f * E

    References
    ----------
    .. [1] von Asmuth, J., Bierkens, M., and Maas, K. (2002) Transfer function-noise modeling in continuous time using predefined impulse response functions, Water Resources Research, 38, 23–1–23–12.

    """
    _name = "Linear"

    def __init__(self):
        RechargeBase.__init__(self)
        self.nparam = 1

    def get_init_parameters(self, name="rch"):
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_f"] = (-1.0, -2.0, 0.0, True, name)
        return parameters

    def simulate(self, prec, evap, p, **kwargs):
        """Simulate the precipitation excess flux.

        Parameters
        ----------
        prec, evap: array_like
            array with the precipitation and evapotranspiration values. These
            arrays must be of the same length and at the same time steps.
        p: float
            parameter value used in recharge calculation.

        Returns
        -------
        recharge: array_like
            array with the recharge series.

        """
        return add(prec, multiply(evap, p))


class FlexModel(RechargeBase):
    """
    Recharge to the groundwater calculate according to [2]_.

    Notes
    -----
    Note that the preferred unit of the precipitation and
    evapotranspiration is mm/d.

    The waterbalance for the unsaturated zone reservoir is written as:

    .. math::

        \\frac{dS}{dt} = P_e - E_a - R

    where the recharge is calculated as:

    .. math::

        R = K_s \\left( \\frac{S}{S_u}\\right) ^\\gamma

    For a detailed description of the recharge model and parameters we refer
    to the following publication [2]_.

    References
    ----------
    .. [2] Collenteur, R.A., Bakker, M., Birk, S. (in Prep.) Estimating groundwater recharge using non-linear transfer function noise models.

    """
    _name = "FlexModel"

    def __init__(self):
        RechargeBase.__init__(self)
        self.nparam = 5

    def get_init_parameters(self, name="rch"):
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_su"] = (150.0, 1e-5, 1e3, True, name)
        parameters.loc[name + "_lp"] = (0.25, 1e-5, 1, False, name)
        parameters.loc[name + "_ks"] = (50.0, 1, 1e3, True, name)
        parameters.loc[name + "_gamma"] = (4.0, 1e-5, 50.0, True, name)
        parameters.loc[name + "_si"] = (2.0, 1e-5, 10.0, False, name)
        return parameters

    def simulate(self, prec, evap, p, dt=1.0, **kwargs):
        """Simulate the recharge flux.

        Parameters
        ----------
        prec: numpy.array
            Precipitation flux in mm/d. Has to have the same length as evap.
        evap: numpy.array
            Potential evapotranspiration flux in mm/d.
        p: numpy.array
            numpy array with the parameter values.
        dt: float, optional
            time step for the calculation of the recharge. Only dt=1 is
            possible now.

        Returns
        -------
        r: numpy.array
            Recharge flux calculated by the model.

        """
        r = self.get_recharge(prec, evap, su=p[0], lp=p[1], ks=p[2],
                              gamma=p[3], si=p[4], dt=dt)[0]
        return r

    @staticmethod
    @njit
    def get_recharge(prec, evap, su=250.0, lp=0.5, ks=50.0, gamma=4.0, si=2.0,
                     dt=1.0):
        """
        Internal method used for the recharge calculation. If Numba is
        available, this method is significantly faster.

        """
        n = prec.size
        # Create an empty arrays to store the fluxes and states
        s = zeros(n, dtype=float64)
        s[0] = 0.5 * su  # Set the initial system state to half-full
        ea = zeros(n, dtype=float64)
        r = zeros(n, dtype=float64)
        i = zeros(n, dtype=float64)
        pe = zeros(n, dtype=float64)
        ei = zeros(n, dtype=float64)
        ep = zeros(n, dtype=float64)
        lp = lp * su  # Do this here outside the for-loop for efficiency

        for t in range(n - 1):
            # Interception bucket
            pe[t] = max(prec[t] - si + i[t], 0.0)  # Effective precipitation
            ei[t] = min(evap[t], i[t])  # Interception evaporation
            ep[t] = evap[t] - ei[t]  # Leftover potential evapotranspiration
            i[t + 1] = i[t] + dt * (prec[t] - pe[t] - ei[t])

            # Make sure the solution is larger then 0.0 and smaller than su
            if s[t] > su:
                s[t] = su
            elif s[t] < 0.0:
                s[t] = 0.0

            # Calculate actual evapotranspiration
            if s[t] / lp < 1.0:
                ea[t] = ep[t] * s[t] / lp
            else:
                ea[t] = ep[t]

            # Calculate the recharge flux
            r[t] = ks * (s[t] / su) ** gamma
            # Make sure the solution is larger then 0.0 and smaller than sr
            s[t + 1] = s[t] + dt * (pe[t] - r[t] - ea[t])

        return r, s, ea, ei, prec


class Berendrecht(RechargeBase):
    """Recharge to the groundwater calculated according to [3]_ and [2]_.

    Notes
    -----
    Note that the preferred unit of the precipitation and evapotranspiration
    is mm/d. The waterbalance for the unsaturated zone reservoir is written as:

    .. math::

        \\frac{dS_e}{dt} = \\frac{1}{D_e}(f_iP - E_a - R)

    where the recharge is calculated as:

    .. math::

        R(S_e) = K_sS_e^\\lambda(1-(1-S_e^{1/m})^m)^2

    For a detailed description of the recharge model and parameters we refer
    to the publications [3]_ and [2]_.

    References
    ----------
    .. [3] Berendrecht, W. L., Heemink, A. W., van Geer, F. C., and Gehrels, J. C. (2006) A non-linear state space approach to model groundwater fluctuations, Advances in Water Resources, 29, 959–973.

    """
    _name = "Berendrecht"

    def __init__(self):
        RechargeBase.__init__(self)
        self.nparam = 7

    def get_init_parameters(self, name="recharge"):
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_fi"] = (0.9, 0.7, 1.3, False, name)
        parameters.loc[name + "_fc"] = (1.0, 0.7, 1.3, False, name)
        parameters.loc[name + "_sr"] = (0.25, 1e-5, 1.0, False, name)
        parameters.loc[name + "_de"] = (250, 20, 1e3, True, name)
        parameters.loc[name + "_l"] = (2, -4, 50, True, name)
        parameters.loc[name + "_m"] = (0.5, 1e-5, 0.5, False, name)
        parameters.loc[name + "_ks"] = (50, 1, 1e3, True, name)
        return parameters

    def simulate(self, prec, evap, p, dt=1.0, **kwargs):
        """Simulate the recharge flux.

        Parameters
        ----------
        prec: numpy.array
            Precipitation flux in mm/d. Has to have the same length as evap.
        evap: numpy.array
            Potential evapotranspiration flux in mm/d.
        p: numpy.array
            numpy array with the parameter values.
        dt: float, optional
            time step for the calculation of the recharge. Only dt=1 is
            possible now.

        Returns
        -------
        r: numpy.array
            Recharge flux calculated by the model.

        """
        r = self.get_recharge(prec, evap, fi=p[0], fc=p[1], sr=p[2], de=p[3],
                              l=p[4], m=p[5], ks=p[6], dt=dt)[0]
        return nan_to_num(r)

    @staticmethod
    @njit
    def get_recharge(prec, evap, fi=1.0, fc=1.0, sr=0.5, de=250.0, l=-2.0,
                     m=0.5, ks=50.0, dt=1.0):
        """
        Internal method used for the recharge calculation. If Numba is
        available, this method is significantly faster.

        """
        n = prec.size
        # Create an empty arrays to store the fluxes and states
        pe = fi * prec
        ep = fc * evap
        s = zeros(n, dtype=float64)
        s[0] = 0.5  # Set the initial system state
        r = zeros(n, dtype=float64)
        ea = zeros(n, dtype=float64)

        for t in range(n - 1):
            # Make sure the reservoir is not too full or empty.
            if s[t] < 0.05:
                s[t] = 0.05 * exp(20.0 * s[t] - 1.0)
            elif s[t] > 0.95:
                s[t] = 1 - (0.05 * exp(19.0 - 20.0 * s[t]))

            # Calculate the actual evaporation
            ea[t] = (1.0 - exp(-3 * s[t] / sr)) * ep[t]

            # Calculate the recharge flux
            r[t] = ks * s[t] ** l * (1.0 - (1.0 - s[t] ** (1.0 / m)) ** m) ** 2

            # Calculate the
            s[t + 1] = s[t] + dt / de * (pe[t] - ea[t] - r[t])
        return r, s, ea, pe
