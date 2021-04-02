"""This module contains the classes for recharge models.

This module contains the different classes that can be used to simulate the
effect of precipitation and evapotranspiration on groundwater levels.
Depending on the mathematical formulation this effect may be interpreted as:

1. seepage to the groundwater
2. precipitation excess,
3. groundwater recharge.

For the implementation of each model we refer to the references listed in
the documentation of each recharge model.

The classes defined here are designed to be used in conjunction with the
stressmodel "RechargeModel", which requires an instance of one of the
classes defined here.

.. codeauthor:: R.A. Collenteur, University of Graz

See Also
--------
pastas.stressmodels.RechargeModel
    The recharge models listed above are provided to a RechargeModel

Examples
--------
Using the recharge models is as follows:

>>> rch = ps.rch.FlexModel()
>>> sm = ps.RechargeModel(prec, evap, recharge=rch, rfunc=ps.Gamma, name="rch")
>>> ml.add_stressmodel(sm)

After solving a model, the simulated recharge flux can be obtained:

>>> rch_sim = ml.get_stress("rch")

"""

from numpy import add, float64, multiply, exp, zeros, nan_to_num, vstack
from pandas import DataFrame

from pastas.decorators import njit


class RechargeBase:
    """Base class for classes that calculate the recharge.

    """

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


class Linear(RechargeBase):
    """Linear model for precipitation excess according to [asmuth_2002]_.

    Notes
    -----
    The precipitation excess is calculated as:

    .. math::
        R = P - f * E

    References
    ----------
    .. [asmuth_2002] von Asmuth, J., Bierkens, M., and Maas, K. (2002) Transfer
       function-noise modeling in continuous time using predefined impulse
       response functions, Water Resources Research, 38, 23–1–23–12.

    """
    _name = "Linear"

    def __init__(self):
        RechargeBase.__init__(self)
        self.nparam = 1

    def get_init_parameters(self, name="recharge"):
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
        p: array_like
            array_like object with the values as floats representing the
            model parameters.

        Returns
        -------
        recharge: array_like
            array with the recharge series.

        """
        return add(prec, multiply(evap, p))

    def get_water_balance(self, prec, evap, p, **kwargs):
        ea = multiply(evap, p)
        r = add(prec, multiply(evap, p))
        data = DataFrame(data=vstack((prec, ea, -r)).T,
                         columns=["P", "Ea", "R"])
        return data


class FlexModel(RechargeBase):
    """Recharge to the groundwater calculate according to [collenteur_2020]_.

    Notes
    -----
    Note that the preferred unit of the precipitation and evaporation is mm/d.

    The water balance for the unsaturated zone reservoir is written as:

    .. math::

        \\frac{dS}{dt} = P_e - E_a - R

    where the recharge is calculated as:

    .. math::

        R = K_s \\left( \\frac{S}{S_u}\\right) ^\\gamma

    For a detailed description of the recharge model and parameters we refer
    to Collenteur et al. (in review).

    References
    ----------
    .. [collenteur_2020] Collenteur, R.A., Bakker, M., Klammler, G., & Birk,
       S. (in Review) Estimating groundwater recharge from groundwater
       levels using non-linear transfer function noise models and comparison to
       lysimeter data. https://doi.org/10.5194/hess-2020-392

    """
    _name = "FlexModel"

    def __init__(self):
        RechargeBase.__init__(self)
        self.nparam = 6

    def get_init_parameters(self, name="recharge"):
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_srmax"] = (250.0, 1e-5, 1e3, True, name)
        parameters.loc[name + "_lp"] = (0.25, 1e-5, 1, False, name)
        parameters.loc[name + "_ks"] = (100.0, 1, 1e4, True, name)
        parameters.loc[name + "_gamma"] = (4.0, 1e-5, 50.0, True, name)
        parameters.loc[name + "_simax"] = (2.0, 1e-5, 10.0, False, name)
        parameters.loc[name + "_kv"] = (1.0, 0.25, 2.0, False, name)
        return parameters

    def simulate(self, prec, evap, p, dt=1.0, **kwargs):
        """Simulate the recharge flux.

        Parameters
        ----------
        prec: numpy.array
            Precipitation flux in mm/d. Has to have the same length as evap.
        evap: numpy.array
            Potential evaporation flux in mm/d.
        p: array_like
            array_like object with the values as floats representing the
            model parameters.
        dt: float, optional
            time step for the calculation of the recharge. Only dt=1 is
            possible now.

        Returns
        -------
        r: numpy.array
            Recharge flux calculated by the model.

        """
        r = self.get_recharge(prec, evap, srmax=p[0], lp=p[1], ks=p[2],
                              gamma=p[3], simax=p[4], kv=p[5], dt=dt)[0]
        return r

    @staticmethod
    @njit
    def get_recharge(prec, evap, srmax=250.0, lp=0.25, ks=100.0, gamma=4.0,
                     simax=2.0, kv=1.0, dt=1.0):
        """
        Internal method used for the recharge calculation. If Numba is
        available, this method is significantly faster.

        """
        n = prec.size
        evap = evap * kv  # Multiply by crop factor
        # Create empty arrays to store the fluxes and states
        su = zeros(n, dtype=float64)  # Root Zone Storage State
        su[0] = 0.5 * srmax  # Set the initial system state to half-full
        ea = zeros(n, dtype=float64)  # Actual evaporation Flux
        r = zeros(n, dtype=float64)  # Recharge Flux
        si = zeros(n, dtype=float64)  # Interception Storage State
        pe = zeros(n, dtype=float64)  # Effective precipitation Flux
        ei = zeros(n, dtype=float64)  # Interception evaporation Flux
        ep = zeros(n, dtype=float64)  # Updated evaporation Flux
        lp = lp * srmax  # Do this here outside the for-loop for efficiency

        for t in range(n - 1):
            # Interception bucket
            pe[t] = max(prec[t] - simax + si[t], 0.0)
            ei[t] = min(evap[t], si[t])
            ep[t] = evap[t] - ei[t]
            si[t + 1] = si[t] + dt * (prec[t] - pe[t] - ei[t])

            # Make sure the solution is larger then 0.0 and smaller than su
            if su[t] > srmax:
                su[t] = srmax
            elif su[t] < 0.0:
                su[t] = 0.0

            # Calculate actual evapotranspiration
            if su[t] / lp < 1.0:
                ea[t] = ep[t] * su[t] / lp
            else:
                ea[t] = ep[t]

            # Calculate the recharge flux
            r[t] = ks * (su[t] / srmax) ** gamma
            # Calculate state of the root zone storage
            su[t + 1] = su[t] + dt * (pe[t] - r[t] - ea[t])

        return r, ea, ei, pe, su, si

    def get_water_balance(self, prec, evap, p, dt=1.0, **kwargs):
        r, ea, ei, pe, sr, si = self.get_recharge(prec, evap, srmax=p[0],
                                                  lp=p[1], ks=p[2],
                                                  gamma=p[3], simax=p[4],
                                                  kv=p[5], dt=dt)

        data = DataFrame(data=vstack((si, -ei, sr, pe, -ea, -r)).T,
                         columns=["Si", "Ei", "Sr", "Pe", "Ea", "R"])
        return data


class Berendrecht(RechargeBase):
    """Recharge to the groundwater calculated according to [berendrecht_2006]_.

    Notes
    -----
    Note that the preferred unit of the precipitation and evaporation is
    mm/d. The waterbalance for the unsaturated zone reservoir is written as:

    .. math::
        \\frac{dS_e}{dt} = \\frac{1}{D_e}(f_iP - E_a - R)

    where the recharge is calculated as:

    .. math::
        R(S_e) = K_sS_e^\\lambda(1-(1-S_e^{1/m})^m)^2

    For a detailed description of the recharge model and parameters we refer
    to the original publication.

    References
    ----------
    .. [berendrecht_2006] Berendrecht, W. L., Heemink, A. W., van Geer, F. C.,
       and Gehrels, J. C. (2006) A non-linear state space approach to model
       groundwater fluctuations, Advances in Water Resources, 29, 959–973.

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
        parameters.loc[name + "_de"] = (250.0, 20, 1e3, True, name)
        parameters.loc[name + "_l"] = (2.0, -4, 50, True, name)
        parameters.loc[name + "_m"] = (0.5, 1e-5, 0.5, False, name)
        parameters.loc[name + "_ks"] = (100.0, 1, 1e4, True, name)
        return parameters

    def simulate(self, prec, evap, p, dt=1.0, **kwargs):
        """Simulate the recharge flux.

        Parameters
        ----------
        prec: numpy.array
            Precipitation flux in mm/d. Has to have the same length as evap.
        evap: numpy.array
            Potential evapotranspiration flux in mm/d.
        p: array_like
            array_like object with the values as floats representing the
            model parameters.
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
        pe = fi * prec  # Effective precipitation flux
        ep = fc * evap  # Potential evaporation flux
        s = zeros(n, dtype=float64)  # Root zone storage state
        s[0] = 0.5  # Set the initial system state
        r = zeros(n, dtype=float64)  # Recharge flux
        ea = zeros(n, dtype=float64)  # Actual evaporation flux

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

    def get_water_balance(self, prec, evap, p, dt=1.0, **kwargs):
        r, s, ea, pe = self.get_recharge(prec, evap, fi=p[0], fc=p[1],
                                         sr=p[2], de=p[3], l=p[4], m=p[5],
                                         ks=p[6], dt=dt)
        s = s * p[3]  # Because S is computed dimensionless in this model
        data = DataFrame(data=vstack((s, pe, -ea, -r)).T,
                         columns=["S", "Pe", "Ea", "R"])
        return data
