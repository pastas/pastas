"""This module contains the classes for recharge models.

This module contains the different classes that can be used to simulate the effect of
precipitation and evapotranspiration on groundwater levels. Depending on the
mathematical formulation this effect may be interpreted as:

1. seepage to the groundwater
2. precipitation excess,
3. groundwater recharge.

For the implementation of each model we refer to the references listed in the
documentation of each recharge model.

The classes defined here are designed to be used in conjunction with the stressmodel
"RechargeModel", which requires an instance of one of the classes defined here.

.. codeauthor:: R.A. Collenteur, University of Graz

See Also
--------
pastas.stressmodels.RechargeModel
    The recharge models listed above are provided to a RechargeModel.

Examples
--------
Using the recharge models is as follows:

>>> rch = ps.rch.FlexModel()
>>> sm = ps.RechargeModel(prec, evap, recharge=rch, rfunc=ps.Gamma(), name="rch")
>>> ml.add_stressmodel(sm)

After solving a model, the simulated recharge flux can be obtained:

>>> rch_sim = ml.get_stress("rch")
"""

from logging import getLogger

# Type Hinting
from typing import Tuple, Union

from numpy import add, exp, float64, multiply, nan_to_num, power, vstack, where, zeros
from pandas import DataFrame

from pastas.typing import ArrayLike

from .decorators import njit

logger = getLogger(__name__)


class RechargeBase:
    """Base class for classes that calculate the recharge."""

    _name = "RechargeBase"

    def __init__(self) -> None:
        self.snow = False
        self.nparam = 0

    @staticmethod
    def get_init_parameters(name: str = "recharge") -> DataFrame:
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
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        return parameters

    def simulate(self, prec, evap, p, dt=1.0, return_full=False, **kwargs):
        pass

    def to_dict(self):
        """Method to export the recharge model object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the StressModel
            object.
        """
        data = {
            "class": self._name,
        }
        return data


class Linear(RechargeBase):
    """Linear model for precipitation excess according to
    :cite:t:`von_asmuth_transfer_2002`.

    Notes
    -----
    The precipitation excess is calculated as:

    .. math::
        R = P - f * E

    """

    _name = "Linear"

    def __init__(self) -> None:
        RechargeBase.__init__(self)
        self.nparam = 1

    def get_init_parameters(self, name: str = "recharge") -> DataFrame:
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_f"] = (-1.0, -2.0, 0.0, True, name)
        return parameters

    def simulate(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, **kwargs
    ) -> ArrayLike:
        """Simulate the precipitation excess flux.

        Parameters
        ----------
        prec, evap: array_like
            array with the precipitation and evapotranspiration values. These arrays
            must be of the same length and at the same time steps.
        p: array_like
            array_like object with the values as floats representing the model
            parameters.

        Returns
        -------
        recharge: array_like
            array with the recharge series.
        """
        return add(prec, multiply(evap, p))

    def get_water_balance(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, **kwargs
    ) -> DataFrame:
        ea = multiply(evap, p)
        r = add(prec, multiply(evap, p))
        return DataFrame(data=vstack((prec, ea, -r)).T, columns=["P", "Ea", "R"])


class FlexModel(RechargeBase):
    """Recharge to the groundwater calculated according to
    :cite:t:`collenteur_estimation_2021`.

    Parameters
    ----------
    interception: bool, optional
        Use an interception reservoir in the model or not.
    snow: bool, optional
        Account for snowfall and snowmelt in the model. If True, a temperature series
        should be provided to the RechargeModel.
    gw_uptake: bool, optional
        If True, the potential evaporation that is left after evaporation from the
        interception reservoir and the root zone reservoir is subtracted from the
        recharge flux. An additional parameter can be used to scale the excess
        evaporation. Note that this is an EXPERIMENTAL FEATURE that may be removed in
        the future!

    Notes
    -----
    For a detailed description of the recharge model and parameters we refer to
    :cite:t:`collenteur_estimation_2021`. The water balance for the unsaturated zone
    reservoir is written as:

    .. math::

        \\frac{dS}{dt} = P_e - E_a - R

    where the recharge is calculated as:

    .. math::

        R = K_s \\left( \\frac{S}{S_u}\\right) ^\\gamma

    If snow=True, a snow reservoir is added on top. For a detailed description of the
    degree-day snow model and parameters we refer to :cite:t:`kavetski_model_2007`.
    The water balance for the snow reservoir is written as:

    .. math::

        \\frac{dSs}{dt} = Ps - M

    Note that the preferred unit of the precipitation and evaporation is mm/d and the
    temperature is degree celsius.

    """

    _name = "FlexModel"

    def __init__(
        self, interception: bool = True, snow: bool = False, gw_uptake: bool = False
    ):
        RechargeBase.__init__(self)
        self.snow = snow
        self.interception = interception
        self.gw_uptake = gw_uptake
        self.nparam = 5
        if self.interception:
            self.nparam += 1
        if self.gw_uptake:
            self.nparam += 1
        if self.snow:
            self.nparam += 2

    def get_init_parameters(self, name: str = "recharge") -> DataFrame:
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_srmax"] = (250.0, 1e-5, 1e3, True, name)
        parameters.loc[name + "_lp"] = (0.25, 1e-5, 1, False, name)
        parameters.loc[name + "_ks"] = (100.0, 1e-5, 1e4, True, name)
        parameters.loc[name + "_gamma"] = (2.0, 1e-5, 20.0, True, name)
        parameters.loc[name + "_kv"] = (1.0, 0.25, 2.0, False, name)
        if self.interception:
            parameters.loc[name + "_simax"] = (2.0, 0.0, 10.0, False, name)
        if self.gw_uptake:
            parameters.loc[name + "_gf"] = (1.0, 0.0, 1.0, True, name)
        if self.snow:
            parameters.loc[name + "_tt"] = (0.0, -10.0, 10.0, True, name)
            parameters.loc[name + "_k"] = (2.0, 1.0, 20.0, True, name)

        return parameters

    def simulate(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        temp: ArrayLike,
        p: ArrayLike,
        dt: float = 1.0,
        return_full: bool = False,
        **kwargs
    ) -> ArrayLike:
        """Simulate the soil water balance model.

        Parameters
        ----------
        prec: array_like
            Precipitation flux in mm/d. Must have the same length as evap.
        evap: array_like
            Potential evaporation flux in mm/d.
        temp: array_like
            Temperature in degrees Celsius.
        p: array_like
            array_like object with the values as floats representing the model
            parameters. Must be length self.nparam.
        dt: float, optional
            time step for the calculation of the recharge. Only dt=1 is possible now.
        return_full: bool
            return all fluxes and states as NumPy arrays.

        Returns
        -------
        r: array_like
            Recharge flux calculated by the model.
        """
        ep = evap * p[4]

        if self.snow:
            ss, ps, m = self.get_snow_balance(prec=prec, temp=temp, tt=p[-2], k=p[-1])
            pr = prec - ps  # Remove snowfall from precipitation
        else:
            pr = prec  # All precipitation is rainfall and melt is zero
            m = 0.0

        if self.interception:
            si, ei, pi = self.get_interception_balance(pr=pr, ep=ep, simax=p[5])
            ep = ep + ei  # Update potential evaporation after interception
            pe = pr - pi  # Update rainfall after interception
        else:
            pe = pr

        sr, r, ea, q, _ = self.get_root_zone_balance(
            pe=pe - m, ep=ep, srmax=p[0], lp=p[1], ks=p[2], gamma=p[3], dt=dt
        )

        # report big water balance errors (error > 0.1%.)
        error = (sr[0] - sr[-1] + (pe - m + r + ea + q).sum()) / (
            pe.sum() + 1e-10
        )  # avoid division by zero
        if abs(error) > 0.1:
            logger.info(
                "Water balance error: %s %% of the total pe flux. Parameters: %s",
                error.round(2),
                p.astype(float).round(2),
            )

        if self.gw_uptake:
            # Compute leftover potential evaporation
            if self.interception:
                gf = p[6]
            else:
                gf = p[5]
            eg = ep + ea  # positive flux
            r = r + gf * eg

        if return_full:
            data = (sr, r, ea, q, pe)
            if self.interception:
                data += (si, ei, pi)
            if self.snow:
                data += (ss, ps, m)
            return data
        else:
            return -r

    @staticmethod
    @njit
    def get_root_zone_balance(
        pe: ArrayLike,
        ep: ArrayLike,
        srmax: float = 250.0,
        lp: float = 0.25,
        ks: float = 100.0,
        gamma: float = 4.0,
        dt: float = 1.0,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Method to compute the water balance of the root zone reservoir.

        Parameters
        ----------
        pe: array_like
            Effective precipitation flux in mm/d.
        ep: array_like
            Potential evaporation flux in mm/d.
        srmax: float, optional
            Maximum storage capacity of the root zone.
        lp: float, optional
            Parameter determining when actual evaporation equals potential.
        ks: float, optional
            Saturated hydraulic conductivity in mm/d.
        gamma: float, optional
            Parameter determining the nonlinearity of outflow / recharge.
        dt: float, optional
            time step for the calculation of the recharge. Only dt=1 is possible now.

        Returns
        -------
        sr: array_like
            Storage in the root zone reservoir.
        r: array_like
            Recharge flux in mm/d
        ea: array_like
            Evaporation flux in mm/d. Consists of transpiration and soil evaporation.
            Does not include interception evaporation.
        q: array_like
            surface runoff flux in mm/d.
        pe: array_like
            Incoming infiltration flux in mm/d.

        """
        n = pe.size
        # Create empty arrays to store the fluxes and states
        sr = zeros(n + 1, dtype=float64)  # Root Zone Storage State
        sr[0] = 0.5 * srmax  # Set the initial system state to half-full
        ea = zeros(n, dtype=float64)  # Actual evaporation Flux
        r = zeros(n, dtype=float64)  # Recharge Flux
        q = zeros(n, dtype=float64)  # Surface runoff Flux
        lp = lp * srmax  # Do this here outside the for-loop for efficiency

        for t in range(n):
            # Make sure the solution is larger than 0.0 and smaller than sr
            if sr[t] > srmax:
                q[t] = sr[t] - srmax  # Surface runoff
                sr[t] = srmax
            elif sr[t] < 0.0:
                sr[t] = 0.0

            # Calculate evaporation from the root zone reservoir
            if sr[t] / lp < 1.0:
                ea[t] = ep[t] * sr[t] / lp
            else:
                ea[t] = ep[t]

            # Calculate the recharge flux
            r[t] = min(ks * (sr[t] / srmax) ** gamma, sr[t])
            # Update storage in the root zone
            sr[t + 1] = sr[t] + dt * (pe[t] - r[t] - ea[t])

        return sr[:-1], -r, -ea, -q, pe

    @staticmethod
    @njit
    def get_interception_balance(
        pr: ArrayLike, ep: ArrayLike, simax: float = 2.0, dt: float = 1.0
    ) -> Tuple[ArrayLike]:
        """Method to compute the water balance of the interception reservoir.

        Parameters
        ----------
        pr: array_like
            NumPy Array with rainfall in mm/day.
        ep: array_like
            NumPy Array with potential evaporation in mm/day.
        simax: float, optional
            storage capacity of the interception reservoir.
        dt: float
            time step used for computation. Only dt=1.0 is possible now.

        Returns
        -------
        si: array_like
            Interception storage.
        ei: array_like
            Interception evaporation.
        pi: array_like
            Incoming rainfall that is intercepted.

        Notes
        -----
        The water balance for the snow storage reservoir is defined as follows:

        .. math::

            \\frac{dS_i}{dt} = P_r - E_i - P_e

        where $S_i$ [L] is the interception storage, $P_r$ [L/T] is the incoming
        rainfall, $E_i$ [L/T] the interception evaporation, and $P_e$ [L/T] the
        overflow from the interception reservoir.
        """
        n = pr.size
        si = zeros(n + 1, dtype=float64)  # Interception Storage State
        pe = zeros(n, dtype=float64)  # Effective precipitation Flux
        ei = zeros(n, dtype=float64)  # Interception evaporation Flux

        for t in range(n):
            # Interception bucket
            ei[t] = min(ep[t], si[t])
            si[t + 1] = si[t] + dt * (pr[t] - ei[t])
            pe[t] = max(si[t + 1] - simax, 0.0)
            si[t + 1] = si[t + 1] - pe[t]

        pi = pr - pe  # Compute intercepted precipitation

        return si[:-1], -ei, pi

    @staticmethod
    @njit
    def get_snow_balance(
        prec: ArrayLike, temp: ArrayLike, tt: float = 0.0, k: float = 2.0
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Method to compute the water balance of the snow reservoir.

        Parameters
        ----------
        prec: array_like
            NumPy Array with precipitation in mm/day.
        temp: array_like
            NumPy Array with the mean daily temperature in degree Celsius.
        tt: float, optional
        k: float, optional

        Returns
        -------
        ss: array_like
            storage in the snow reservoir.
        ps: array_like
            snowfall flux in mm/d.
        m: array_like
            snow melt flux in mm/d.

        Notes
        -----
        The water balance from the snow reservoir is as follows:

        .. math::

            \\frac{dS_s}{dt} = P_s - M

        where $S_s$ [L] is the snow storage, $P_s$ [L/T] the snowfall, and $M$ [L/T]
        the snow melt from the snow reservoir.
        """
        n = prec.size
        # Create empty arrays to store the fluxes and states
        ss = zeros(n + 1, dtype=float64)  # Snow Storage
        ps = where(temp <= tt, prec, 0.0)  # Snowfall
        m = where(temp > tt, k * (temp - tt), 0.0)  # Potential Snow melt

        # Snow bucket
        for t in range(n):
            if temp[t] > tt:
                smoothing_factor = 1.0 - exp(-(ss[t] / 1.5))
                m[t] = min(m[t] * smoothing_factor, ss[t])
            ss[t + 1] = ss[t] + ps[t] - m[t]

        return ss[:-1], ps, -m

    def get_water_balance(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        temp: ArrayLike,
        p: ArrayLike,
        dt: float = 1.0,
        **kwargs
    ) -> DataFrame:
        data = self.simulate(
            prec=prec, evap=evap, temp=temp, p=p, dt=dt, return_full=True, **kwargs
        )

        columns = [
            "State Root zone (Sr)",
            "Recharge (R)",
            "Actual evaporation (Ea)",
            "Surface Runoff (Q)",
            "Effective precipitation (Pe)",
        ]

        if self.interception:
            columns += [
                "State Interception (Si)",
                "Interception evaporation (Ei)",
                "Intercepted precipitation (Pi)",
            ]

        if self.snow:
            columns += [
                "State Snow (Ss)",
                "Snowfall (Ps)",
                "Snowmelt (M)",
            ]

        return DataFrame(data=vstack(data).T, columns=columns)

    def check_snow_balance(self, prec: ArrayLike, temp: ArrayLike, **kwargs) -> float:
        ss, ps, m = self.get_snow_balance(prec, temp)
        error = ss[0] - ss[-1] + (ps + m).sum()
        return error

    def check_interception_balance(
        self, prec: ArrayLike, evap: ArrayLike, **kwargs
    ) -> float:
        si, ei, pi = self.get_interception_balance(prec, evap)
        error = si[0] - si[-1] + (pi + ei).sum()
        return error

    def check_root_zone_balance(
        self, prec: ArrayLike, evap: ArrayLike, **kwargs
    ) -> float:
        sr, r, ea, q, pe = self.get_root_zone_balance(prec, evap)
        error = sr[0] - sr[-1] + (r + ea + q + pe).sum()
        return error

    def to_dict(self):
        """Method to export the recharge model object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the recharge
            object.
        """
        data = {
            "class": self._name,
            "interception": self.interception,
            "snow": self.snow,
            "gw_uptake": self.gw_uptake,
        }
        return data


class Berendrecht(RechargeBase):
    """Recharge to the groundwater calculated according to
    :cite:t:`berendrecht_non-linear_2006`.

    Notes
    -----
    Note that the preferred unit of the precipitation and evaporation is mm/d. The
    water balance for the unsaturated zone reservoir is written as:

    .. math::
        \\frac{dS_e}{dt} = \\frac{1}{D_e}(f_iP - E_a - R)

    where the recharge is calculated as:

    .. math::
        R(S_e) = K_sS_e^\\lambda(1-(1-S_e^{1/m})^m)^2

    For a detailed description of the recharge model and parameters we refer to the
    original publication.

    """

    _name = "Berendrecht"

    def __init__(self) -> None:
        RechargeBase.__init__(self)
        self.nparam = 7

    def get_init_parameters(self, name: str = "recharge") -> DataFrame:
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_fi"] = (0.9, 0.7, 1.3, False, name)
        parameters.loc[name + "_fc"] = (1.0, 0.7, 1.3, False, name)
        parameters.loc[name + "_sr"] = (0.25, 1e-5, 1.0, False, name)
        parameters.loc[name + "_de"] = (250.0, 20, 1e3, True, name)
        parameters.loc[name + "_l"] = (2.0, -4, 50, True, name)
        parameters.loc[name + "_m"] = (0.5, 1e-5, 0.5, False, name)
        parameters.loc[name + "_ks"] = (100.0, 1, 1e4, True, name)
        return parameters

    def simulate(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        p: ArrayLike,
        dt: ArrayLike = 1.0,
        return_full: bool = False,
        **kwargs
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]]:
        """Simulate the recharge flux.

        Parameters
        ----------
        prec: array_like
            Precipitation flux in mm/d. Has to have the same length as evap.
        evap: array_like
            Potential evapotranspiration flux in mm/d.
        p: array_like
            array_like object with the values as floats representing the model
            parameters.
        dt: float, optional
            time step for the calculation of the recharge. Only dt=1 is possible now.
        return_full: bool
            return all fluxes and states as NumPy arrays.

        Returns
        -------
        r: array_like or list of array_like
            Recharge flux calculated by the model is the argument full_output is
            False, otherwise a list with all fluxes and states.
        """
        r, s, ea, pe = self.get_recharge(
            prec,
            evap,
            fi=p[0],
            fc=p[1],
            sr=p[2],
            de=p[3],
            l=p[4],
            m=p[5],
            ks=p[6],
            dt=dt,
        )
        if return_full:
            return r, s, ea, pe
        else:
            return nan_to_num(r)

    @staticmethod
    @njit
    def get_recharge(
        prec: ArrayLike,
        evap: ArrayLike,
        fi: float = 1.0,
        fc: float = 1.0,
        sr: float = 0.5,
        de: float = 250.0,
        l: float = -2.0,
        m: float = 0.5,
        ks: float = 50.0,
        dt: float = 1.0,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Internal method used for the recharge calculation.

        Notes
        -----
        If Numba is available, this method is significantly faster.
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

    def get_water_balance(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, dt: float = 1.0, **kwargs
    ) -> DataFrame:
        r, s, ea, pe = self.simulate(prec, evap, p=p, dt=dt, return_full=True, **kwargs)
        s = s * p[3]  # Because S is computed dimensionless in this model
        data = DataFrame(data=vstack((s, pe, ea, r)).T, columns=["S", "Pe", "Ea", "R"])
        return data


class Peterson(RechargeBase):
    """Recharge to the groundwater calculated based on
    :cite:t:`peterson_nonlinear_2014`.

    The water balance for the unsaturated zone reservoir is written as:

    .. math::
        \\frac{dS}{dt} = P_e - E_a - R

    where the fluxes $P_e$, $E_a$ and $R$ are calculated as:

    .. math::
        P_e = P \\left(1 - \\frac{S}{\\hat{S_{cap}}}\\right)^\\alpha

    .. math::
        E_a = E_p \\left(\\frac{S}{\\hat{S_{cap}}}\\right)^\\gamma

    .. math::
        R = \\hat{k_{sat}}\\left(\\frac{S}{\\hat{S_{cap}}}\\right)^{\\hat{\\beta}}

    with the parameters:

    .. math::
        \\hat{S_{cap}} = 10^{S_{cap}}; \\hat{k_{sat}} = 10^{k_{sat}}; \\hat{\\beta} =
        10^{\\beta}

    Note that the method currently uses forward Euler method to solve the ODE so
    significant water balance errors can occur.

    """

    _name = "Peterson"

    def __init__(self) -> None:
        RechargeBase.__init__(self)
        self.nparam = 5

    def get_init_parameters(self, name: str = "recharge") -> DataFrame:
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_scap"] = (1.5, 0.5, 3.0, True, name)
        parameters.loc[name + "_alpha"] = (1.0, 0.0, 1.5, True, name)
        parameters.loc[name + "_ksat"] = (1.0, 0.0, 3.0, True, name)
        parameters.loc[name + "_beta"] = (0.5, 0.0, 1.5, True, name)
        parameters.loc[name + "_gamma"] = (1.0, 0.0, 2.0, True, name)
        return parameters

    def simulate(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        p: ArrayLike,
        dt: float = 1.0,
        return_full: bool = False,
        **kwargs
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]]:
        """Simulate the recharge flux.

        Parameters
        ----------
        prec: array_like
            Precipitation flux in mm/d. Must have the same length as evap.
        evap: array_like
            Potential evapotranspiration flux in mm/d.
        p: array_like
            array_like object with the values as floats representing the model
            parameters.
        dt: float, optional
            time step for the calculation of the recharge.
        return_full: bool
            return all fluxes and states as NumPy arrays.

        Returns
        -------
        r: array_like or list of array_like
            Recharge flux calculated by the model is the argument full_output is
            False, otherwise a list with all fluxes and states.

        """
        r, s, ea, pe = self.get_recharge(
            prec, evap, scap=p[0], alpha=p[1], ksat=p[2], beta=p[3], gamma=p[4], dt=dt
        )
        if return_full:
            return r, s, ea, pe
        else:
            return nan_to_num(r)

    @staticmethod
    @njit
    def get_recharge(
        prec: ArrayLike,
        evap: ArrayLike,
        scap: float = 1.0,
        alpha: float = 1.0,
        ksat: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0,
        dt: float = 1.0,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Internal method used for the recharge calculation.

        Notes
        -----
        If Numba is available, this method is significantly faster.
        """
        n = len(prec)
        # Create an empty arrays to store the fluxes and states
        pe = zeros(n, dtype=float64)  # Effective precipitation flux
        sm = zeros(n + 1, dtype=float64)  # Root zone storage state
        r = zeros(n, dtype=float64)  # Recharge flux
        ea = zeros(n, dtype=float64)  # Actual evaporation flux
        # Update params
        smsc = power(10, scap)
        ksat = power(10, ksat)
        beta = power(10, beta)
        # Set the initial system state
        sm[0] = smsc / 2

        for t in range(n):
            sm_frac = sm[t] / smsc
            pe[t] = prec[t] * power(1 - sm_frac, alpha)
            ea[t] = max(sm[t + 1], evap[t] * power(sm_frac, gamma))
            r[t] = max(sm[t + 1], ksat * power(sm_frac, beta))
            sm[t + 1] = min(smsc, max(0.0, sm[t] + (pe[t] - ea[t] - r[t]) * dt))
        return r, sm[1:], ea, pe

    def get_water_balance(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, dt: float = 1.0, **kwargs
    ) -> DataFrame:
        r, s, ea, pe = self.simulate(prec, evap, p=p, dt=dt, return_full=True, **kwargs)
        data = DataFrame(data=vstack((s, pe, ea, r)).T, columns=["S", "Pe", "Ea", "R"])
        return data
