"""Module containing the classes for recharge models.

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

See Also
--------
pastas.stressmodels.RechargeModel
    The recharge models listed above are provided to a RechargeModel.

Examples
--------
Use recharge models with a RechargeModel stress model::

    rch = ps.rch.FlexModel()
    sm = ps.RechargeModel(prec, evap, recharge=rch, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)

After solving a model, the simulated recharge flux can be obtained::

    rch_sim = ml.get_stress("rch")
"""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any

import numpy as np
from numpy import (
    add,
    complex128,
    exp,
    multiply,
    nan_to_num,
    power,
    vstack,
    where,
    zeros,
)
from pandas import DataFrame

from pastas.typing import ArrayLike

from .decorators import njit

logger = getLogger(__name__)


class RechargeBase(ABC):
    """Base class for classes that calculate the recharge."""

    def __init__(self) -> None:
        pass

    @property
    def _name(self) -> str:
        """Name of the recharge model."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def nparam(self) -> int:
        """Number of parameters of the recharge model."""

    @abstractmethod
    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters and bounds for the recharge model."""

    @abstractmethod
    def simulate(self, *args, **kwargs) -> ArrayLike | tuple[ArrayLike, ...]:
        """Simulate recharge from precipitation and evaporation inputs."""

    def to_dict(self) -> dict[str, Any]:
        """Export the recharge model object to a dictionary."""
        return {"class": self._name}


class Linear(RechargeBase):
    r"""Linear recharge model using scaled precipitation excess.

    According to :cite:t:`von_asmuth_transfer_2002`.

    Notes
    -----
    The precipitation excess is calculated as:

    .. math::

        R = P - f * E

    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def nparam(self) -> int:
        """Number of parameters of the Linear recharge model."""
        return 1

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters and bounds for the Linear recharge model."""
        parameters = DataFrame(
            [(-1.0, -2.0, 0.0, True, name)],
            columns=["initial", "pmin", "pmax", "vary", "name"],
            index=[name + "_f"],
        )

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
        """Get the water balance for the linear recharge model."""
        ea = multiply(evap, p)
        r = add(prec, multiply(evap, p))
        return DataFrame(data=vstack((prec, ea, -r)).T, columns=["P", "Ea", "R"])

    def to_dict(self) -> dict[str, Any]:
        """Export the recharge model object to a dictionary."""
        return super().to_dict()


class FlexModel(RechargeBase):
    r"""Nonlinear recharge to the groundwater.

    Calculated according to :cite:t:`collenteur_estimation_2021`.

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

    def __init__(
        self, interception: bool = True, snow: bool = False, gw_uptake: bool = False
    ):
        super().__init__()
        self.snow = snow
        self.interception = interception
        self.gw_uptake = gw_uptake

    @property
    def nparam(self) -> int:
        """Number of parameters of the FlexModel recharge model."""
        _nparam = 5
        if self.interception:
            _nparam += 1
        if self.gw_uptake:
            _nparam += 1
        if self.snow:
            _nparam += 2
        return _nparam

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters and bounds for the FlexModel recharge model."""
        parameters = DataFrame(
            [
                (250.0, 1e-5, 1e3, True, name),  # srmax
                (0.25, 1e-5, 1.0, False, name),  # lp
                (100.0, 1e-5, 1e4, True, name),  # ks
                (2.0, 1e-5, 20.0, True, name),  # gamma
                (1.0, 0.25, 2.0, True, name),  # kv
            ],
            columns=["initial", "pmin", "pmax", "vary", "name"],
            index=[
                name + "_srmax",
                name + "_lp",
                name + "_ks",
                name + "_gamma",
                name + "_kv",
            ],
        )
        if self.interception:
            parameters.loc[name + "_simax"] = (2.0, 0.0, 10.0, False, name)
        if self.gw_uptake:
            parameters.loc[name + "_gf"] = (1.0, 0.0, 1.0, True, name)
        if self.snow:
            parameters.loc[name + "_tt"] = (0.0, -10.0, 10.0, False, name)
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
        **kwargs,
    ) -> ArrayLike | tuple[ArrayLike, ...]:
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
                error.real.round(2),
                p.real.astype(float).round(2),
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
            # Strip imaginary part when not doing complex-step Jacobian
            if not np.iscomplexobj(p):
                data = tuple(arr.real for arr in data)
            return data
        else:
            result = -r
            # Strip imaginary part when not doing complex-step Jacobian
            return result if np.iscomplexobj(p) else result.real

    @staticmethod
    @njit
    def get_root_zone_balance(
        pe: ArrayLike,
        ep: ArrayLike,
        srmax: complex | float = 250.0,
        lp: complex | float = 0.25,
        ks: complex | float = 100.0,
        gamma: complex | float = 4.0,
        dt: float = 1.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Compute the water balance of the root zone reservoir.

        Parameters
        ----------
        pe: array_like
            Effective precipitation flux in mm/d.
        ep: array_like
            Potential evaporation flux in mm/d.
        srmax: complex or float, optional
            Maximum storage capacity of the root zone.
        lp: complex or float, optional
            Parameter determining when actual evaporation equals potential.
        ks: complex or float, optional
            Saturated hydraulic conductivity in mm/d.
        gamma: complex or float, optional
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
        sr = zeros(n + 1, dtype=np.complex128)  # Root Zone Storage State
        sr[0] = 0.5 * srmax  # Set the initial system state to half-full
        ea = zeros(n, dtype=np.complex128)  # Actual evaporation Flux
        r = zeros(n, dtype=np.complex128)  # Recharge Flux
        q = zeros(n, dtype=np.complex128)  # Surface runoff Flux
        lp = lp * srmax  # Do this here outside the for-loop for efficiency

        for t in range(n):
            # Make sure the solution is larger than 0.0 and smaller than sr
            if sr[t].real > srmax.real:
                q[t] = sr[t] - srmax  # Surface runoff
                sr[t] = srmax
            elif sr[t].real < 0.0:
                sr[t] = 0.0

            # Calculate evaporation from the root zone reservoir
            if (sr[t] / lp).real < 1.0:
                ea[t] = ep[t] * sr[t] / lp
            else:
                ea[t] = ep[t]

            # Calculate the recharge flux
            # Use .real for comparison to support complex-step differentiation
            recharge = ks * (sr[t] / srmax) ** gamma
            r[t] = recharge if recharge.real < sr[t].real else sr[t]
            # Update storage in the root zone
            sr[t + 1] = sr[t] + dt * (pe[t] - r[t] - ea[t])

        return sr[:-1], -r, -ea, -q, pe

    @staticmethod
    @njit
    def get_interception_balance(
        pr: ArrayLike, ep: ArrayLike, simax: np.complex128 = 2.0, dt: float = 1.0
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        r"""Compute the water balance of the interception reservoir.

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
        si = zeros(n + 1, dtype=np.complex128)  # Interception Storage State
        pe = zeros(n, dtype=np.complex128)  # Effective precipitation Flux
        ei = zeros(n, dtype=np.complex128)  # Interception evaporation Flux

        for t in range(n):
            # Interception bucket
            # Use .real for comparisons to support complex-step differentiation
            if ep[t].real < si[t].real:
                ei[t] = ep[t]
            else:
                ei[t] = si[t]
            si[t + 1] = si[t] + dt * (pr[t] - ei[t])
            diff = si[t + 1] - simax
            if diff.real > 0.0:
                pe[t] = diff
            else:
                pe[t] = 0.0
            si[t + 1] = si[t + 1] - pe[t]

        pi = pr - pe  # Compute intercepted precipitation

        return si[:-1], -ei, pi

    @staticmethod
    @njit
    def get_snow_balance(
        prec: ArrayLike,
        temp: ArrayLike,
        tt: complex | float = 0.0,
        k: complex | float = 2.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        r"""Compute the water balance of the snow reservoir.

        Parameters
        ----------
        prec: array_like
            NumPy Array with precipitation in mm/day.
        temp: array_like
            NumPy Array with the mean daily temperature in degree Celsius.
        tt: complex or float, optional
            Temperature threshold for snowfall in degree Celsius.
        k: complex or  float, optional
            Degree-day factor in mm/d/°C.

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
        ss = zeros(n + 1, dtype=np.complex128)  # Snow Storage
        m = zeros(n, dtype=np.complex128)  # Potential Snow melt
        ps = where(temp <= tt.real, prec, 0.0)  # Snowfall

        # Snow bucket
        for t in range(n):
            if temp[t] > tt.real:
                m[t] = k * (temp[t] - tt)
                smoothing_factor = 1.0 - exp(-(ss[t] / 1.5))
                melt = m[t] * smoothing_factor
                # Use .real for comparison to support complex-step differentiation
                if melt.real < ss[t].real:
                    m[t] = melt
                else:
                    m[t] = ss[t]
            ss[t + 1] = ss[t] + ps[t] - m[t]

        return ss[:-1], ps, -m

    def get_water_balance(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        temp: ArrayLike,
        p: ArrayLike,
        dt: float = 1.0,
        **kwargs,
    ) -> DataFrame:
        """Get the water balance for the FlexModel recharge model."""
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
        """Check the water balance of the snow reservoir."""
        ss, ps, m = self.get_snow_balance(prec, temp)
        error = ss[0] - ss[-1] + (ps + m).sum()
        return error

    def check_interception_balance(
        self, prec: ArrayLike, evap: ArrayLike, **kwargs
    ) -> float:
        """Check the water balance of the interception reservoir."""
        si, ei, pi = self.get_interception_balance(prec, evap)
        error = si[0] - si[-1] + (pi + ei).sum()
        return error

    def check_root_zone_balance(
        self, prec: ArrayLike, evap: ArrayLike, **kwargs
    ) -> float:
        """Check the water balance of the root zone reservoir."""
        sr, r, ea, q, pe = self.get_root_zone_balance(prec, evap)
        error = sr[0] - sr[-1] + (r + ea + q + pe).sum()
        return error

    def to_dict(self) -> dict[str, Any]:
        """Export the recharge model object to a dictionary."""
        data = super().to_dict() | {
            "interception": self.interception,
            "snow": self.snow,
            "gw_uptake": self.gw_uptake,
        }
        return data


class Berendrecht(RechargeBase):
    r"""Nonlinear recharge to the groundwater.

    Calculated according to :cite:t:`berendrecht_non-linear_2006`.

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

    def __init__(self) -> None:
        super().__init__()

    @property
    def nparam(self) -> int:
        """Number of parameters of the Berendrecht recharge model."""
        return 7

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters and bounds for the Berendrecht recharge model."""
        parameters = DataFrame(
            [
                (0.9, 0.7, 1.3, False, name),  # fi
                (1.0, 0.7, 1.3, False, name),  # fc
                (0.25, 1e-5, 1.0, False, name),  # sr
                (250.0, 20.0, 1e3, True, name),  # de
                (2.0, -4.0, 50.0, True, name),  # l
                (0.5, 1e-5, 0.5, False, name),  # m
                (100.0, 1.0, 1e4, True, name),  # ks
            ],
            columns=["initial", "pmin", "pmax", "vary", "name"],
            index=[
                name + "_fi",
                name + "_fc",
                name + "_sr",
                name + "_de",
                name + "_l",
                name + "_m",
                name + "_ks",
            ],
        )
        return parameters

    def simulate(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        p: ArrayLike,
        dt: ArrayLike = 1.0,
        return_full: bool = False,
        **kwargs,
    ) -> ArrayLike | tuple[ArrayLike, ...]:
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
            # Strip imaginary part when not doing complex-step Jacobian
            if not np.iscomplexobj(p):
                return r.real, s.real, ea.real, pe.real
            return r, s, ea, pe
        else:
            result = nan_to_num(r)
            return result if np.iscomplexobj(p) else result.real

    @staticmethod
    @njit
    def get_recharge(
        prec: ArrayLike,
        evap: ArrayLike,
        fi: complex | float = 1.0,
        fc: complex | float = 1.0,
        sr: complex | float = 0.5,
        de: complex | float = 250.0,
        l: complex | float = -2.0,  # noqa: E741
        m: complex | float = 0.5,
        ks: complex | float = 50.0,
        dt: float = 1.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Calculate recharge flux sped up with numba."""
        n = prec.size
        # Create an empty arrays to store the fluxes and states
        pe = fi * prec  # Effective precipitation flux
        ep = fc * evap  # Potential evaporation flux
        s = zeros(n, dtype=complex128)  # Root zone storage state
        s[0] = 0.5  # Set the initial system state
        r = zeros(n, dtype=complex128)  # Recharge flux
        ea = zeros(n, dtype=complex128)  # Actual evaporation flux

        for t in range(n - 1):
            # Make sure the reservoir is not too full or empty.
            if s[t].real < 0.05:
                s[t] = 0.05 * exp(20.0 * s[t] - 1.0)
            elif s[t].real > 0.95:
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
        """Get the water balance for the Berendrecht recharge model."""
        r, s, ea, pe = self.simulate(prec, evap, p=p, dt=dt, return_full=True, **kwargs)
        s = s * p[3]  # Because S is computed dimensionless in this model
        data = DataFrame(data=vstack((s, pe, ea, r)).T, columns=["S", "Pe", "Ea", "R"])
        return data


class Peterson(RechargeBase):
    r"""Nonlinear recharge to the groundwater.

    Calculated based on :cite:t:`peterson_nonlinear_2014`.

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

    def __init__(self) -> None:
        super().__init__()

    @property
    def nparam(self) -> int:
        """Number of parameters of the Peterson recharge model."""
        return 5

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters and bounds for the Peterson recharge model."""
        parameters = DataFrame(
            [
                (1.5, 0.5, 3.0, True, name),  # scap
                (1.0, 0.0, 1.5, True, name),  # alpha
                (1.0, 0.0, 3.0, True, name),  # ksat
                (0.5, 0.0, 1.5, True, name),  # beta
                (1.0, 0.0, 2.0, True, name),  # gamma
            ],
            columns=["initial", "pmin", "pmax", "vary", "name"],
            index=[
                name + "_scap",
                name + "_alpha",
                name + "_ksat",
                name + "_beta",
                name + "_gamma",
            ],
        )
        return parameters

    def simulate(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        p: ArrayLike,
        dt: float = 1.0,
        return_full: bool = False,
        **kwargs,
    ) -> ArrayLike | tuple[ArrayLike, ...]:
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
            # Strip imaginary part when not doing complex-step Jacobian
            if not np.iscomplexobj(p):
                return r.real, s.real, ea.real, pe.real
            return r, s, ea, pe
        else:
            result = nan_to_num(r)
            return result if np.iscomplexobj(p) else result.real

    @staticmethod
    @njit
    def get_recharge(
        prec: ArrayLike,
        evap: ArrayLike,
        scap: complex | float = 1.0,
        alpha: complex | float = 1.0,
        ksat: complex | float = 1.0,
        beta: complex | float = 0.5,
        gamma: complex | float = 1.0,
        dt: float = 1.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Calculate recharge flux sped up with numba."""
        n = len(prec)
        # Create an empty arrays to store the fluxes and states
        pe = zeros(n, dtype=complex128)  # Effective precipitation flux
        sm = zeros(n + 1, dtype=complex128)  # Root zone storage state
        r = zeros(n, dtype=complex128)  # Recharge flux
        ea = zeros(n, dtype=complex128)  # Actual evaporation flux
        # Update params
        smsc = power(10, scap)
        ksat = power(10, ksat)
        beta = power(10, beta)
        # Set the initial system state
        sm[0] = smsc / 2

        for t in range(n):
            sm_frac = sm[t] / smsc
            pe[t] = prec[t] * power(1 - sm_frac, alpha)
            # Use .real for comparisons to support complex-step differentiation
            ea_val = evap[t] * power(sm_frac, gamma)
            ea[t] = ea_val if ea_val.real > sm[t + 1].real else sm[t + 1]
            r_val = ksat * power(sm_frac, beta)
            r[t] = r_val if r_val.real > sm[t + 1].real else sm[t + 1]
            sm_new = sm[t] + (pe[t] - ea[t] - r[t]) * dt
            if sm_new.real < 0.0:
                sm[t + 1] = complex(0.0)
            elif sm_new.real > smsc.real:
                sm[t + 1] = smsc
            else:
                sm[t + 1] = sm_new
        return r, sm[1:], ea, pe

    def get_water_balance(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, dt: float = 1.0, **kwargs
    ) -> DataFrame:
        """Get the water balance for the Peterson recharge model."""
        r, s, ea, pe = self.simulate(prec, evap, p=p, dt=dt, return_full=True, **kwargs)
        data = DataFrame(data=vstack((s, pe, ea, r)).T, columns=["S", "Pe", "Ea", "R"])
        return data


class Ireson(RechargeBase):
    """Nonlinear recharge model with soil moisture deficit approach.

    Calculated according to the simple recharge model evaluated in
    :cite:t:`ireson_nonlinear_2013`. This approach is similar to
    conventional approaches for modelling recharge to Chalk aquifers.


    Notes
    -----
    The water balance tracks the actual soil moisture deficit (SMDa).
    Bypass flow (B) occurs when precipitation (P) exceeds a threshold (TH):
    B = BF * (P - TH)

    A potential soil moisture deficit (SMDp) is calculated using potential
    evaporation (Ep): SMDp = SMDa(t-1) - (P - B) + Ep

    Drainage (D) occurs when SMDp is negative. Recharge (R) is the sum
    of drainage and bypass flow: R = D + B

    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def nparam(self) -> int:
        """Number of parameters of the Ireson recharge model."""
        return 4

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters and bounds for the Ireson recharge model."""
        parameters = DataFrame(
            [
                (500.0, 10.0, 2000.0, True, name),  # rc: root constant
                (1000.0, 20.0, 4000.0, True, name),  # pwp: permanent wilting point
                (0.0, 0.0, 0.3, False, name),  # bf: bypass fraction
                (0.0, 0.0, 30.0, False, name),  # th: bypass threshold
            ],
            columns=["initial", "pmin", "pmax", "vary", "name"],
            index=[
                name + "_rc",
                name + "_pwp",
                name + "_bf",
                name + "_th",
            ],
        )
        return parameters

    def simulate(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        p: ArrayLike,
        dt: float = 1.0,
        return_full: bool = False,
        **kwargs,
    ) -> ArrayLike | tuple[ArrayLike, ...]:
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
            time step for the calculation of the recharge. Only dt=1 is possible now.
        return_full: bool
            return all fluxes and states as NumPy arrays.

        Returns
        -------
        r: array_like or tuple of array_like
            Recharge flux calculated by the model if the argument full_output is
            False, otherwise a tuple with all fluxes and states.
        """
        r, smda, smdp, ea, b, d = self.get_recharge(
            prec, evap, rc=p[0], pwp=p[1], bf=p[2], th=p[3], dt=dt
        )
        if return_full:
            # Strip imaginary part when not doing complex-step Jacobian
            if not np.iscomplexobj(p):
                return r.real, smda.real, smdp.real, ea.real, b.real, d.real
            return r, smda, smdp, ea, b, d
        else:
            result = nan_to_num(r)
            return result if np.iscomplexobj(p) else result.real

    @staticmethod
    @njit
    def get_recharge(
        prec: ArrayLike,
        evap: ArrayLike,
        rc: complex | float,
        pwp: complex | float,
        bf: complex | float,
        th: complex | float,
        dt: float = 1.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Calculate recharge flux sped up with numba."""
        n = prec.size
        # Create empty arrays to store the fluxes and states
        r = zeros(n, dtype=complex128)  # Recharge flux
        ea = zeros(n, dtype=complex128)  # Actual evaporation
        b = zeros(n, dtype=complex128)  # Bypass flow
        d = zeros(n, dtype=complex128)  # Soil drainage
        smda = zeros(n + 1, dtype=complex128)  # Actual soil moisture deficit
        smdp = zeros(n, dtype=complex128)  # Potential soil moisture deficit

        # Initial condition
        smda[0] = 0.0

        for t in range(n):
            # Calculate bypass flow (B)
            if (prec[t] - th).real > 0.0:
                b[t] = bf * (prec[t] - th)
            else:
                b[t] = 0.0

            # Calculate potential soil moisture deficit (SMDp)
            smdp[t] = smda[t] - (prec[t] - b[t]) + evap[t]

            # Calculate actual evaporation (Ea)
            if smdp[t].real <= rc.real:
                ea[t] = evap[t]
            elif smdp[t].real < pwp.real:
                ea[t] = evap[t] * ((pwp - smdp[t]) / (pwp - rc))
            else:
                ea[t] = 0.0

            # Calculate soil drainage (D)
            if smdp[t].real < 0.0:
                d[t] = -smdp[t]
            else:
                d[t] = 0.0

            # Update actual soil moisture deficit (SMDa) for next step
            if smdp[t].real < 0.0:
                smda[t + 1] = 0.0
            else:
                smda[t + 1] = smda[t] - (prec[t] - b[t]) + ea[t]

            # Calculate total recharge (R)
            r[t] = d[t] + b[t]

        return r, smda[:-1], smdp, ea, b, d

    def get_water_balance(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, dt: float = 1.0, **kwargs
    ) -> DataFrame:
        """Get the water balance for the Ireson recharge model."""
        r, smda, smdp, ea, b, d = self.simulate(
            prec, evap, p=p, dt=dt, return_full=True, **kwargs
        )
        data = DataFrame(
            data=vstack((smda, smdp, ea, b, d, r)).T,
            columns=["SMDa", "SMDp", "Ea", "B", "D", "R"],
        )
        return data
