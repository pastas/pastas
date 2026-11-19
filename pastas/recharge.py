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
    sm = ps.RechargeModel(ml, prec, evap, recharge=rch, rfunc=ps.Gamma(), name="rch")

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

from .decorators import _real, njit

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
    def simulate(self, *args, **kwargs) -> ArrayLike:
        """Simulate recharge from precipitation and evaporation inputs.

        A 2-D `p` of shape (n_real, n_param) is supported by all models
        and returns an array of shape (n_real, n_points). With
        `return_full=True`, all fluxes and states are stacked along the
        first axis of a single array: (n_fluxes, n_points) for a 1-D `p`
        and (n_fluxes, n_real, n_points) for a 2-D `p`.
        """

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
        _n_param = 5
        if self.interception:
            _n_param += 1
        if self.gw_uptake:
            _n_param += 1
        if self.snow:
            _n_param += 2
        return _n_param

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters and bounds for the FlexModel recharge model."""
        parameters = DataFrame(
            [
                (250.0, 10.0, 1e3, True, name),  # srmax
                (0.25, 1e-5, 1.0, False, name),  # lp
                (100.0, 1.0, 1e3, True, name),  # ks
                (2.0, 1e-5, 5.0, True, name),  # gamma
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
        return_full: bool, optional
            Return all fluxes and states stacked along the first axis of a
            single array, instead of only the recharge flux.

        Returns
        -------
        array_like
            Recharge flux calculated by the model: shape (n_points,) for a 1-D
            `p`, (n_real, n_points) for a 2-D `p` of shape (n_real, n_param).
            With `return_full=True`, all fluxes and states are stacked along
            the first axis: (n_fluxes, n_points) for a 1-D `p` and
            (n_fluxes, n_real, n_points) for a 2-D `p`.
        """
        if np.ndim(p) > 1:
            return self._simulate_ensemble(
                prec, evap, temp, np.asarray(p), dt, return_full=return_full
            )
        is_complex = np.iscomplexobj(p)
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
        error_value = error.real if is_complex else error
        if abs(error) > 0.1:
            p_log = p.real.astype(float) if is_complex else p.astype(float)
            logger.info(
                "Water balance error: %s %% of the total pe flux. Parameters: %s",
                error_value.round(2),
                p_log.round(2),
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
            data = (sr, -r, ea, q, pe)
            if self.interception:
                data += (si, ei, pi)
            if self.snow:
                data += (ss, ps, m)
            return np.stack(data)
        return -r

    def _simulate_ensemble(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        temp: ArrayLike,
        P: ArrayLike,
        dt: float = 1.0,
        return_full: bool = False,
    ) -> ArrayLike:
        """Simulate the soil water balance for a 2-D parameter array (no loop).

        Parameters
        ----------
        prec, evap, temp: array_like
            Input fluxes, shape (n_points,), shared over all realizations.
        P: array_like, shape (n_real, n_param)
            Parameter sets, one per realization.
        dt: float, optional
            Time step for the calculation of the recharge.
        return_full: bool, optional
            Return all fluxes and states stacked along the first axis.

        Returns
        -------
        array_like
            The recharge flux for each realization, shape (n_real, n_points).
            With `return_full=True`, all fluxes and states stacked along the
            first axis: shape (n_fluxes, n_real, n_points).
        """
        n_real = P.shape[0]
        prec = np.asarray(prec)
        evap = np.asarray(evap)
        prec2 = np.repeat(prec[None, :], n_real, axis=0)
        evap2 = np.repeat(evap[None, :], n_real, axis=0)
        ep = evap2 * P[:, 4][:, None]

        if self.snow:
            temp2 = np.repeat(np.asarray(temp)[None, :], n_real, axis=0)
            ss, ps, m = self._snow_balance_ensemble(prec2, temp2, P[:, -2], P[:, -1])
            pr = prec2 - ps  # Remove snowfall from precipitation
        else:
            pr = prec2  # All precipitation is rainfall and melt is zero
            m = 0.0

        if self.interception:
            si, ei, pi = self._interception_balance_ensemble(pr, ep, P[:, 5])
            ep = ep + ei  # Update potential evaporation after interception
            pe = pr - pi  # Update rainfall after interception
        else:
            pe = pr

        sr, r, ea, q, _ = self._root_zone_balance_ensemble(
            pe - m, ep, P[:, 0], P[:, 1], P[:, 2], P[:, 3], dt
        )

        # report big water balance errors (error > 0.1%)
        error = (sr[:, 0] - sr[:, -1] + (pe - m + r + ea + q).sum(axis=1)) / (
            pe.sum(axis=1) + 1e-10
        )  # avoid division by zero
        if np.any(np.abs(error) > 0.1):
            logger.info(
                "Water balance error > 0.1%% of the total pe flux for %s "
                "realization(s). Parameters: %s",
                int(np.sum(np.abs(error) > 0.1)),
                P[np.abs(error) > 0.1].real.astype(float).round(2),
            )

        if self.gw_uptake:
            # Compute leftover potential evaporation
            gf = P[:, 6][:, None] if self.interception else P[:, 5][:, None]
            eg = ep + ea  # positive flux
            r = r + gf * eg

        if return_full:
            data = (sr, -r, ea, q, pe)
            if self.interception:
                data += (si, ei, pi)
            if self.snow:
                data += (ss, ps, m)
            return np.stack(data)
        return -r

    @staticmethod
    @njit
    def get_root_zone_balance(
        pe: ArrayLike,
        ep: ArrayLike,
        srmax: float | complex128 = 250.0,
        lp: float | complex128 = 0.25,
        ks: float | complex128 = 100.0,
        gamma: float | complex128 = 4.0,
        dt: float = 1.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Compute the water balance of the root zone reservoir.

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

        Notes
        -----
        Thin 1-D adapter over the `_root_zone_balance_ensemble` static method: the inputs are
        reshaped to (1, n_points) and the parameters to (1,) arrays, so a single
        2-D-native kernel implementation serves both the 1-D and the ensemble
        path.
        """
        sr, r, ea, q, pe_out = _root_zone_balance_ensemble(
            pe.reshape(1, -1),
            ep.reshape(1, -1),
            np.full(1, srmax),
            np.full(1, lp),
            np.full(1, ks),
            np.full(1, gamma),
            dt,
        )
        return sr[0], r[0], ea[0], q[0], pe_out[0]

    @staticmethod
    @njit
    def _root_zone_balance_ensemble(pe, ep, srmax, lp, ks, gamma, dt=1.0):
        """Root zone water balance for an ensemble of realizations (2-D fluxes).

        Fluxes are (n_real, n_points) arrays and the scalar parameters are
        (n_real,) arrays, so all realizations are computed in a single pass
        over time.
        """
        n_real, n = pe.shape
        # Infer target dtype purely from the scalar parameters
        dtype = np.asarray(srmax[0] + lp[0] + ks[0] + gamma[0]).dtype
        sr = zeros((n_real, n + 1), dtype=dtype)
        sr[:, 0] = 0.5 * srmax
        ea = zeros((n_real, n), dtype=dtype)
        r = zeros((n_real, n), dtype=dtype)
        q = zeros((n_real, n), dtype=dtype)
        lp_scaled = lp * srmax
        for t in range(n):
            sr_t = sr[:, t].copy()
            over = _real(sr_t) > _real(srmax)
            under = (~over) & (_real(sr_t) < 0.0)
            q[:, t] = where(over, sr_t - srmax, 0.0)
            sr[:, t] = where(over, srmax, where(under, 0.0, sr_t))
            ea[:, t] = where(
                _real(sr[:, t] / lp_scaled) < 1.0,
                ep[:, t] * sr[:, t] / lp_scaled,
                ep[:, t],
            )
            recharge = ks * (sr[:, t] / srmax) ** gamma
            r[:, t] = where(_real(recharge) < _real(sr[:, t]), recharge, sr[:, t])
            sr[:, t + 1] = sr[:, t] + dt * (pe[:, t] - r[:, t] - ea[:, t])
        return sr[:, :-1], -r, -ea, -q, pe

    @staticmethod
    @njit
    def get_interception_balance(
        pr: ArrayLike,
        ep: ArrayLike,
        simax: float | complex128 = 2.0,
        dt: float = 1.0,
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

        Notes
        -----
        Thin 1-D adapter over the `_interception_balance_ensemble` static method.
        """
        si, ei, pi = _interception_balance_ensemble(
            pr.reshape(1, -1), ep.reshape(1, -1), np.full(1, simax), dt
        )
        return si[0], ei[0], pi[0]

    @staticmethod
    @njit
    def _interception_balance_ensemble(pr, ep, simax, dt=1.0):
        """Compute the interception reservoir water balance for an ensemble of realizations."""
        n_real, n = pr.shape
        dtype = np.asarray(simax[0]).dtype
        si = zeros((n_real, n + 1), dtype=dtype)
        pe = zeros((n_real, n), dtype=dtype)
        ei = zeros((n_real, n), dtype=dtype)
        for t in range(n):
            ei[:, t] = where(_real(ep[:, t]) < _real(si[:, t]), ep[:, t], si[:, t])
            si[:, t + 1] = si[:, t] + dt * (pr[:, t] - ei[:, t])
            diff = si[:, t + 1] - simax
            pe[:, t] = where(_real(diff) > 0.0, diff, 0.0)
            si[:, t + 1] = si[:, t + 1] - pe[:, t]
        pi = pr - pe  # Compute intercepted precipitation
        return si[:, :-1], -ei, pi

    @staticmethod
    @njit
    def get_snow_balance(
        prec: ArrayLike,
        temp: ArrayLike,
        tt: float | complex128 = 0.0,
        k: float | complex128 = 2.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        r"""Compute the water balance of the snow reservoir.

        Parameters
        ----------
        prec: array_like
            NumPy Array with precipitation in mm/day.
        temp: array_like
            NumPy Array with the mean daily temperature in degree Celsius.
        tt: float, optional
            Temperature threshold for snowfall in degree Celsius.
        k: float, optional
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

        Notes
        -----
        Thin 1-D adapter over the `_snow_balance_ensemble` static method.
        """
        ss, ps, m = _snow_balance_ensemble(
            prec.reshape(1, -1), temp.reshape(1, -1), np.full(1, tt), np.full(1, k)
        )
        return ss[0], ps[0], m[0]

    @staticmethod
    @njit
    def _snow_balance_ensemble(prec, temp, tt, k):
        """Snow reservoir water balance for an ensemble of realizations.

        prec, temp: (n_real, n_points); tt, k: (n_real,).
        """
        n_real, n = prec.shape
        # promote types via scalar arithmetic (Numba compiles this at compile-time)
        dtype = np.asarray(tt[0] + k[0]).dtype
        ss = zeros((n_real, n + 1), dtype=dtype)  # Snow Storage
        m = zeros((n_real, n), dtype=dtype)  # Potential Snow melt
        ps = where(temp <= _real(tt)[:, None], prec, 0.0)  # Snowfall
        for t in range(n):
            warm = temp[:, t] > _real(tt)
            melt = k * (temp[:, t] - tt) * (1.0 - exp(-(ss[:, t] / 1.5)))
            m[:, t] = where(
                warm, where(_real(melt) < _real(ss[:, t]), melt, ss[:, t]), 0.0
            )
            ss[:, t + 1] = ss[:, t] + ps[:, t] - m[:, t]
        return ss[:, :-1], ps, -m

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
        if np.ndim(p) > 1:
            raise NotImplementedError("get_water_balance is only supported for 1-D p.")
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

        return DataFrame(data=np.asarray(data).T, columns=columns)

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
    ) -> ArrayLike:
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
        return_full: bool, optional
            Return all fluxes and states stacked along the first axis of a
            single array, instead of only the recharge flux.

        Returns
        -------
        array_like
            Recharge flux calculated by the model: shape (n_points,) for a 1-D
            `p`, (n_real, n_points) for a 2-D `p` of shape (n_real, n_param).
            With `return_full=True`, all fluxes and states are stacked along
            the first axis: (n_fluxes, n_points) for a 1-D `p` and
            (n_fluxes, n_real, n_points) for a 2-D `p`.
        """
        if np.ndim(p) > 1:
            P = np.asarray(p)
            n_real = P.shape[0]
            prec2 = np.repeat(np.asarray(prec)[None, :], n_real, axis=0)
            evap2 = np.repeat(np.asarray(evap)[None, :], n_real, axis=0)
            r, s, ea, pe = self._berendrecht_recharge_ensemble(
                prec2,
                evap2,
                P[:, 0],
                P[:, 1],
                P[:, 2],
                P[:, 3],
                P[:, 4],
                P[:, 5],
                P[:, 6],
                dt,
            )
            if return_full:
                return np.stack((r, s, ea, pe))
            return nan_to_num(r)
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
            return np.stack((r, s, ea, pe))
        return nan_to_num(r)

    @staticmethod
    @njit
    def get_recharge(
        prec: ArrayLike,
        evap: ArrayLike,
        fi: float | complex128 = 1.0,
        fc: float | complex128 = 1.0,
        sr: float | complex128 = 0.5,
        de: float | complex128 = 250.0,
        l: float | complex128 = -2.0,
        m: float | complex128 = 0.5,
        ks: float | complex128 = 50.0,
        dt: float = 1.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Calculate recharge flux sped up with numba.

        Notes
        -----
        Thin 1-D adapter over the `_berendrecht_recharge_ensemble` static method.
        """
        r, s, ea, pe = _berendrecht_recharge_ensemble(
            prec.reshape(1, -1),
            evap.reshape(1, -1),
            np.full(1, fi),
            np.full(1, fc),
            np.full(1, sr),
            np.full(1, de),
            np.full(1, l),
            np.full(1, m),
            np.full(1, ks),
            dt,
        )
        return r[0], s[0], ea[0], pe[0]

    @staticmethod
    @njit
    def _berendrecht_recharge_ensemble(prec, evap, fi, fc, sr, de, l, m, ks, dt=1.0):
        """Berendrecht recharge for an ensemble of realizations (2-D fluxes)."""
        n_real, n = prec.shape
        pe = fi[:, None] * prec  # Effective precipitation flux
        ep = fc[:, None] * evap  # Potential evaporation flux
        dtype = np.asarray(fi[0] + fc[0] + sr[0] + de[0] + l[0] + m[0] + ks[0]).dtype
        s = zeros((n_real, n), dtype=dtype)  # Root zone storage state
        s[:, 0] = 0.5  # Set the initial system state
        r = zeros((n_real, n), dtype=dtype)  # Recharge flux
        ea = zeros((n_real, n), dtype=dtype)  # Actual evaporation flux
        for t in range(n - 1):
            s_t = s[:, t]
            low = _real(s_t) < 0.05
            high = (~low) & (_real(s_t) > 0.95)
            s[:, t] = where(
                low,
                0.05 * exp(20.0 * s_t - 1.0),
                where(high, 1.0 - (0.05 * exp(19.0 - 20.0 * s_t)), s_t),
            )
            ea[:, t] = (1.0 - exp(-3.0 * s[:, t] / sr)) * ep[:, t]
            r[:, t] = ks * s[:, t] ** l * (1.0 - (1.0 - s[:, t] ** (1.0 / m)) ** m) ** 2
            s[:, t + 1] = s[:, t] + dt / de * (pe[:, t] - ea[:, t] - r[:, t])
        return r, s, ea, pe

    def get_water_balance(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, dt: float = 1.0, **kwargs
    ) -> DataFrame:
        """Get the water balance for the Berendrecht recharge model."""
        if np.ndim(p) > 1:
            raise NotImplementedError("get_water_balance is only supported for 1-D p.")
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
    ) -> ArrayLike:
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
        return_full: bool, optional
            Return all fluxes and states stacked along the first axis of a
            single array, instead of only the recharge flux.

        Returns
        -------
        r: array_like or list of array_like
            Recharge flux calculated by the model is the argument full_output is
            False, otherwise a list with all fluxes and states. For a 2-D `p` of
            shape (n_real, n_param), an array of shape
            (n_real, n_points) is returned.

        """
        if np.ndim(p) > 1:
            P = np.asarray(p)
            n_real = P.shape[0]
            prec2 = np.repeat(np.asarray(prec)[None, :], n_real, axis=0)
            evap2 = np.repeat(np.asarray(evap)[None, :], n_real, axis=0)
            r, s, ea, pe = self._peterson_recharge_ensemble(
                prec2, evap2, P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4], dt
            )
            if return_full:
                return np.stack((r, s, ea, pe))
            return nan_to_num(r)
        r, s, ea, pe = self.get_recharge(
            prec,
            evap,
            scap=p[0],
            alpha=p[1],
            ksat=p[2],
            beta=p[3],
            gamma=p[4],
            dt=dt,
        )
        if return_full:
            return np.stack((r, s, ea, pe))
        return nan_to_num(r)

    @staticmethod
    @njit
    def get_recharge(
        prec: ArrayLike,
        evap: ArrayLike,
        scap: float | complex128 = 1.0,
        alpha: float | complex128 = 1.0,
        ksat: float | complex128 = 1.0,
        beta: float | complex128 = 0.5,
        gamma: float | complex128 = 1.0,
        dt: float = 1.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Calculate recharge flux sped up with numba.

        Notes
        -----
        Thin 1-D adapter over the `_peterson_recharge_ensemble` static method.
        """
        r, sm, ea, pe = _peterson_recharge_ensemble(
            prec.reshape(1, -1),
            evap.reshape(1, -1),
            np.full(1, scap),
            np.full(1, alpha),
            np.full(1, ksat),
            np.full(1, beta),
            np.full(1, gamma),
            dt,
        )
        return r[0], sm[0], ea[0], pe[0]

    @staticmethod
    @njit
    def _peterson_recharge_ensemble(prec, evap, scap, alpha, ksat, beta, gamma, dt=1.0):
        """Peterson recharge for an ensemble of realizations (2-D fluxes)."""
        n_real, n = prec.shape
        dtype = np.asarray(scap[0] + alpha[0] + ksat[0] + beta[0] + gamma[0]).dtype
        pe = zeros((n_real, n), dtype=dtype)  # Effective precipitation flux
        sm = zeros((n_real, n + 1), dtype=dtype)  # Root zone storage state
        r = zeros((n_real, n), dtype=dtype)  # Recharge flux
        ea = zeros((n_real, n), dtype=dtype)  # Actual evaporation flux
        # Update params
        smsc = power(10, scap)
        ksat_ = power(10, ksat)
        beta_ = power(10, beta)
        sm[:, 0] = smsc / 2  # Set the initial system state
        for t in range(n):
            sm_frac = sm[:, t] / smsc
            pe[:, t] = prec[:, t] * power(1.0 - sm_frac, alpha)
            ea_val = evap[:, t] * power(sm_frac, gamma)
            ea[:, t] = where(_real(ea_val) > _real(sm[:, t + 1]), ea_val, sm[:, t + 1])
            r_val = ksat_ * power(sm_frac, beta_)
            r[:, t] = where(_real(r_val) > _real(sm[:, t + 1]), r_val, sm[:, t + 1])
            sm_new = sm[:, t] + (pe[:, t] - ea[:, t] - r[:, t]) * dt
            sm[:, t + 1] = where(
                _real(sm_new) < 0.0,
                0.0,
                where(_real(sm_new) > _real(smsc), smsc, sm_new),
            )
        return r, sm[:, 1:], ea, pe

    def get_water_balance(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, dt: float = 1.0, **kwargs
    ) -> DataFrame:
        """Get the water balance for the Peterson recharge model."""
        if np.ndim(p) > 1:
            raise NotImplementedError("get_water_balance is only supported for 1-D p.")
        r, s, ea, pe = self.simulate(prec, evap, p=p, dt=dt, return_full=True, **kwargs)
        data = DataFrame(data=vstack((s, pe, ea, r)).T, columns=["S", "Pe", "Ea", "R"])
        return data


# Module-level aliases to the ensemble kernels, which are defined as static
# methods on the classes above. Numba resolves jitted callees through module
# globals, so the jitted 1-D adapters (e.g. FlexModel.get_root_zone_balance)
# call these aliases.
_root_zone_balance_ensemble = FlexModel._root_zone_balance_ensemble
_interception_balance_ensemble = FlexModel._interception_balance_ensemble
_snow_balance_ensemble = FlexModel._snow_balance_ensemble
_berendrecht_recharge_ensemble = Berendrecht._berendrecht_recharge_ensemble
_peterson_recharge_ensemble = Peterson._peterson_recharge_ensemble
