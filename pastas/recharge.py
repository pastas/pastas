"""Recharge models for Pastas."""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any

import numpy as np
from pandas import DataFrame

from pastas.typing import ArrayLike

from .decorators import _real, njit

logger = getLogger(__name__)


def _prepare_2d_flux(flux: ArrayLike, n_real: int) -> np.ndarray:
    """Return a 2-D (n_real, n_points) view without copying 1-D inputs."""
    arr = np.asarray(flux)
    if arr.ndim == 1:
        return np.broadcast_to(arr, (n_real, arr.shape[0]))
    return arr


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
        return DataFrame(
            [(-1.0, -2.0, 0.0, True, name)],
            columns=["initial", "pmin", "pmax", "vary", "name"],
            index=[name + "_f"],
        )

    def simulate(self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike) -> ArrayLike:
        """Simulate the precipitation excess flux.

        Parameters
        ----------
        prec : ArrayLike
            Precipitation flux values.
        evap : ArrayLike
            Potential evaporation flux values.
        p : ArrayLike
            Recharge model parameters: shape (1,) for a single realization or
            (n_real, 1) for an ensemble of realizations.

        Returns
        -------
        ArrayLike
            Recharge flux: shape (n_points,) for 1-D `p` or
            (n_real, n_points) for 2-D `p`.
        """
        return np.add(prec, np.multiply(evap, np.asarray(p)))

    def get_water_balance(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        p: ArrayLike,
    ) -> DataFrame:
        """Get the water balance for the linear recharge model."""
        ea = np.multiply(evap, p)
        r = np.add(prec, np.multiply(evap, p))
        return DataFrame(data=np.vstack((prec, ea, -r)).T, columns=["P", "Ea", "R"])

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

        \frac{dS}{dt} = P_e - E_a - R

    where the recharge is calculated as:

    .. math::

        R = K_s \left( \frac{S}{S_u}\right) ^\gamma

    If snow=True, a snow reservoir is added on top. For a detailed description of the
    degree-day snow model and parameters we refer to :cite:t:`kavetski_model_2007`.
    The water balance for the snow reservoir is written as:

    .. math::

        \frac{dSs}{dt} = Ps - M

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
            Model parameters: shape (n_param,) for a single realization or
            (n_real, n_param) for an ensemble of realizations.
        dt: float, optional
            Time step for the calculation of the recharge. Only dt=1 is possible now.
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
        p_arr = np.asarray(p)
        is_1d = p_arr.ndim == 1
        P = np.atleast_2d(p_arr)
        n_real = P.shape[0]

        prec_arr = np.asarray(prec)
        evap_arr = np.asarray(evap)
        ep = evap_arr[None, :] * P[:, 4:5]

        if self.snow:
            ss, ps, m = self._snow_kernel(
                _prepare_2d_flux(prec_arr, n_real),
                _prepare_2d_flux(temp, n_real),
                P[:, -2],
                P[:, -1],
            )
            pr = prec_arr[None, :] - ps
        else:
            pr = _prepare_2d_flux(prec_arr, n_real)
            m = 0.0

        if self.interception:
            si, ei, pi = self._interception_kernel(pr, ep, P[:, 5], dt)
            ep = ep + ei
            pe = pr - pi
        else:
            pe = pr

        pe_in = pe - m
        sr, r, ea, q = self._root_zone_kernel(
            pe_in, ep, P[:, 0], P[:, 1], P[:, 2], P[:, 3], dt
        )

        # Report big water balance errors (error > 0.1%)
        error = (sr[:, 0] - sr[:, -1] + (pe_in + r + ea + q).sum(axis=1)) / (
            pe_in.sum(axis=1) + 1e-10
        )
        bad = np.abs(error) > 0.1
        if np.any(bad):
            if is_1d:
                err_val = error[0].real if np.iscomplexobj(error) else error[0]
                logger.info(
                    "Water balance error: %s %% of the total pe flux. Parameters: %s",
                    err_val.round(2),
                    p_arr.real.astype(float).round(2),
                )
            else:
                logger.info(
                    "Water balance error > 0.1%% of the total pe flux for %s "
                    "realization(s). Parameters: %s",
                    int(np.sum(bad)),
                    P[bad].real.astype(float).round(2),
                )

        if self.gw_uptake:
            gf = P[:, 6:7] if self.interception else P[:, 5:6]
            r = r + gf * (ep + ea)

        if return_full:
            data = (sr, -r, ea, q, pe_in)
            if self.interception:
                data += (si, ei, pi)
            if self.snow:
                data += (ss, ps, m)
            out = np.stack(data)
            return out[:, 0, :] if is_1d else out

        return -r[0] if is_1d else -r

    @staticmethod
    @njit
    def get_root_zone_balance(
        pe: ArrayLike,
        ep: ArrayLike,
        srmax: float | np.complex128 = 250.0,
        lp: float | np.complex128 = 0.25,
        ks: float | np.complex128 = 100.0,
        gamma: float | np.complex128 = 4.0,
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
            Time step for the calculation of the recharge. Only dt=1 is possible now.

        Returns
        -------
        sr: array_like
            Storage in the root zone reservoir.
        r: array_like
            Recharge flux in mm/d.
        ea: array_like
            Evaporation flux in mm/d. Consists of transpiration and soil evaporation.
            Does not include interception evaporation.
        q: array_like
            Surface runoff flux in mm/d.
        pe: array_like
            Incoming infiltration flux in mm/d.
        """
        sr, r, ea, q = _flex_root_zone_kernel(
            pe.reshape(1, -1),
            ep.reshape(1, -1),
            np.full(1, srmax),
            np.full(1, lp),
            np.full(1, ks),
            np.full(1, gamma),
            dt,
        )
        return sr[0], r[0], ea[0], q[0], pe

    @staticmethod
    @njit
    def _root_zone_kernel(
        pe: np.ndarray,
        ep: np.ndarray,
        srmax: np.ndarray,
        lp: np.ndarray,
        ks: np.ndarray,
        gamma: np.ndarray,
        dt: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute root zone water balance across realizations (2-D arrays)."""
        n_real, n = pe.shape
        dtype = np.asarray(
            pe[0, 0] + ep[0, 0] + srmax[0] + lp[0] + ks[0] + gamma[0]
        ).dtype
        sr = np.zeros((n_real, n), dtype=dtype)
        ea = np.zeros((n_real, n), dtype=dtype)
        r = np.zeros((n_real, n), dtype=dtype)
        q = np.zeros((n_real, n), dtype=dtype)
        for i in range(n_real):
            srmax_i = srmax[i]
            lp_scaled_i = lp[i] * srmax_i
            ks_i = ks[i]
            gamma_i = gamma[i]
            sr_cur = 0.5 * srmax_i
            for t in range(n):
                if _real(sr_cur) > _real(srmax_i):
                    q_t = sr_cur - srmax_i
                    sr_cur = srmax_i
                elif _real(sr_cur) < 0.0:
                    q_t = 0.0
                    sr_cur = 0.0
                else:
                    q_t = 0.0

                if _real(sr_cur / lp_scaled_i) < 1.0:
                    ea_t = ep[i, t] * sr_cur / lp_scaled_i
                else:
                    ea_t = ep[i, t]

                recharge = ks_i * (sr_cur / srmax_i) ** gamma_i
                r_t = recharge if _real(recharge) < _real(sr_cur) else sr_cur

                sr[i, t] = sr_cur
                q[i, t] = -q_t
                ea[i, t] = -ea_t
                r[i, t] = -r_t
                sr_cur = sr_cur + dt * (pe[i, t] - r_t - ea_t)
        return sr, r, ea, q

    @staticmethod
    @njit
    def get_interception_balance(
        prec: ArrayLike,
        evap: ArrayLike,
        simax: float | np.complex128 = 2.0,
        dt: float = 1.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        r"""Compute the water balance of the interception reservoir.

        Parameters
        ----------
        prec: array_like
            NumPy Array with precipitation in mm/day.
        evap: array_like
            NumPy Array with potential evaporation in mm/day.
        simax: float, optional
            Storage capacity of the interception reservoir.
        dt: float, optional
            Time step used for computation. Only dt=1.0 is possible now.

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
        The water balance for the interception storage reservoir is defined as follows:

        .. math::

            \frac{dS_i}{dt} = P_r - E_i - P_e

        where $S_i$ [L] is the interception storage, $P_r$ [L/T] is the incoming
        rainfall, $E_i$ [L/T] the interception evaporation, and $P_e$ [L/T] the
        overflow from the interception reservoir.
        """
        si, ei, pi = _flex_interception_kernel(
            prec.reshape(1, -1), evap.reshape(1, -1), np.full(1, simax), dt
        )
        return si[0], ei[0], pi[0]

    @staticmethod
    @njit
    def _interception_kernel(
        prec: np.ndarray,
        evap: np.ndarray,
        simax: np.ndarray,
        dt: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute interception reservoir water balance across realizations (2-D arrays)."""
        n_real, n = prec.shape
        dtype = np.asarray(prec[0, 0] + evap[0, 0] + simax[0]).dtype
        si = np.zeros((n_real, n), dtype=dtype)
        ei = np.zeros((n_real, n), dtype=dtype)
        pi = np.zeros((n_real, n), dtype=dtype)
        for i in range(n_real):
            simax_i = simax[i]
            si_cur = 0.0
            for t in range(n):
                si[i, t] = si_cur
                ep_t = evap[i, t]
                pr_t = prec[i, t]
                ei_t = ep_t if _real(ep_t) < _real(si_cur) else si_cur
                si_next = si_cur + dt * (pr_t - ei_t)
                diff = si_next - simax_i
                if _real(diff) > 0.0:
                    pe_t = diff
                    si_next = si_next - diff
                else:
                    pe_t = 0.0
                ei[i, t] = -ei_t
                pi[i, t] = pr_t - pe_t
                si_cur = si_next
        return si, ei, pi

    @staticmethod
    @njit
    def get_snow_balance(
        prec: ArrayLike,
        temp: ArrayLike,
        tt: float | np.complex128 = 0.0,
        k: float | np.complex128 = 2.0,
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
            Storage in the snow reservoir.
        ps: array_like
            Snowfall flux in mm/d.
        m: array_like
            Snow melt flux in mm/d.

        Notes
        -----
        The water balance from the snow reservoir is as follows:

        .. math::

            \frac{dS_s}{dt} = P_s - M

        where $S_s$ [L] is the snow storage, $P_s$ [L/T] the snowfall, and $M$ [L/T]
        the snow melt from the snow reservoir.
        """
        ss, ps, m = _flex_snow_kernel(
            prec.reshape(1, -1), temp.reshape(1, -1), np.full(1, tt), np.full(1, k)
        )
        return ss[0], ps[0], m[0]

    @staticmethod
    @njit
    def _snow_kernel(
        prec: np.ndarray,
        temp: np.ndarray,
        tt: np.ndarray,
        k: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute snow reservoir water balance across realizations (2-D arrays)."""
        n_real, n = prec.shape
        dtype = np.asarray(prec[0, 0] + temp[0, 0] + tt[0] + k[0]).dtype
        ss = np.zeros((n_real, n), dtype=dtype)
        ps = np.zeros((n_real, n), dtype=dtype)
        m = np.zeros((n_real, n), dtype=dtype)
        for i in range(n_real):
            tt_i = tt[i]
            tt_real = _real(tt_i)
            k_i = k[i]
            ss_cur = 0.0
            for t in range(n):
                ss[i, t] = ss_cur
                temp_t = temp[i, t]
                if temp_t <= tt_real:
                    ps_t = prec[i, t]
                    m_t = 0.0
                else:
                    ps_t = 0.0
                    melt = k_i * (temp_t - tt_i) * (1.0 - np.exp(-(ss_cur / 1.5)))
                    m_t = melt if _real(melt) < _real(ss_cur) else ss_cur
                ps[i, t] = ps_t
                m[i, t] = -m_t
                ss_cur = ss_cur + ps_t - m_t
        return ss, ps, m

    def get_water_balance(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        temp: ArrayLike,
        p: ArrayLike,
        dt: float = 1.0,
    ) -> DataFrame:
        """Get the water balance for the FlexModel recharge model."""
        if np.ndim(p) > 1:
            raise NotImplementedError("get_water_balance is only supported for 1-D p.")
        data = self.simulate(
            prec=prec, evap=evap, temp=temp, p=p, dt=dt, return_full=True
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

    def check_snow_balance(self, prec: ArrayLike, temp: ArrayLike) -> float:
        """Check the water balance of the snow reservoir."""
        ss, ps, m = self.get_snow_balance(prec, temp)
        return ss[0] - ss[-1] + (ps + m).sum()

    def check_interception_balance(self, prec: ArrayLike, evap: ArrayLike) -> float:
        """Check the water balance of the interception reservoir."""
        si, ei, pi = self.get_interception_balance(prec, evap)
        return si[0] - si[-1] + (pi + ei).sum()

    def check_root_zone_balance(self, prec: ArrayLike, evap: ArrayLike) -> float:
        """Check the water balance of the root zone reservoir."""
        sr, r, ea, q, pe = self.get_root_zone_balance(prec, evap)
        return sr[0] - sr[-1] + (r + ea + q + pe).sum()

    def to_dict(self) -> dict[str, Any]:
        """Export the recharge model object to a dictionary."""
        return super().to_dict() | {
            "interception": self.interception,
            "snow": self.snow,
            "gw_uptake": self.gw_uptake,
        }


class Berendrecht(RechargeBase):
    r"""Nonlinear recharge to the groundwater.

    Calculated according to :cite:t:`berendrecht_non-linear_2006`.

    Notes
    -----
    Note that the preferred unit of the precipitation and evaporation is mm/d. The
    water balance for the unsaturated zone reservoir is written as:

    .. math::

        \frac{dS_e}{dt} = \frac{1}{D_e}(f_iP - E_a - R)

    where the recharge is calculated as:

    .. math::

        R(S_e) = K_sS_e^\lambda(1-(1-S_e^{1/m})^m)^2

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
        return DataFrame(
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

    def simulate(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        p: ArrayLike,
        dt: float = 1.0,
        return_full: bool = False,
    ) -> ArrayLike:
        """Simulate the recharge flux.

        Parameters
        ----------
        prec: array_like
            Precipitation flux in mm/d. Has to have the same length as evap.
        evap: array_like
            Potential evapotranspiration flux in mm/d.
        p: array_like
            Model parameters: shape (n_param,) for a single realization or
            (n_real, n_param) for an ensemble of realizations.
        dt: float, optional
            Time step for the calculation of the recharge. Only dt=1 is possible now.
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
        p_arr = np.asarray(p)
        is_1d = p_arr.ndim == 1
        P = np.atleast_2d(p_arr)
        n_real = P.shape[0]
        r, s, ea, pe = self._recharge_kernel(
            _prepare_2d_flux(prec, n_real),
            _prepare_2d_flux(evap, n_real),
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
            out = np.stack((r, s, ea, pe))
            return out[:, 0, :] if is_1d else out
        r_clean = np.nan_to_num(r)
        return r_clean[0] if is_1d else r_clean

    @staticmethod
    @njit
    def get_recharge(
        prec: ArrayLike,
        evap: ArrayLike,
        fi: float | np.complex128 = 1.0,
        fc: float | np.complex128 = 1.0,
        sr: float | np.complex128 = 0.5,
        de: float | np.complex128 = 250.0,
        l: float | np.complex128 = -2.0,
        m: float | np.complex128 = 0.5,
        ks: float | np.complex128 = 50.0,
        dt: float = 1.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Calculate recharge flux sped up with numba."""
        r, s, ea, pe = _berendrecht_kernel(
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
    def _recharge_kernel(
        prec: np.ndarray,
        evap: np.ndarray,
        fi: np.ndarray,
        fc: np.ndarray,
        sr: np.ndarray,
        de: np.ndarray,
        l: np.ndarray,
        m: np.ndarray,
        ks: np.ndarray,
        dt: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Berendrecht recharge across realizations (2-D arrays)."""
        n_real, n = prec.shape
        dtype = np.asarray(
            prec[0, 0]
            + evap[0, 0]
            + fi[0]
            + fc[0]
            + sr[0]
            + de[0]
            + l[0]
            + m[0]
            + ks[0]
        ).dtype
        s = np.zeros((n_real, n), dtype=dtype)
        r = np.zeros((n_real, n), dtype=dtype)
        ea = np.zeros((n_real, n), dtype=dtype)
        pe = np.zeros((n_real, n), dtype=dtype)
        for i in range(n_real):
            fi_i = fi[i]
            fc_i = fc[i]
            sr_i = sr[i]
            de_i = de[i]
            l_i = l[i]
            m_i = m[i]
            inv_m_i = 1.0 / m_i
            ks_i = ks[i]
            s_cur = 0.5
            for t in range(n - 1):
                pe_t = fi_i * prec[i, t]
                ep_t = fc_i * evap[i, t]
                s_real = _real(s_cur)
                if s_real < 0.05:
                    s_cur = 0.05 * np.exp(20.0 * s_cur - 1.0)
                elif s_real > 0.95:
                    s_cur = 1.0 - 0.05 * np.exp(19.0 - 20.0 * s_cur)
                ea_t = (1.0 - np.exp(-3.0 * s_cur / sr_i)) * ep_t
                r_t = ks_i * (s_cur**l_i) * (1.0 - (1.0 - s_cur**inv_m_i) ** m_i) ** 2
                s[i, t] = s_cur
                pe[i, t] = pe_t
                ea[i, t] = ea_t
                r[i, t] = r_t
                s_cur = s_cur + (dt / de_i) * (pe_t - ea_t - r_t)
            s[i, n - 1] = s_cur
            pe[i, n - 1] = fi_i * prec[i, n - 1]
        return r, s, ea, pe

    def get_water_balance(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        p: ArrayLike,
        dt: float = 1.0,
    ) -> DataFrame:
        """Get the water balance for the Berendrecht recharge model."""
        if np.ndim(p) > 1:
            raise NotImplementedError("get_water_balance is only supported for 1-D p.")
        r, s, ea, pe = self.simulate(prec=prec, evap=evap, p=p, dt=dt, return_full=True)
        s = s * p[3]  # Because S is computed dimensionless in this model
        return DataFrame(
            data=np.vstack((s, pe, ea, r)).T, columns=["S", "Pe", "Ea", "R"]
        )


class Peterson(RechargeBase):
    r"""Nonlinear recharge to the groundwater.

    Calculated based on :cite:t:`peterson_nonlinear_2014`.

    The water balance for the unsaturated zone reservoir is written as:

    .. math::

        \frac{dS}{dt} = P_e - E_a - R

    where the fluxes $P_e$, $E_a$ and $R$ are calculated as:

    .. math::

        P_e = P \left(1 - \frac{S}{\hat{S_{cap}}}\right)^\alpha

    .. math::

        E_a = E_p \left(\frac{S}{\hat{S_{cap}}}\right)^\gamma

    .. math::

        R = \hat{k_{sat}}\left(\frac{S}{\hat{S_{cap}}}\right)^{\hat{\beta}}

    with the parameters:

    .. math::

        \hat{S_{cap}} = 10^{S_{cap}}; \hat{k_{sat}} = 10^{k_{sat}}; \hat{\beta} =
        10^{\beta}

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
        return DataFrame(
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

    def simulate(
        self,
        prec: ArrayLike,
        evap: ArrayLike,
        p: ArrayLike,
        dt: float = 1.0,
        return_full: bool = False,
    ) -> ArrayLike:
        """Simulate the recharge flux.

        Parameters
        ----------
        prec: array_like
            Precipitation flux in mm/d. Must have the same length as evap.
        evap: array_like
            Potential evaporation flux in mm/d.
        p: array_like
            Model parameters: shape (n_param,) for a single realization or
            (n_real, n_param) for an ensemble of realizations.
        dt: float, optional
            Time step for the calculation of the recharge. Only dt=1 is possible now.
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
        p_arr = np.asarray(p)
        is_1d = p_arr.ndim == 1
        P = np.atleast_2d(p_arr)
        n_real = P.shape[0]
        r, s, ea, pe = self._recharge_kernel(
            _prepare_2d_flux(prec, n_real),
            _prepare_2d_flux(evap, n_real),
            P[:, 0],
            P[:, 1],
            P[:, 2],
            P[:, 3],
            P[:, 4],
            dt,
        )
        if return_full:
            out = np.stack((r, s, ea, pe))
            return out[:, 0, :] if is_1d else out
        r_clean = np.nan_to_num(r)
        return r_clean[0] if is_1d else r_clean

    @staticmethod
    @njit
    def get_recharge(
        prec: ArrayLike,
        evap: ArrayLike,
        scap: float | np.complex128 = 1.0,
        alpha: float | np.complex128 = 1.0,
        ksat: float | np.complex128 = 1.0,
        beta: float | np.complex128 = 0.5,
        gamma: float | np.complex128 = 1.0,
        dt: float = 1.0,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Calculate recharge flux sped up with numba."""
        r, sm, ea, pe = _peterson_kernel(
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
    def _recharge_kernel(
        prec: np.ndarray,
        evap: np.ndarray,
        scap: np.ndarray,
        alpha: np.ndarray,
        ksat: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        dt: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Peterson recharge across realizations (2-D arrays)."""
        n_real, n = prec.shape
        dtype = np.asarray(
            prec[0, 0] + evap[0, 0] + scap[0] + alpha[0] + ksat[0] + beta[0] + gamma[0]
        ).dtype
        pe = np.zeros((n_real, n), dtype=dtype)
        sm = np.zeros((n_real, n), dtype=dtype)
        r = np.zeros((n_real, n), dtype=dtype)
        ea = np.zeros((n_real, n), dtype=dtype)
        for i in range(n_real):
            smsc_i = 10.0 ** scap[i]
            alpha_i = alpha[i]
            ksat_i = 10.0 ** ksat[i]
            beta_i = 10.0 ** beta[i]
            gamma_i = gamma[i]
            sm_cur = 0.5 * smsc_i
            for t in range(n):
                sm_frac = sm_cur / smsc_i
                pe_t = prec[i, t] * ((1.0 - sm_frac) ** alpha_i)
                ea_val = evap[i, t] * (sm_frac**gamma_i)
                ea_t = ea_val if _real(ea_val) > 0.0 else 0.0
                r_val = ksat_i * (sm_frac**beta_i)
                r_t = r_val if _real(r_val) > 0.0 else 0.0
                sm_new = sm_cur + (pe_t - ea_t - r_t) * dt
                if _real(sm_new) < 0.0:
                    sm_cur = 0.0
                elif _real(sm_new) > _real(smsc_i):
                    sm_cur = smsc_i
                else:
                    sm_cur = sm_new
                pe[i, t] = pe_t
                ea[i, t] = ea_t
                r[i, t] = r_t
                sm[i, t] = sm_cur
        return r, sm, ea, pe

    def get_water_balance(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, dt: float = 1.0
    ) -> DataFrame:
        """Get the water balance for the Peterson recharge model."""
        if np.ndim(p) > 1:
            raise NotImplementedError("get_water_balance is only supported for 1-D p.")
        r, s, ea, pe = self.simulate(prec=prec, evap=evap, p=p, dt=dt, return_full=True)
        return DataFrame(
            data=np.vstack((s, pe, ea, r)).T, columns=["S", "Pe", "Ea", "R"]
        )


# Module-level aliases required by Numba to resolve jitted callees inside get_*
_flex_root_zone_kernel = FlexModel._root_zone_kernel
_flex_interception_kernel = FlexModel._interception_kernel
_flex_snow_kernel = FlexModel._snow_kernel
_berendrecht_kernel = Berendrecht._recharge_kernel
_peterson_kernel = Peterson._recharge_kernel
