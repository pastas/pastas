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
            Evaporation flux values.
        p : ArrayLike
            Recharge model parameters (1-D or 2-D array).

        Returns
        -------
        ArrayLike
            Recharge flux (mm/d) for 1-D or 2-D `p`.
        """
        return np.add(prec, np.multiply(evap, np.asarray(p)))

    def get_water_balance(
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, **kwargs
    ) -> DataFrame:
        """Get the water balance for the linear recharge model."""
        _ = kwargs
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
    ) -> ArrayLike:
        """Simulate the soil water balance model for 1-D or 2-D `p`.

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
        p_arr = np.asarray(p)
        is_1d = p_arr.ndim == 1
        P = np.atleast_2d(p_arr)

        prec_arr = np.asarray(prec)
        evap_arr = np.asarray(evap)
        ep = evap_arr[None, :] * P[:, 4:5]

        if self.snow:
            ss, ps, m = self.get_snow_balance(
                prec=prec_arr, temp=temp, tt=P[:, -2], k=P[:, -1]
            )
            pr = prec_arr[None, :] - ps
        else:
            pr = prec_arr
            m = 0.0

        if self.interception:
            si, ei, pi = self.get_interception_balance(pr=pr, ep=ep, simax=P[:, 5])
            ep = ep + ei
            pe = (pr[None, :] if np.ndim(pr) == 1 else pr) - pi
        else:
            pe = pr

        pe_in = (pe[None, :] if np.ndim(pe) == 1 else pe) - m
        sr, r, ea, q, pe_out = self.get_root_zone_balance(
            pe=pe_in, ep=ep, srmax=P[:, 0], lp=P[:, 1], ks=P[:, 2], gamma=P[:, 3], dt=dt
        )

        # Report big water balance errors (error > 0.1%)
        error = (sr[:, 0] - sr[:, -1] + (pe_out + r + ea + q).sum(axis=1)) / (
            pe_out.sum(axis=1) + 1e-10
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
            data = (sr, -r, ea, q, pe_out)
            if self.interception:
                data += (si, ei, pi)
            if self.snow:
                data += (ss, ps, m)
            out = np.stack(data)
            return out[:, 0, :] if is_1d else out

        return -r[0] if is_1d else -r

    @classmethod
    def get_root_zone_balance(
        cls,
        pe: ArrayLike,
        ep: ArrayLike,
        srmax: float | np.complex128 | ArrayLike = 250.0,
        lp: float | np.complex128 | ArrayLike = 0.25,
        ks: float | np.complex128 | ArrayLike = 100.0,
        gamma: float | np.complex128 | ArrayLike = 4.0,
        dt: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the water balance of the root zone reservoir (1-D or 2-D)."""
        is_1d = (
            np.ndim(pe) == 1
            and np.ndim(ep) == 1
            and np.ndim(srmax) == 0
            and np.ndim(lp) == 0
            and np.ndim(ks) == 0
            and np.ndim(gamma) == 0
        )
        srmax_1d = np.atleast_1d(srmax)
        lp_1d = np.atleast_1d(lp)
        ks_1d = np.atleast_1d(ks)
        gamma_1d = np.atleast_1d(gamma)
        n_real = max(
            srmax_1d.shape[0],
            lp_1d.shape[0],
            ks_1d.shape[0],
            gamma_1d.shape[0],
            np.atleast_2d(pe).shape[0],
            np.atleast_2d(ep).shape[0],
        )
        pe_2d = _prepare_2d_flux(pe, n_real)
        ep_2d = _prepare_2d_flux(ep, n_real)
        dtype = np.result_type(
            pe_2d, ep_2d, srmax_1d, lp_1d, ks_1d, gamma_1d, np.float64
        )
        sr, r, ea, q = cls._root_zone_kernel(
            pe_2d,
            ep_2d,
            np.broadcast_to(srmax_1d, (n_real,)),
            np.broadcast_to(lp_1d, (n_real,)),
            np.broadcast_to(ks_1d, (n_real,)),
            np.broadcast_to(gamma_1d, (n_real,)),
            dt,
            dtype,
        )
        if is_1d:
            return sr[0], r[0], ea[0], q[0], pe_2d[0]
        return sr, r, ea, q, pe_2d

    @staticmethod
    @njit
    def _root_zone_kernel(pe, ep, srmax, lp, ks, gamma, dt, dtype):
        n_real, n = pe.shape
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

    @classmethod
    def get_interception_balance(
        cls,
        pr: ArrayLike,
        ep: ArrayLike,
        simax: float | np.complex128 | ArrayLike = 2.0,
        dt: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the water balance of the interception reservoir (1-D or 2-D)."""
        is_1d = np.ndim(pr) == 1 and np.ndim(ep) == 1 and np.ndim(simax) == 0
        simax_1d = np.atleast_1d(simax)
        n_real = max(
            simax_1d.shape[0], np.atleast_2d(pr).shape[0], np.atleast_2d(ep).shape[0]
        )
        pr_2d = _prepare_2d_flux(pr, n_real)
        ep_2d = _prepare_2d_flux(ep, n_real)
        dtype = np.result_type(pr_2d, ep_2d, simax_1d, np.float64)
        si, ei, pi = cls._interception_kernel(
            pr_2d, ep_2d, np.broadcast_to(simax_1d, (n_real,)), dt, dtype
        )
        if is_1d:
            return si[0], ei[0], pi[0]
        return si, ei, pi

    @staticmethod
    @njit
    def _interception_kernel(pr, ep, simax, dt, dtype):
        n_real, n = pr.shape
        si = np.zeros((n_real, n), dtype=dtype)
        ei = np.zeros((n_real, n), dtype=dtype)
        pi = np.zeros((n_real, n), dtype=dtype)
        for i in range(n_real):
            simax_i = simax[i]
            si_cur = 0.0
            for t in range(n):
                si[i, t] = si_cur
                ep_t = ep[i, t]
                pr_t = pr[i, t]
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

    @classmethod
    def get_snow_balance(
        cls,
        prec: ArrayLike,
        temp: ArrayLike,
        tt: float | np.complex128 | ArrayLike = 0.0,
        k: float | np.complex128 | ArrayLike = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the water balance of the snow reservoir (1-D or 2-D)."""
        is_1d = (
            np.ndim(prec) == 1
            and np.ndim(temp) == 1
            and np.ndim(tt) == 0
            and np.ndim(k) == 0
        )
        tt_1d = np.atleast_1d(tt)
        k_1d = np.atleast_1d(k)
        n_real = max(
            tt_1d.shape[0],
            k_1d.shape[0],
            np.atleast_2d(prec).shape[0],
            np.atleast_2d(temp).shape[0],
        )
        prec_2d = _prepare_2d_flux(prec, n_real)
        temp_2d = _prepare_2d_flux(temp, n_real)
        dtype = np.result_type(prec_2d, temp_2d, tt_1d, k_1d, np.float64)
        ss, ps, m = cls._snow_kernel(
            prec_2d,
            temp_2d,
            np.broadcast_to(tt_1d, (n_real,)),
            np.broadcast_to(k_1d, (n_real,)),
            dtype,
        )
        if is_1d:
            return ss[0], ps[0], m[0]
        return ss, ps, m

    @staticmethod
    @njit
    def _snow_kernel(prec, temp, tt, k, dtype):
        n_real, n = prec.shape
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
        return ss[0] - ss[-1] + (ps + m).sum()

    def check_interception_balance(
        self, prec: ArrayLike, evap: ArrayLike, **kwargs
    ) -> float:
        """Check the water balance of the interception reservoir."""
        si, ei, pi = self.get_interception_balance(prec, evap)
        return si[0] - si[-1] + (pi + ei).sum()

    def check_root_zone_balance(
        self, prec: ArrayLike, evap: ArrayLike, **kwargs
    ) -> float:
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
        """Simulate the recharge flux for 1-D or 2-D `p`."""
        p_arr = np.asarray(p)
        is_1d = p_arr.ndim == 1
        P = np.atleast_2d(p_arr)
        r, s, ea, pe = self.get_recharge(
            prec,
            evap,
            fi=P[:, 0],
            fc=P[:, 1],
            sr=P[:, 2],
            de=P[:, 3],
            l=P[:, 4],
            m=P[:, 5],
            ks=P[:, 6],
            dt=dt,
        )
        if return_full:
            out = np.stack((r, s, ea, pe))
            return out[:, 0, :] if is_1d else out
        r_clean = np.nan_to_num(r)
        return r_clean[0] if is_1d else r_clean

    @classmethod
    def get_recharge(
        cls,
        prec: ArrayLike,
        evap: ArrayLike,
        fi: float | np.complex128 | ArrayLike = 1.0,
        fc: float | np.complex128 | ArrayLike = 1.0,
        sr: float | np.complex128 | ArrayLike = 0.5,
        de: float | np.complex128 | ArrayLike = 250.0,
        l: float | np.complex128 | ArrayLike = -2.0,
        m: float | np.complex128 | ArrayLike = 0.5,
        ks: float | np.complex128 | ArrayLike = 50.0,
        dt: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate recharge flux sped up with numba (1-D or 2-D)."""
        params = [np.atleast_1d(x) for x in (fi, fc, sr, de, l, m, ks)]
        is_1d = (
            np.ndim(prec) == 1
            and np.ndim(evap) == 1
            and all(np.ndim(x) == 0 for x in (fi, fc, sr, de, l, m, ks))
        )
        n_real = max(
            *(x.shape[0] for x in params),
            np.atleast_2d(prec).shape[0],
            np.atleast_2d(evap).shape[0],
        )
        prec_2d = _prepare_2d_flux(prec, n_real)
        evap_2d = _prepare_2d_flux(evap, n_real)
        dtype = np.result_type(prec_2d, evap_2d, *params, np.float64)
        b_params = [np.broadcast_to(x, (n_real,)) for x in params]
        r, s, ea, pe = cls._recharge_kernel(prec_2d, evap_2d, *b_params, dt, dtype)
        if is_1d:
            return r[0], s[0], ea[0], pe[0]
        return r, s, ea, pe

    @staticmethod
    @njit
    def _recharge_kernel(prec, evap, fi, fc, sr, de, l, m, ks, dt, dtype):
        n_real, n = prec.shape
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
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, dt: float = 1.0, **kwargs
    ) -> DataFrame:
        """Get the water balance for the Berendrecht recharge model."""
        if np.ndim(p) > 1:
            raise NotImplementedError("get_water_balance is only supported for 1-D p.")
        r, s, ea, pe = self.simulate(prec, evap, p=p, dt=dt, return_full=True, **kwargs)
        s = s * p[3]
        return DataFrame(
            data=np.vstack((s, pe, ea, r)).T, columns=["S", "Pe", "Ea", "R"]
        )


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
        """Simulate the recharge flux for 1-D or 2-D `p`.

        Parameters
        ----------
        prec: array_like
            Precipitation flux in mm/d. Must have the same length as evap.
        evap: array_like
            Potential evaporation flux in mm/d.
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
            `p`, (n_real, n_points) for a 2-D `p` of shape (n_real, n_param). With
            `return_full=True`, all fluxes and  states are stacked along the first
            axis: (n_fluxes, n_points) for a 1-D `p` and (n_fluxes, n_real, n_points)
            for a 2-D `p`.
        """
        p_arr = np.asarray(p)
        is_1d = p_arr.ndim == 1
        P = np.atleast_2d(p_arr)
        r, s, ea, pe = self.get_recharge(
            prec,
            evap,
            scap=P[:, 0],
            alpha=P[:, 1],
            ksat=P[:, 2],
            beta=P[:, 3],
            gamma=P[:, 4],
            dt=dt,
        )
        if return_full:
            out = np.stack((r, s, ea, pe))
            return out[:, 0, :] if is_1d else out
        r_clean = np.nan_to_num(r)
        return r_clean[0] if is_1d else r_clean

    @classmethod
    def get_recharge(
        cls,
        prec: ArrayLike,
        evap: ArrayLike,
        scap: float | np.complex128 | ArrayLike = 1.0,
        alpha: float | np.complex128 | ArrayLike = 1.0,
        ksat: float | np.complex128 | ArrayLike = 1.0,
        beta: float | np.complex128 | ArrayLike = 0.5,
        gamma: float | np.complex128 | ArrayLike = 1.0,
        dt: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate recharge flux sped up with numba (1-D or 2-D)."""
        params = [np.atleast_1d(x) for x in (scap, alpha, ksat, beta, gamma)]
        is_1d = (
            np.ndim(prec) == 1
            and np.ndim(evap) == 1
            and all(np.ndim(x) == 0 for x in (scap, alpha, ksat, beta, gamma))
        )
        n_real = max(
            *(x.shape[0] for x in params),
            np.atleast_2d(prec).shape[0],
            np.atleast_2d(evap).shape[0],
        )
        prec_2d = _prepare_2d_flux(prec, n_real)
        evap_2d = _prepare_2d_flux(evap, n_real)
        dtype = np.result_type(prec_2d, evap_2d, *params, np.float64)
        b_params = [np.broadcast_to(x, (n_real,)) for x in params]
        r, sm, ea, pe = cls._recharge_kernel(prec_2d, evap_2d, *b_params, dt, dtype)
        if is_1d:
            return r[0], sm[0], ea[0], pe[0]
        return r, sm, ea, pe

    @staticmethod
    @njit
    def _recharge_kernel(prec, evap, scap, alpha, ksat, beta, gamma, dt, dtype):
        n_real, n = prec.shape
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
        self, prec: ArrayLike, evap: ArrayLike, p: ArrayLike, dt: float = 1.0, **kwargs
    ) -> DataFrame:
        """Get the water balance for the Peterson recharge model."""
        if np.ndim(p) > 1:
            raise NotImplementedError("get_water_balance is only supported for 1-D p.")
        r, s, ea, pe = self.simulate(prec, evap, p=p, dt=dt, return_full=True, **kwargs)
        return DataFrame(
            data=np.vstack((s, pe, ea, r)).T, columns=["S", "Pe", "Ea", "R"]
        )
