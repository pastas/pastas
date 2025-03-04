"""This module contains all the transforms that can be added to a model.

These transforms are applied after the simulation, to incorporate nonlinear effects.
"""

import numpy as np
from pandas import DataFrame, Series

# Type Hinting
from pastas.typing import ArrayLike, Model

from .decorators import set_parameter
from .utils import validate_name


class ThresholdTransform:
    """ThresholdTransform lowers the simulation when it exceeds a certain value.

    Parameters
    ----------
    value : float, optional
        The initial starting value above which the simulation is lowered.
    vmin : float, optional
        The minimum value above which the simulation is lowered.
    vmax : float, optional
        The maximum value above which the simulation is lowered.
    name: str, optional
        Name of the transform.
    nparam : int, optional
        The number of parameters. Default is nparam=2. The first parameter then is
        the threshold, and the second parameter is the factor with which the
        simulation is lowered.

    Notes
    -----
    In geohydrology this transform can be used in a situation where the groundwater
    level reaches the surface level and forms a lake. Because of the larger storage
    of the lake, the (groundwater) level then rises slower when it rains.
    """

    _name = "ThresholdTransform"

    def __init__(
        self,
        value: float = np.nan,
        vmin: float = np.nan,
        vmax: float = np.nan,
        name: str = "ThresholdTransform",
        nparam: int = 2,
    ) -> None:
        self.value = value
        self.vmin = vmin
        self.vmax = vmax
        self.name = validate_name(name)
        self.nparam = nparam
        self.parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])

    def set_model(self, ml: Model) -> None:
        obs = ml.observations()
        if np.isnan(self.value):
            self.value = obs.min() + 0.75 * (obs.max() - obs.min())
        if np.isnan(self.vmin):
            self.vmin = obs.min() + 0.5 * (obs.max() - obs.min())
        if np.isnan(self.vmax):
            self.vmax = obs.max()
        self.set_init_parameters()

    def set_init_parameters(self) -> None:
        self.parameters.loc[self.name + "_1"] = (
            self.value,
            self.vmin,
            self.vmax,
            True,
            self.name,
        )
        if self.nparam == 2:
            self.parameters.loc[self.name + "_2"] = (0.5, 0.0, 1.0, True, self.name)

    @set_parameter
    def _set_initial(self, name: str, value: float) -> None:
        """Internal method to set the initial parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "initial"] = value

    @set_parameter
    def _set_pmin(self, name: str, value: float) -> None:
        """Internal method to set the lower bound of the parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "pmin"] = value

    @set_parameter
    def _set_pmax(self, name: str, value: float) -> None:
        """Internal method to set the upper bound of the parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "pmax"] = value

    @set_parameter
    def _set_vary(self, name: str, value: float) -> None:
        """Internal method to set if the parameter is varied during optimization.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "vary"] = bool(value)

    @set_parameter
    def _set_dist(self, name: str, value: str) -> None:
        """Internal method to set distribution of prior of the parameter.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "dist"] = str(value)

    def simulate(self, h: Series, p: ArrayLike) -> Series:
        if self.nparam == 1:
            # value above a threshold p[0] are equal to the threshold
            h[h > p[0]] = p[0]
        elif self.nparam == 2:
            # values above a threshold p[0] are scaled by p[1]
            mask = h > p[0]
            h[mask] = p[0] + p[1] * (h[mask] - p[0])
        else:
            raise ValueError("Not yet implemented yet")
        return h

    def to_dict(self) -> dict:
        data = {
            "class": self._name,
            "value": self.value,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "name": self.name,
            "nparam": self.nparam,
        }
        return data
