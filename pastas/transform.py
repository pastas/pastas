"""Nonlinear transforms for time series models.

These transforms are applied after the simulation, to incorporate nonlinear effects.

Examples
--------
Add a threshold transform to a model::

    transform = ps.ThresholdTransform()
    ml.add_transform(transform)

"""

import numpy as np
from pandas import DataFrame, Series

from pastas.typing import ArrayLike, Model

from .decorators import check_argument_model, deprecate_args_or_kwargs, set_parameter
from .utils import validate_name


class ThresholdTransform:
    """ThresholdTransform lowers the simulation when it exceeds a certain value.

    Parameters
    ----------
    model : Model
        Instance of a Pastas Model to which the Transform is added.
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

    @check_argument_model
    def __init__(
        self,
        model: Model,
        value: float = np.nan,
        vmin: float = np.nan,
        vmax: float = np.nan,
        name: str = "transform",
        nparam: int = 2,
    ) -> None:
        self.model = model
        self.value = value
        self.vmin = vmin
        self.vmax = vmax
        self.name = validate_name(name)
        self._nparam = nparam
        self.parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        if self.model is not None:
            self.model._add_transform(self)
            self.set_init_parameters()

    def set_model(self, model: Model) -> None:
        """Set model observations and initialize parameters."""
        self.model = model
        self.set_init_parameters()

    @property
    def nparam(self) -> int:
        """Number of parameters."""
        return self._nparam

    @nparam.setter
    def nparam(self, value: int) -> None:
        if hasattr(self, "_nparam"):
            raise AttributeError("nparam can only be set during initialization.")
        self._nparam = value

    def set_init_parameters(self) -> None:
        """Set the initial parameter values in the parameters DataFrame."""
        obs = self.model.observations()

        if np.isnan(self.value):
            self.value = obs.min() + 0.75 * (obs.max() - obs.min())
        if np.isnan(self.vmin):
            self.vmin = obs.min() + 0.5 * (obs.max() - obs.min())
        if np.isnan(self.vmax):
            self.vmax = obs.max()

        self.parameters.loc[self.name + "_d"] = (
            self.value,
            self.vmin,
            self.vmax,
            True,
            self.name,
        )
        if self.nparam == 2:
            self.parameters.loc[self.name + "_f"] = (0.5, 0.0, 1.0, True, self.name)

    @set_parameter
    def _set_initial(self, name: str, value: float) -> None:
        """Set the initial parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "initial"] = value

    @set_parameter
    def _set_pmin(self, name: str, value: float) -> None:
        """Set the lower bound of the parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "pmin"] = value

    @set_parameter
    def _set_pmax(self, name: str, value: float) -> None:
        """Set the upper bound of the parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "pmax"] = value

    @set_parameter
    def _set_vary(self, name: str, value: float) -> None:
        """Set if the parameter is varied during optimization.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name, "vary"] = bool(value)

    @set_parameter
    def _set_dist(self, name: str, value: str) -> None:
        """Set distribution of prior of the parameter.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.at[name] = str(value)

    def simulate(
        self,
        series: Series | ArrayLike | None = None,
        p: ArrayLike | None = None,
        **kwargs,
    ) -> Series:
        """Apply the threshold transform to the series.

        Parameters
        ----------
        series : pandas.Series
            The series to transform.
        p : ArrayLike
            The parameters for the transform.

        Returns
        -------
        pandas.Series
            The transformed series.
        """
        if "h" in kwargs:
            deprecate_args_or_kwargs(
                name="h",
                version="2.4.0",
                reason="Please use `series` instead of `h`.",
            )
            if series is None:
                series = kwargs.pop("h")
            else:
                kwargs.pop("h")
        if kwargs:
            raise TypeError(
                f"simulate() got unexpected keyword argument '{next(iter(kwargs))}'"
            )
        if series is None:
            raise TypeError("simulate() missing required argument: 'series'")
        if p is None:
            raise TypeError("simulate() missing required argument: 'p'")

        if self.nparam == 1:
            # value above a threshold p[0] are equal to the threshold
            series[series > p[0]] = p[0]
        elif self.nparam == 2:
            # values above a threshold p[0] are scaled by p[1]
            mask = series > p[0]
            series[mask] = p[0] + p[1] * (series[mask] - p[0])
        else:
            raise ValueError("Not yet implemented yet")
        return series

    @property
    def _name(self) -> str:
        return self.__class__.__name__

    def to_dict(self) -> dict:
        """Return the transform as a dictionary.

        Returns
        -------
        dict
            Dictionary with the transform properties.
        """
        data = {
            "class": self._name,
            "value": self.value,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "name": self.name,
            "nparam": self.nparam,
        }
        return data
