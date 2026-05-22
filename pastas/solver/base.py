"""This module contains the base solver that used available for Pastas.

All solvers inherit from the SolverBase class, which contains general method for
selecting the correct time series to misfit and options to weight the residuals or
noise series.
"""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any

import pandas as pd

from pastas.decorators import PastasDeprecationWarning, set_parameter
from pastas.typing import Model

logger = getLogger(__name__)


class SolverBase(ABC):
    """All solver instances inherit from the SolverBase class.

    Attributes
    ----------
    name: str
        Name of the solver instance.
    ml: pastas.Model
        The Pastas Model instance that is being solved.
    parameters: pd.DataFrame
        DataFrame with the initial parameters of the solver itself.

    """

    def __init__(self, name: str = "solver", **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.ml: Model | None = None
        self.parameters: pd.DataFrame | None = None
        self.set_init_parameters()  # adds self.Parameters DataFrame

    def get_init_parameters(self, name: str) -> pd.DataFrame:
        """Get the initial parameters of the solver.

        Returns
        -------
        init_parameters: pd.DataFrame
            DataFrame with the initial parameters of the solver.
        """
        _ = name
        parameters = pd.DataFrame(
            [],
            index=[],
            columns=["initial", "pmin", "pmax", "vary", "name"],
        )
        return parameters

    def set_init_parameters(self) -> None:
        """Set the initial parameters (back) to their default values."""
        self.parameters = self.get_init_parameters(name=self.name)

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

    def set_model(self, ml: Model) -> None:
        """Method to set the Pastas Model instance.

        Parameters
        ----------
        ml: pastas.Model instance

        """
        if self.ml is not None:
            raise UserWarning(
                "This solver instance is already used by another model. Please create "
                "a separate solver instance for each Pastas Model."
            )
        self.ml = ml

    @abstractmethod
    def fit_report(self, all_options: bool = False) -> str:
        """Abstract method that has to be implemented by all solvers."""
        pass

    @abstractmethod
    def solve(self) -> Any:
        """Solve method that has to be implemented by all solvers."""
        pass

    @property
    def _name(self) -> str:
        """Name of the solver class."""
        return self.__class__.__name__

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the solver instance.

        Notes
        -----
        This method is used to store Pastas models.
        """
        return {
            "class": self._name,
            "name": self.name,
            "kwargs": self.kwargs,
        }


@PastasDeprecationWarning(
    version="2.0.0", reason="Use SolverBase instead of BaseSolver."
)
class BaseSolver(SolverBase):
    """BaseSolver is deprecated and will be removed in a future version of Pastas.

    Please use the SolverBase class instead, which provides a more structured and
    flexible approach to implementing solvers in Pastas. The SolverBase class includes
    methods for setting initial parameters, bounds, and whether parameters should be
    varied during optimization, as well as a method for associating the solver with a
    Pastas Model instance.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
