"""This module contains the base solver that used available for Pastas.

All solvers inherit from the BaseSolver class, which contains general method for
selecting the correct time series to misfit and options to weight the residuals or
noise series.
"""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any

import pandas as pd

from pastas.typing import Model

logger = getLogger(__name__)


class BaseSolver(ABC):
    """All solver instances inherit from the BaseSolver class.

    Attributes
    ----------
    name: str
        Name of the solver instance.
    ml: pastas.Model
        The Pastas Model instance that is being solved.
    parameters: pd.DataFrame
        DataFrame with the initial parameters of the solver.

    """

    def __init__(self, name: str = "solver", **kwargs) -> None:
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
    def fit_report(self) -> str:
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

    def to_dict(self) -> dict:
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
