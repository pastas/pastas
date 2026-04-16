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
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.ml: Model | None = None

    def get_init_parameters(self) -> pd.DataFrame:
        """Get the initial parameters of the solver.

        Returns
        -------
        init_parameters: pd.DataFrame
            DataFrame with the initial parameters of the solver.
        """
        parameters = pd.DataFrame(
            [],
            index=[],
            columns=["initial", "pmin", "pmax", "vary", "name"],
        )
        return parameters

    def set_model(self, ml: Model):
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

    def to_dict(self) -> dict:
        return {"class": self._name, "name": self.name}

    @abstractmethod
    def fit_report(self) -> None:
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
