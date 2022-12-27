# Type hinting for Pastas library

# Base Classes for TypeVar
# Internal
from pastas.noisemodels import NoiseModelBase
from pastas.stressmodels import StressModelBase
from pastas.solver import BaseSolver
from pastas.timeseries import TimeSeries
from pastas.model import Model
from pastas.plots import TrackSolve
from pastas.timer import SolveTimer
from pastas.rfunc import RfuncBase
from pastas.reservoir import ReservoirBase

# External
# from pandas import Series, DataFrame
from pandas import Timestamp
from matplotlib.axes._base import _AxesBase
from matplotlib import FigureBase

from numpy.typing import ArrayLike
from typing import Type, Optional, Union, TypeVar, Tuple, Any
pstAx = TypeVar("pstAx", bound=_AxesBase)  # Matplotlib Axes
pstFi = TypeVar("pstFi", bound=FigureBase)  # Matplotlib Figure
# pstS = TypeVar("pstS", bound=Type[Series])
# pstDF = TypeVar("pstDF", bound=Type[DataFrame])
pstTm = TypeVar("pstTm", bound=Union[str, Timestamp])  # Tmin or Tmax
pstMl = TypeVar("pstMl", bound=Model)  # Model
pstTS = TypeVar("pstTS", bound=TimeSeries)  # Time Series
pstSM = TypeVar("pstSM", bound=StressModelBase)  # Stress Model
pstNM = TypeVar("pstNM", bound=NoiseModelBase)  # Noise Model
pstBS = TypeVar("pstBS", bound=BaseSolver)  # Base Solver
pstRB = TypeVar("pstRB", bound=ReservoirBase)  # Reservoir Base
pstCB = TypeVar("pstCB", bound=Union[SolveTimer, TrackSolve])  # Callback
pstFu = TypeVar("pstFu")  # Function (e.g. Objective Function)
pstRF = TypeVar("pstRF", bound=RfuncBase)  # rFunc Base
pstAL = TypeVar("pstAL", bound=Type[ArrayLike])  # Array Like (NumPy based)
