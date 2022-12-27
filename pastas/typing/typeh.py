# Type hinting for Pastas library

# Base Classes for TypeVar
# Internal
import pastas as ps
# from pastas.noisemodels import NoiseModelBase
# from pastas.stressmodels import StressModelBase
# from pastas.solver import BaseSolver
# from pastas.timeseries import TimeSeries
# from pastas.model import Model
# from pastas.plots import TrackSolve
# from pastas.timer import SolveTimer
# from pastas.rfunc import RfuncBase
# from pastas.reservoir import ReservoirBase
# from pastas.recharge import RechargeBase

# External libraries
# Pandas
from pandas import Timestamp  # Series, DataFrame
# Matplotlib
from matplotlib.axes._base import _AxesBase
from matplotlib.figure import FigureBase
# Numpy
from numpy.typing import ArrayLike
# Typing
from typing import Type, Union, Optional, Tuple, Any, TypeVar

pstAx = TypeVar("pstAx", bound=_AxesBase)  # Matplotlib Axes
pstFi = TypeVar("pstFi", bound=FigureBase)  # Matplotlib Figure
# pstS = TypeVar("pstS", bound=Type[Series])
# pstDF = TypeVar("pstDF", bound=Type[DataFrame])
pstTm = TypeVar("pstTm", bound=Union[str, Timestamp])  # Tmin or Tmax
pstMl = TypeVar("pstMl", bound=Any)  # Model
pstTS = TypeVar("pstTS", bound=Any)  # Time Series
pstSM = TypeVar("pstSM", bound=Any)  # Stress Model
pstNM = TypeVar("pstNM", bound=Any)  # Noise Model
pstBS = TypeVar("pstBS", bound=Any)  # Base Solver
pstRB = TypeVar("pstRB", bound=Any)  # Recharge Base
pstRV = TypeVar("pstRV", bound=Any)  # Reservoir Base
pstCB = TypeVar("pstCB", bound=Any)  # Callback
pstFu = TypeVar("pstFu", bound=Any)  # Function (e.g. Objective Function)
pstRF = TypeVar("pstRF", bound=Any)  # rFunc Base
pstAL = TypeVar("pstAL", bound=Type[ArrayLike])  # Array Like (NumPy based)
