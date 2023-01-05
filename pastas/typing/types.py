# Type hinting for Pastas library

# Base Classes for TypeVar
# Internal
# import pastas as ps
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
from typing import Union, Any, TypeVar

Axes = TypeVar("Axes", bound=_AxesBase)  # Matplotlib Axes
Figure = TypeVar("Figure", bound=FigureBase)  # Matplotlib Figure
# pstS = TypeVar("pstS", bound=Type[Series])
# pstDF = TypeVar("pstDF", bound=Type[DataFrame])
Tminmax = TypeVar("Tminmax", bound=Union[str, Timestamp])  # Tmin or Tmax
Model = TypeVar("Model", bound=Any)  # Model
TimeSeries = TypeVar("TimeSeries", bound=Any)  # Time Series
StressModel = TypeVar("StressModel", bound=Any)  # Stress Model
NoiseModel = TypeVar("NoiseModel", bound=Any)  # Noise Model
Solver = TypeVar("Solver", bound=Any)  # Base Solver
Recharge = TypeVar("Recharge", bound=Any)  # Recharge Base
Reservoir = TypeVar("Reservoir", bound=Any)  # Reservoir Base
CallBack = TypeVar("CallBack", bound=Any)  # Callback
Function = TypeVar("Function", bound=Any)  # Function (e.g. Objective Function)
RFunc = TypeVar("RFunc", bound=Any)  # rFunc Base
Array_Like = TypeVar("Array_Like", bound=ArrayLike)  # Array Like (NumPy based)
