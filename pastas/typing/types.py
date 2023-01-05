# Type hinting for Pastas library
# Typing
from typing import Union, Any, TypeVar, TYPE_CHECKING

# External libraries
# Pandas
from pandas import Timestamp
# Matplotlib
from matplotlib.axes._base import _AxesBase
from matplotlib.figure import FigureBase
# Numpy
from numpy.typing import ArrayLike

# External Types
Axes = TypeVar("Axes", bound=_AxesBase)  # Matplotlib Axes
Figure = TypeVar("Figure", bound=FigureBase)  # Matplotlib Figure
Array_Like = TypeVar("Array_Like", bound=ArrayLike)  # Array Like (NumPy based)

# Internal library
if TYPE_CHECKING: # https://mypy.readthedocs.io/en/latest/runtime_troubles.html
    import pastas as ps

# Internal Types
Tminmax = TypeVar("Tminmax", bound=Union[str, Timestamp])  # Tmin or Tmax
Model = TypeVar("Model", bound="ps.Model")  # Model
TimeSeries = TypeVar("TimeSeries", bound="ps.TimeSeries")  # Time Series
StressModel = TypeVar("StressModel", bound="ps.stressmodels.StressModelBase")  # Stress Model
NoiseModel = TypeVar("NoiseModel", bound="ps.noisemodels.NoiseModelBase")  # Noise Model
Solver = TypeVar("Solver", bound="ps.solver.BaseSolver")  # Base Solver
Recharge = TypeVar("Recharge", bound="ps.recharge.RechargeBase")  # Recharge Base
Reservoir = TypeVar("Reservoir", bound="ps.reservoir.ReservoirBase")  # Reservoir Base
CallBack = TypeVar("CallBack", bound=Any)  # Callback
Function = TypeVar("Function", bound=Any)  # Function (e.g. Objective Function)
RFunc = TypeVar("RFunc", bound="ps.rfunc.RfuncBase")  # rFunc Base
