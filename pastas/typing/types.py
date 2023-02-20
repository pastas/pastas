# flake8: noqa
# Type hinting for Pastas library
# Typing
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union

# External libraries
# Matplotlib
from matplotlib.axes import Axes as MatplotlibAxes
from matplotlib.figure import Figure as MatplotlibFigure

# Numpy
from numpy.typing import ArrayLike as NumpyArrayLike

# Pandas
from pandas import Timestamp

# External Types
Axes = TypeVar("Axes", bound=MatplotlibAxes)  # Matplotlib Axes
Figure = TypeVar("Figure", bound=MatplotlibFigure)  # Matplotlib Figure
ArrayLike = TypeVar("ArrayLike", bound=NumpyArrayLike)  # Array Like (NumPy based)

# Internal library
if TYPE_CHECKING:  # https://mypy.readthedocs.io/en/latest/runtime_troubles.html
    import pastas as ps

# Internal Types
TimestampType = TypeVar("TimestampType", bound=Union[str, Timestamp])  # Tmin or Tmax
Model = TypeVar("Model", bound="ps.Model")  # Model
TimeSeries = TypeVar("TimeSeries", bound="ps.timeseries.TimeSeries")  # Time Series
StressModel = TypeVar(
    "StressModel", bound="ps.stressmodels.StressModelBase"
)  # Stress Model
NoiseModel = TypeVar("NoiseModel", bound="ps.noisemodels.NoiseModelBase")  # Noise Model
Solver = TypeVar("Solver", bound="ps.solver.BaseSolver")  # Base Solver
Recharge = TypeVar("Recharge", bound="ps.recharge.RechargeBase")  # Recharge Base
CallBack = TypeVar("CallBack", bound=Any)  # Callback
Function = Callable[..., Any]  # Function (e.g. Objective Function)
RFunc = TypeVar("RFunc", bound="ps.rfunc.RfuncBase")  # rFunc Base
