# Type hinting for Pastas library
# Typing
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar

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
TimestampType = TypeVar("TimestampType", bound=str | Timestamp)  # Tmin or Tmax
Model = TypeVar("Model", bound="ps.Model")  # Model
TimeSeries = TypeVar("TimeSeries", bound="ps.timeseries.TimeSeries")  # Time Series
StressModel = TypeVar(
    "StressModel", bound="ps.stressmodels.StressModelBase"
)  # Stress Model
NoiseModel = TypeVar("NoiseModel", bound="ps.noisemodels.NoiseModelBase")  # Noise Model
Solver = TypeVar("Solver", bound="ps.solver.BaseSolver")  # Base Solver
Recharge = TypeVar("Recharge", bound="ps.recharge.RechargeBase")  # Recharge Base
CallBack = TypeVar("CallBack", bound=Any)  # Callback
RFunc = TypeVar("RFunc", bound="ps.rfunc.RfuncBase")  # rFunc Base


class OseriesSettingsDict(TypedDict):
    """
    Time series settings dictionary defining logic for filling and downsampling time series.

    Parameters
    ----------
    sample_down : {"drop", "mean", "sum", "min", "max"}
      Method for down-sampling time series (decreasing frequency, e.g. daily to weekly).
      - "drop": Drop NaNs from time series.
      - "mean": Resample by taking the mean.
      - "sum": Resample by summing values.
      - "max": Resample with maximum value.
      - "min": Resample with minimum value.
    fill_nan : {"drop", "mean", "interpolate"} or float
      Method for filling NaNs.
      - "drop": Drop NaNs from time series.
      - "mean": Fill NaNs with mean value of time series.
      - "interpolate": Fill NaNs by interpolating between finite values.
      - float: Fill NaN with provided value, e.g. 0.0.
    """

    sample_down: Literal["mean", "drop", "sum", "min", "max"]
    fill_nan: Literal["drop", "mean", "interpolate"] | float


class StressSettingsDict(TypedDict):
    """
    Stress time series settings dictionary defining logic for filling and up- or
    downsampling time series.

    Parameters
    ----------
    sample_up : {"mean", "interpolate", "divide", "bfill", "ffill"}
      Method for up-sampling time series (increasing frequency, e.g. weekly to daily).
      - "mean": Fill up-sampled time steps with mean of timeseries.
      - "interpolate": Fill up-sampled time steps by interpolating between current values.
      - "divide": Fill up-sampled steps with current value divided by length of current time steps.
      - "bfill": Back-fill up-sampled time steps with current values.
      - "ffill": Forward-fill up-sampled time steps with current values.
    sample_down : {"mean", "drop", "sum", "min", "max"}
      Method for down-sampling time series (decreasing frequency, e.g. daily to weekly).
      - "mean": Resample time series by taking the mean.
      - "drop": Resample by taking the mean, dropping any NaN-values.
      - "sum": Resample by summing values.
      - "max": Resample with maximum value.
      - "min": Resample with minimum value.
    fill_nan : {"drop", "mean", "interpolate"} or float
      Method for filling NaNs.
      - "drop": Drop NaNs from time series.
      - "mean": Fill NaNs with mean value of time series.
      - "interpolate": Fill NaNs by interpolating between finite values.
      - float: Fill NaN with provided value, e.g. 0.0.
    fill_before : {"mean", "bfill"} or float
      Method for extending time series into the past.
      - "mean": Extend into past with mean value of time series.
      - "bfill": Back-fill into past with first value.
      - float: Extend into past with provided value, e.g. 0.0.
    fill_after : {"mean", "ffill"} or float
      Method for extending time series into the future.
      - "mean": Extend into future with mean value of time series.
      - "ffill": Forward-fill into future with last value.
      - float: Extend into future with provided value, e.g. 0.0.
    """

    sample_up: Literal["mean", "interpolate", "divide", "bfill", "ffill"]
    sample_down: Literal["mean", "drop", "sum", "min", "max"]
    fill_nan: Literal["drop", "mean", "interpolate"] | float
    fill_before: Literal["mean", "bfill"] | float
    fill_after: Literal["mean", "ffill"] | float
