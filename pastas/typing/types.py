# flake8: noqa
# Type hinting for Pastas library
# Typing
from typing import TYPE_CHECKING, Any, Callable, TypedDict, TypeVar, Union

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


class OseriesSettingsDict(TypedDict):
    """
    Time series settings is a dictionary defining logic for filling and up- or
    downsampling time series.

    Time series settings
    --------------------
    fill_nan : {"drop", "mean", "interpolate"} or float
        Method for filling NaNs.
           * `drop`: drop NaNs from time series
           * `mean`: fill NaNs with mean value of time series
           * `interpolate`: fill NaNs by interpolating between finite values
           * `float`: fill NaN with provided value, e.g. 0.0
    sample_down : {"drop", "mean", "sum", "min", "max"}
        Method for down-sampling time series (decreasing frequency, e.g. going from
        daily to weekly values).
           * `drop`: resample the time series by taking the mean, dropping any NaN-values
           * `mean`: resample time series by taking the mean
           * `sum`: resample time series by summing values
           * `max`: resample time series with maximum value
           * `min`: resample time series with minimum value
    """

    sample_down: str
    fill_nan: str


class StressSettingsDict(TypedDict):
    """
    Time series settings is a dictionary defining logic for filling and up- or
    downsampling time series.

    Time series settings
    --------------------
    fill_nan : {"drop", "mean", "interpolate"} or float
        Method for filling NaNs.
           * `drop`: drop NaNs from time series
           * `mean`: fill NaNs with mean value of time series
           * `interpolate`: fill NaNs by interpolating between finite values
           * `float`: fill NaN with provided value, e.g. 0.0
    fill_before : {"mean", "bfill"} or float
        Method for extending time series into past.
           * `mean`: extend time series into past with mean value of time series
           * `bfill`: extend time series into past by back-filling first value
           * `float`: extend time series into past with provided value, e.g. 0.0
    fill_after : {"mean", "ffill"} or float
        Method for extending time series into future.
           * `mean`: extend time series into future with mean value of time series
           * `ffill`: extend time series into future by forward-filling last value
           * `float`: extend time series into future with provided value, e.g. 0.0
    sample_up : {"mean", "interpolate", "divide"} or float
        Method for up-sampling time series (increasing frequency, e.g. going from weekly
        to daily values).
           * `bfill` or `backfill`: fill up-sampled time steps by back-filling current
             values
           * `ffill` or `pad`: fill up-sampled time steps by forward-filling current
             values
           * `mean`: fill up-sampled time steps with mean of timeseries
           * `interpolate`: fill up-sampled time steps by interpolating between current
             values
           * `divide`: fill up-sampled steps with current value divided by length of
             current time steps (i.e. spread value over new time steps).
    sample_down : {"mean", "drop", "sum", "min", "max"}
        Method for down-sampling time series (decreasing frequency, e.g. going from
        daily to weekly values).
           * `mean`: resample time series by taking the mean
           * `drop`: resample the time series by taking the mean, dropping any
             NaN-values
           * `sum`: resample time series by summing values
           * `max`: resample time series with maximum value
           * `min`: resample time series with minimum value
    """

    sample_up: str
    sample_down: str
    fill_nan: Union[str, float]
    fill_before: Union[str, float]
    fill_after: Union[str, float]
