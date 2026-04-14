"""Type definitions and aliases for Pastas internals."""

# Type hinting for Pastas library
# Typing
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar

# External libraries
from matplotlib.axes import Axes as MatplotlibAxes
from matplotlib.figure import Figure as MatplotlibFigure
from numpy.typing import ArrayLike as NumpyArrayLike
from pandas import Timedelta, Timestamp

# External Types
Axes = TypeVar("Axes", bound=MatplotlibAxes)  # Matplotlib Axes
Figure = TypeVar("Figure", bound=MatplotlibFigure)  # Matplotlib Figure
ArrayLike = TypeVar("ArrayLike", bound=NumpyArrayLike)  # Array Like (NumPy based)

# Internal library
if TYPE_CHECKING:  # https://mypy.readthedocs.io/en/latest/runtime_troubles.html
    import pastas as ps

# Internal Types
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


class ModelSettingsDict(TypedDict):
    """Model settings dictionary defining logic for handling time series and model fitting.

    Parameters
    ----------
    tmin: pandas.Timestamp
        A pandas.Timestamp with the start date for the simulation period
        (E.g. '1980-01-01 00:00:00'). If none is provided, the tmin from
        the oseries is used.
    tmax: pandas.Timestamp
        A pandas.Timestamp with the end date for the simulation period
        (E.g. '2020-01-01 00:00:00'). If none is provided, the tmax from
        the oseries is used.
    freq: str
        String with the frequency the stressmodels are simulated. Must be one of
        the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
    warmup: Timedelta
        Warmup period (in days) for which the simulation is calculated, but not
        used for the calibration period.
    fit_constant: bool, optional
        Argument that determines if the constant is fitted as a parameter. If it
        is set to False, the constant is set equal to the mean of the residuals.
    freq_obs: str, optional
        String with the frequency of the observations that the model will be
        calibrated on. Must be one of the following (D, h, m, s, ms, us, ns) or a
        multiple of that e.g. "7D". Should generally be larger than the frequency
        of the original observations and the model frequency (freq). If freq_obs
        is None, the frequency of the model (freq) will be used.

    """

    tmin: Timestamp
    tmax: Timestamp
    freq: str
    warmup: Timedelta
    solver: (
        Literal["LeastSquares", "LmfitSolve", "EmceeSolve"] | None
    )  # TODO: check if still needed
    fit_constant: bool
    freq_obs: str | None
    noise: bool  # TODO: remove as deleted in PR #1122
