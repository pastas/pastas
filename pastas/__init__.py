# flake8: noqa
import logging

from pandas.plotting import register_matplotlib_converters

import pastas.plots as plots
import pastas.recharge as rch
import pastas.stats as stats
import pastas.timeseries_utils as ts

from .decorators import set_use_numba
from .model import Model
from .modelcompare import CompareModels
from .noisemodels import ArmaModel, NoiseModel
from .plots import TrackSolve
from .rcparams import rcParams
from .rfunc import (
    DoubleExponential,
    Exponential,
    FourParam,
    Gamma,
    Hantush,
    HantushWellModel,
    Kraijenhoff,
    One,
    Polder,
    Spline,
)
from .solver import LeastSquares, LmfitSolve
from .stressmodels import (
    ChangeModel,
    Constant,
    LinearTrend,
    RechargeModel,
    StepModel,
    StressModel,
    TarsoModel,
    WellModel,
)
from .timeseries import validate_oseries, validate_stress
from .transform import ThresholdTransform
from .utils import initialize_logger, set_log_level
from .version import __version__, show_versions

logger = logging.getLogger(__name__)
initialize_logger(logger)

# Register matplotlib converters when using Pastas
# https://github.com/pastas/pastas/issues/92

register_matplotlib_converters()
