# flake8: noqa
import logging

from pandas.plotting import register_matplotlib_converters

import pastas.objective_functions as objfunc
import pastas.plotting.plots as plots
import pastas.recharge as rch
import pastas.stats as stats
import pastas.timeseries_utils as ts
from pastas import extensions
from pastas.decorators import set_use_numba
from pastas.model import Model
from pastas.noisemodels import ArmaModel, NoiseModel
from pastas.plotting.modelcompare import CompareModels
from pastas.plotting.plots import TrackSolve
from pastas.rcparams import rcParams
from pastas.rfunc import (
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
from pastas.solver import EmceeSolve, LeastSquares, LmfitSolve
from pastas.stressmodels import (
    ChangeModel,
    Constant,
    LinearTrend,
    RechargeModel,
    StepModel,
    StressModel,
    TarsoModel,
    WellModel,
)
from pastas.timeseries import validate_oseries, validate_stress
from pastas.transform import ThresholdTransform
from pastas.utils import initialize_logger, set_log_level
from pastas.version import __version__, show_versions

logger = logging.getLogger(__name__)
initialize_logger(logger)

# Register matplotlib converters when using Pastas
# https://github.com/pastas/pastas/issues/92

register_matplotlib_converters()
