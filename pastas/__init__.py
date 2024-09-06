# ruff: noqa: D104 F401
import logging
import warnings

from pandas.plotting import register_matplotlib_converters

import pastas.objective_functions as objfunc
import pastas.plotting.plots as plots
import pastas.recharge as rch
import pastas.stats as stats
import pastas.timeseries_utils as ts
from pastas import extensions
from pastas.dataset import list_datasets, load_dataset
from pastas.decorators import set_use_numba
from pastas.model import Model
from pastas.noisemodels import ArmaModel, ArmaNoiseModel, ArNoiseModel, NoiseModel
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
from pastas.utils import set_log_level
from pastas.version import __version__, show_versions


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg)


warnings.formatwarning = custom_formatwarning
warnings.warn(
    """DeprecationWarning: As of Pastas 1.5, no noisemodel is added to the pastas Model class by default anymore. To solve your model using a noisemodel, you have to explicitly add a noisemodel to your model before solving. For more information, and how to adapt your code, please see this issue on GitHub: https://github.com/pastas/pastas/issues/735""",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

# Register matplotlib converters when using Pastas
# https://github.com/pastas/pastas/issues/92

register_matplotlib_converters()
