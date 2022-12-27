import logging

from pandas.plotting import register_matplotlib_converters

import pastas.plots as plots
import pastas.recharge as rch
import pastas.stats as stats

from pastas.model import Model
from pastas.modelcompare import CompareModels
from pastas.noisemodels import ArmaModel, NoiseModel
from pastas.plots import TrackSolve
from pastas.rcparams import rcParams
from pastas.read import (read_dino, read_dino_level_gauge, read_knmi, read_meny,
                         read_waterbase)
from pastas.rfunc import (DoubleExponential, Exponential, FourParam, Gamma, Hantush,
                          HantushWellModel, Kraijenhoff, One, Polder, Spline)
from pastas.solver import LeastSquares, LmfitSolve
from pastas.stressmodels import (ChangeModel, Constant, LinearTrend, RechargeModel,
                                 ReservoirModel, StepModel, StressModel,
                                 TarsoModel, WellModel)
from pastas.timeseries import TimeSeries
from pastas.transform import ThresholdTransform
from pastas.utils import initialize_logger, set_log_level, show_versions
from pastas.version import __version__

logger = logging.getLogger(__name__)
initialize_logger(logger)

# Register matplotlib converters when using Pastas
# https://github.com/pastas/pastas/issues/92

register_matplotlib_converters()
