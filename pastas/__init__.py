import logging

import pastas.recharge as rch
from .model import Model
from .noisemodels import NoiseModel, ArmaModel
from .plots import TrackSolve
from .read import read_meny, read_dino, read_dino_level_gauge, read_knmi, \
    read_waterbase
from .rfunc import Gamma, Exponential, Hantush, Polder, One, FourParam, \
    DoubleExponential, HantushWellModel
from .solver import LmfitSolve, LeastSquares, LmfitSolveNew
from .stressmodels import StressModel, StressModel2, Constant, FactorModel, \
    RechargeModel, WellModel, StepModel, LinearTrend, TarsoModel
from .timeseries import TimeSeries
from .transform import ThresholdTransform
from .utils import initialize_logger, set_log_level, show_versions
from .version import __version__
from .rcparams import rcParams

logger = logging.getLogger(__name__)
initialize_logger(logger)

# Register matplotlib converters when using Pastas
# https://github.com/pastas/pastas/issues/92
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


class Project:
    def __init__(self):
        msg = "The Pastas Project class has been removed. Please use the " \
              "Pastastore (https://github.com/pastas/pastastore). "
        logger.error(msg)
