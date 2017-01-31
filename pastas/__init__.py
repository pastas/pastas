from __future__ import print_function

# Get the version.
version = {}
with open("pastas/version.py") as fp:
    exec(fp.read(), version)
__version__ = version['__version__']
del version

from .model import Model
from .read.read_series import ReadSeries
from .recharge.recharge_func import Preferential, Linear, Percolation, \
    Combination
from .rfunc import Gamma, Exponential, Hantush, Theis
from .solver import LmfitSolve, DESolve
from .stats import Statistics
from .tseries import Tseries, Tseries2, Recharge, Well, Constant, NoiseModel
