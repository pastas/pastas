__name__ = 'pasta'
__author__ = 'pasta team'
__docformat__ = 'NumPy'

from model import Model
from rfunc import Gamma, Exponential, Hantush, Theis
from tseries import Tseries, Tseries2, Recharge, Well, Constant, NoiseModel
from stats import Statistics
from read.read_series import ReadSeries
from recharge.recharge_func import Preferential, Linear, Percolation, Combination
from solver import LmfitSolve, DESolve
