__name__='gwtsa'
__author__='gwtsa team'
__docformat__ = 'NumPy'
from version import __version__

from model import Model
from rfunc import Gamma, Exponential
from tseries import Tseries, Tseries2, Constant, NoiseModel
from stats import Statistics