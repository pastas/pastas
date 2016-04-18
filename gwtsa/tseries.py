import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

'''Need a description of the requirements of all the timeseries'''

class Tseries:
    '''Time series model consisting of the convolution of one stress with one response function'''
    def __init__(self, stress, rfunc, name):
        self.stress = stress
        self.rfunc = rfunc
        self.npoints = len(self.stress)
        self.nparam = rfunc.nparam
        self.name = name
        self.parameters = self.rfunc.set_parameters(self.name)
    def simulate(self, tindex=None, p=None):
        b = self.rfunc.block(p)
        h = pd.Series(fftconvolve(self.stress, b, 'full')[:self.npoints],
                      index=self.stress.index)
        if tindex is not None:
            h = h[tindex]
        return h
    
class Tseries2:
    '''Time series model consisting of the convolution of two stresses with one response function'''
    def __init__(self, stress1, stress2, rfunc, name):
        self.stress1 = stress1
        self.stress2 = stress2
        self.rfunc = rfunc
        self.nparam = self.rfunc.nparam + 1
        self.name = name
        self.parameters = self.rfunc.set_parameters(self.name)
        self.parameters.loc[self.name + '_f'] = (-1.0, -5.0, 0.0, 1)
    def simulate(self, tindex=None, p=None):
        b = self.rfunc.block(p[:self.nparam-1]) # nparam-1 depending on rfunc
        stress = self.stress1 + p[self.nparam-1] * self.stress2
        stress.fillna(stress.mean(), inplace=True)
        self.npoints = len(stress)
        h = pd.Series(fftconvolve(stress, b, 'full')[:self.npoints],
                      index=stress.index)
        if tindex is not None:
            h = h[tindex]
        return h
    
class Constant:
    '''A constant value'''
    def __init__(self, value=0.0):
        self.nparam = 1
        self.parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
        self.parameters.loc['constant_d'] = (value, np.nan, np.nan, 1)
    def simulate(self, tindex=None, p=None):
        return p
    
class NoiseModel:
    def __init__(self):
        self.nparam = 1
        self.parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
        self.parameters.loc['noise_alpha'] = (14.0, 0, 5000, 1)
    def simulate(self, res, delt, tindex=None, p=None):
        innovations = pd.Series(res, index=res.index)
        innovations[1:] -= np.exp(-delt[1:] / p) * res.values[:-1]  # values is needed else it gets messed up with the dates
        return innovations