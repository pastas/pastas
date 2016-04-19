import numpy as np
import pandas as pd
from scipy.special import gammainc, gammaincinv

'''Need a description of the requirements of all the response functions'''
  
class Gamma:
    def __init__(self):
        self.nparam = 3
        self.cutoff = 0.99
    def set_parameters(self, name):
        parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
        parameters.loc[name + '_A'] = (1.0, 0.0, 5.0, 1)
        parameters.loc[name + '_n'] = (1.0, 0.0, 5.0, 1)
        parameters.loc[name + '_a'] = (100.0, 1.0, 5000.0, 1)
        return parameters
    def step(self, p):
        self.tmax = gammaincinv(p[1], self.cutoff) * p[2]
        t = np.arange(1.0, self.tmax)
        s = p[0] * gammainc(p[1], t / p[2])
        return s
    def block(self, p):
        s = self.step(p)
        return s[1:] - s[:-1]


class ExpDecay:
    def __init__(self):
        self.nparam = 2
        self.cutoff = 0.99
    def set_parameters(self, name):
        parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
        parameters.loc[name + '_A'] = (1.0, 0.0, 5.0, 1)
        parameters.loc[name + '_a'] = (100.0, 1.0, 5000.0, 1)
        return parameters
    def step(self, p):
        self.tmax = -np.log(1.0 / p[1]) * p[1]
        t = np.arange(1.0, self.tmax)
        s = -p[0] * np.exp(- t / p[1]) + p[0]
        return s
    def block(self, p):
        s = self.step(p)
        return s[1:] - s[:-1]