import numpy as np
import pandas as pd
from scipy.special import gammainc, gammaincinv

'''
rfunc.py contains the response functions.

Each response function is stored as a class and should at least contain the 
following:

Attributes
----------
nparam: integer
    number of parameters.
cutoff: float
    percentage after which the response is cutoff.

Functions
---------
__init__(self)
    function that initialized an instance of the response function.
set_parameters(self, name)
    This function is called by an tseries object. Name is used as a prefix
    for the parameter names. A Pandas dataframe is returned that contains
    the values, minimum, maximum, and vary for each parameter.
step(self, p)
    This function calculates and returns an array of the step response. Input
    p is an array of parameters in a specific order.
block(self, p)
    This function calls the step function and returns and array of the block
    response. Input p is an array of parameters in a specific order.

More information on how to write a response class can be found here:
https://github.com/gwtsa/gwtsa/wiki

gwtsa -2016
'''


class Gamma:
    def __init__(self):
        self.nparam = 3
        self.cutoff = 0.99
    def set_parameters(self, name):
        parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
        parameters.loc[name + '_A'] = (500.0, 0.0, 5000.0, 1)
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


class Exponential:
    def __init__(self):
        self.nparam = 2
        self.cutoff = 0.99
    def set_parameters(self, name):
        parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
        parameters.loc[name + '_A'] = (500.0, 0.0, 5000.0, 1)
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