import numpy as np
import pandas as pd
from scipy.special import gammainc, gammaincinv

'''
rfunc module.
Contains classes for the response functions.
Each response function class needs the following:

Attributes
----------
nparam: integer
    number of parameters.
cutoff: float
    percentage after which the step function is cut off.

Functions
---------
set_parameters(self, name)
    A function that returs a Pandas DataFrame of the parameters of the
    response function. Columns of the dataframe need to be
    ['value', 'pmin', 'pmax', 'vary'].
    Rows of the DataFrame have names of the parameters.
    Input name is used as a prefix.
    This function is called by a Tseries object.
step(self, p)
    Returns an array of the step response. Input
    p is a numpy array of parameter values in the same order as
    defined in set_parameters.
block(self, p)
    Returns an array of the block response. Input
    p is a numpy array of parameter values in the same order as
    defined in set_parameters.

More information on how to write a response class can be found here:
https://github.com/gwtsa/gwtsa/wiki

gwtsa -2016
'''


class Gamma:
    '''
    step(t) = A * Gammainc(n, t / a)
    '''
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
    '''
    step(t) = A * (1 - exp(-t / a))
    '''
    
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
        s = p[0] * (1.0 - np.exp(-t / p[1]))
        return s

    def block(self, p):
        s = self.step(p)
        return s[1:] - s[:-1]
