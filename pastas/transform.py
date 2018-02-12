"""The stressmodels module contains all the transforms that can be added to the simulation of a model.

"""
import pandas as pd
import numpy as np

class NonLinTransform():
    def __init__(self, parameters, pmin=np.nan, pmax=np.nan):
        self.nparam = len(parameters)
        self.pmin = pmin
        self.pmax = pmax
        self.name = "transform"
        self.set_init_parameters(parameters)

    def set_init_parameters(self, parameters):
        self.parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        for i in range(self.nparam):
            self.parameters.loc[self.name + str(i+1)] = (parameters[i], self.pmin, self.pmax, 1, self.name)

    def simulate(self, h, p):
        if self.nparam == 1:
            # value above a threshold p[0] are equal to the threshold
            h[h > p[0]] = p[0]
        elif self.nparam == 2:
            # values above a threshold p[0] are scaled by p[1]
            mask = h > p[0]
            h[mask] = p[0] + p[1] * (h[mask] - p[0])
        else:
            raise ValueError('Not yet implemented yet')
        return h
