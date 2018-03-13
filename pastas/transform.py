"""The stressmodels module contains all the transforms that can be added to the simulation of a model.

"""
import numpy as np
import pandas as pd
from .model import Model


class NonLinTransform:
    _name = "NonLinTransform"
    def __init__(self, value=0.0, name='NonLinTransform', nparam=2):
        if isinstance(value, Model):
            # determine the initial parameter from the model
            ml = value
            value = ml.oseries.mean()
        self.value = value
        self.name = name
        self.nparam = nparam
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.parameters.loc[self.name + '_1'] = (self.value, np.nan, np.nan, 1, self.name)
        if self.nparam==2:
            self.parameters.loc[self.name + '_2'] = (0.5, 0., 1., 1, self.name)

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

    def dump(self):
        data = dict()
        data["transform"] = self._name
        data["value"] = self.value
        data["name"] = self.name
        data['nparam'] = self.nparam
        return data
