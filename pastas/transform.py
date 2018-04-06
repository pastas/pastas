"""The stressmodels module contains all the transforms that can be added to the simulation of a model.

"""
import numpy as np
import pandas as pd
from .model import Model


class ThresholdTransform:
    _name = "ThresholdTransform"
    def __init__(self, value=0.0, vmin=np.nan, vmax = np.nan, name='ThresholdTransform', nparam=2):
        if isinstance(value, Model):
            # determine the initial parameter from the model
            ml = value
            value = ml.oseries.min() + 0.75*(ml.oseries.max()-ml.oseries.min())
            if np.isnan(vmin):
                vmin = ml.oseries.min() + 0.5*(ml.oseries.max()-ml.oseries.min())
            if np.isnan(vmax):
                vmax = ml.oseries.max()
        self.value = value
        self.vmin = vmin
        self.vmax = vmax
        self.name = name
        self.nparam = nparam
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.parameters.loc[self.name + '_1'] = (self.value, self.vmin, self.vmax, 1, self.name)
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
        data["vmin"] = self.vmin
        data["vmax"] = self.vmax
        data["name"] = self.name
        data['nparam'] = self.nparam
        return data
