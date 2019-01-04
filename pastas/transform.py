"""The stressmodels module contains all the transforms that can be added to the
 simulation of a model. These transforms are applied after the simulation,
 to incorporate nonlineair effects.

"""
import numpy as np
from pandas import DataFrame


class ThresholdTransform:
    """ThresholdTransform lowers the simulation when it exceeds a certain value

    In geohydrology this transform can for example be used in a situation where
    the groundwater level reaches the surface level and forms a lake. Beacuase
    of the larger storage of the lake, the (groundwater) level then rises
    slower when it rains.


    Parameters
    ----------
    value : float
        The starting value above which the simulation is lowered
    vmin : float
        The minimum value above which the simulation is lowered
    vmin : float
        The maximum value above which the simulation is lowered
    name: str
        Name of the transform
    nparam : int
        The number of parameters. Default is nparam=2. The first parameter
        then is the threshold, and the second parameter is the factor with
        which the simulation is lowered.

    """
    _name = "ThresholdTransform"

    def __init__(self, value=np.nan, vmin=np.nan, vmax=np.nan,
                 name='ThresholdTransform', nparam=2):
        self.value = value
        self.vmin = vmin
        self.vmax = vmax
        self.name = name
        self.nparam = nparam

    def set_model(self, ml):
        obs = ml.observations()
        if np.isnan(self.value):
            self.value = obs.min() + 0.75 * (obs.max() - obs.min())
        if np.isnan(self.vmin):
            self.vmin = obs.min() + 0.5 * (obs.max() - obs.min())
        if np.isnan(self.vmax):
            self.vmax = obs.max()
        self.set_init_parameters()

    def set_init_parameters(self):
        self.parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        self.parameters.loc[self.name + '_1'] = (
            self.value, self.vmin, self.vmax, 1, self.name)
        if self.nparam == 2:
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
