""" This file contains the recharge models


"""

from recharge import pref, perc, comb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Recharge:
    def __init__(self):
        pass

    def plot(self, p=None):
        recharge = self.simulate(p)
        plt.bar(recharge)


class Preferential(Recharge):
    def __init__(self):
        Recharge.__init__(self)
        self.nparam = 3
        self.dt = 1
        self.solver = 1

    def set_parameters(self, name):
        parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
        parameters.loc[name + '_Srmax'] = (0.26, np.nan, np.nan, 0)
        parameters.loc[name + '_Beta'] = (3.0, 0.0, np.nan, 1)
        parameters.loc[name + '_Imax'] = (1.5e-3, np.nan, np.nan, 0)
        return parameters

    def simulate(self, precipitation, evaporation, p=None):
        t = np.arange(len(precipitation))
        recharge = pref(t, precipitation, evaporation, p[0], p[1], p[2],
                        self.dt, self.solver)[0]
        return recharge

