from __future__ import print_function, division

import numpy as np

def residuals(parameters, model, tmin=None, tmax=None, noise=True, freq='D'):
    if noise:
        return model.innovations(parameters, tmin, tmax, freq,
            model.oseries_calib)
    else:
        return model.residuals(parameters, tmin, tmax, freq,
            model.oseries_calib)
