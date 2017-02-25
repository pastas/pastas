from __future__ import print_function, division

import numpy as np

def residuals(parameters, model, tmin=None, tmax=None, noise=True, freq='D'):
    p = np.array([p.value for p in parameters.values()])
    if noise:
        return model.innovations(p, tmin, tmax, freq, model.oseries_calib)
    else:
        return model.residuals(p, tmin, tmax, freq, model.oseries_calib)
