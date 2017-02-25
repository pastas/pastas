from __future__ import print_function, division

def objfunc_residuals(model, parameters, tmin, tmax, noise, freq):
    p = np.array([p.value for p in parameters.values()])
    if noise:
        return model.innovations(p, tmin, tmax, freq, model.oseries_calib)
    else:
        return model.residuals(p, tmin, tmax, freq, model.oseries_calib)
