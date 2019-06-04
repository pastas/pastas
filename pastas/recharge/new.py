
class Preferential:
    """
    Preferential flow recharge model

    The water balance for the root zone is calculated as:
    dS / dt = Pe * (1 - (Sr / Srmax)**Beta)- Epu * min(1, Sr / (0.5 * Srmax))

    """

    def __init__(self):
        self.nparam = 3
        self.dt = 1  # Has to be 1 right now.
        self.solver = 1  # 1 = implicit, 2 = explicit

    def set_parameters(self, name):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_Srmax'] = (0.26, np.nan, np.nan, 0, name)
        parameters.loc[name + '_Beta'] = (3.0, 0.0, np.nan, 1, name)
        parameters.loc[name + '_Imax'] = (1.5e-3, np.nan, np.nan, 0, name)
        return parameters

    def simulate(self, precip, evap, p=None):
        t = np.arange(len(precip))
        recharge = pref(t, precip, evap, p[0], p[1], p[2],
                        self.dt, self.solver)[0]
        return recharge


class Percolation:
    """
    Percolation flow recharge model

    Other water balance for the root zone s calculated as:

    dS/dt = Pe - Kp * (Sr / Srmax)**Gamma - Epu * min(1, Sr / (0.5 * Srmax))

    """

    def __init__(self):
        self.nparam = 4
        self.dt = 1
        self.solver = 1

    def set_parameters(self, name):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_Srmax'] = (0.26, np.nan, np.nan, 0, name)
        parameters.loc[name + '_Kp'] = (1.0e-2, 0.0, np.nan, 1, name)
        parameters.loc[name + '_Gamma'] = (3.0, 0.0, np.nan, 1, name)
        parameters.loc[name + '_Imax'] = (1.5e-3, 0.0, np.nan, 0, name)
        return parameters

    def simulate(self, precip, evap, p=None):
        t = np.arange(len(precip))
        recharge = perc(t, precip, evap, p[0], p[1], p[2], p[3],
                        self.dt, self.solver)[0]
        return recharge


class Combination:
    """
    Combination flow recharge model

    Other water balance for the root zone is calculated as:

    dS/ dt = Pe[t] * (1 - (Sr[t] / Srmax)**Beta) - Kp * (Sr / Srmax)**Gamma -
    Epu * min(1, Sr/ (0.5 * Srmax))

    """

    def __init__(self):
        self.nparam = 5
        self.dt = 1
        self.solver = 1

    def set_parameters(self, name):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_Srmax'] = (0.26, np.nan, np.nan, 0, name)
        parameters.loc[name + '_Kp'] = (1.0e-2, 0.0, np.nan, 1, name)
        parameters.loc[name + '_Beta'] = (3.0, 0.0, np.nan, 1, name)
        parameters.loc[name + '_Gamma'] = (3.0, 0.0, np.nan, 1, name)
        parameters.loc[name + '_Imax'] = (1.5e-3, 0.0, np.nan, 0, name)
        return parameters

    def simulate(self, precip, evap, p=None):
        t = np.arange(len(precip))
        Rs, Rf = comb(t, precip, evap, p[0], p[1], p[2], p[3],
                      p[4], self.dt, self.solver)[0:2]
        recharge = Rs + Rf  # Slow plus fast recharge
        return recharge
