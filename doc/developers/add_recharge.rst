Adding recharge classes
=======================
the recharge class is developed specifically for the simulation of groundwater,
with different options to calculate the groundwater recharge. The base of the
recharge models is a `Recharge` class in tseries.py. This class takes two
stresses: (potential) evaporation and precipitation, and uses a recharge function
to calculate a recharge stress from these two input stresses. Finally, the
calculated recharge series is convoluted with a response function to calculate
the contribution to the groundwater heads. the parameters for both the response
function and recharge model are optimized simultaneously this way.

In this section it is described how to add a recharge class to calculate the
recharge. The actual functions to calculate the recharge are contained in the
`recharge` folder. Within this folder, two files are important to take a look at:
`recharge_func.py` and `recharge.py`. The first contains a class for each
recharge model (similar to the classes in rfunc.py) and the second contains the
computational core to calculate recharge series. The other files are solutions
to reduce computation times but we'll come back to those.

* Please take a look at the Recharge example notebook!*

The recharge_func.py classes
============================
The classes defined in this file are the classes that are provided when a
Recharge tseries object is created. E.g.:

>>> ts = Recharge(precip, evap, Gamma(), Preferential())

In this section we will have a closer look at the Preferential() class. 

::

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
            parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
            parameters.loc[name + '_Srmax'] = (0.26, np.nan, np.nan, 0)
            parameters.loc[name + '_Beta'] = (3.0, 0.0, np.nan, 1)
            parameters.loc[name + '_Imax'] = (1.5e-3, np.nan, np.nan, 0)
            return parameters

        def simulate(self, precip, evap, p=None):
            t = np.arange(len(precip))
            recharge = pref(t, precip, evap, p[0], p[1], p[2],
                            self.dt, self.solver)[0]
            return recharge