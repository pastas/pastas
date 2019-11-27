"""recharge_func module

Author: R.A. Collenteur, University of Graz

Contains the classes for the different models that are available to calculate
the recharge from precipitation and evaporation data.

Each Recharge class contains at least the following:

Attributes
----------
nparam: int
    Number of parameters needed for this model.

Functions
---------
get_init_parameters(self, name)
    A function that returns a Pandas DataFrame of the parameters of the
    recharge function. Columns of the dataframe need to be ['value', 'pmin',
    'pmax', 'vary']. Rows of the DataFrame have names of the parameters. Input
    name is used as a prefix. This function is called by a stressmodel object.
simulate(self, evap, prec, p=None)
    A function that returns an array of the simulated recharge series.

"""

import numpy as np
import pandas as pd


class RechargeBase:
    """Base class for classes that calculate the recharge.

    """

    def __init__(self):
        self.temp = False
        self.nparam = 0

    @staticmethod
    def get_init_parameters(name="recharge"):
        """

        Parameters
        ----------
        name: str, optional
            String with the name that is used as prefix for the parameters.

        Returns
        -------
        parameters: pandas.DataFrame
            Pandas DataFrame with the parameters.

        """
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        return parameters

    def simulate(self, prec, evap, p, temp=None):
        pass


class Linear(RechargeBase):
    """Linear recharge model.

    The recharge to the groundwater is calculated as:
    R = P - f * E

    """
    _name = "Linear"

    def __init__(self):
        RechargeBase.__init__(self)
        self.nparam = 1

    def get_init_parameters(self, name="recharge"):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_f'] = (-1.0, -2.0, 0.0, True, name)
        return parameters

    def simulate(self, prec, evap, p, **kwargs):
        """

        Parameters
        ----------
        prec, evap: array_like
            array with the precipitation and evaporation values. These
            arrays must be of the same length and at the same time steps.
        p: float
            parameter value used in recharge calculation.

        Returns
        -------
        recharge: array_like
            array with the recharge series.

        """
        return np.add(prec, np.multiply(evap, p))
