"""Contains the classes for the different models that are available to
calculate the recharge from precipitation and evaporation data.

.. codeauthor:: R.A. Collenteur, University of Graz

Supported Recharge models
-------------------------
The following recharge models are currently supported and tested:

.. autosummary::
    :nosignatures:
    :toctree: ./generated

    Linear

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
