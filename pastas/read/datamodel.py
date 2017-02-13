import warnings
import numpy as np
import pandas as pd


class DataModel():
    def __init__(self):
        """This is the standard datamodel class that is returned by all import
         methods.

        """
        self.series = pd.Series()
        self.x = None
        self.y = None
        self.metadata = {}

    def rd2wgs(self, x, y):
        """

        Parameters
        ----------
        xy: float
            float of the location ???

        Returns
        -------
        XY location in lat-lon format

        """
        try:
            from pyproj import Proj, transform
            outProj = Proj(init='epsg:4326')
            inProj = Proj(init='epsg:28992')
            lon, lat = transform(inProj, outProj, x, y)
        except ImportError:
            warnings.warn('The module pyproj could not be imported. Please '
                          'install through:'
                          '>>> pip install requests'
                          'or ... conda install requests', ImportWarning)
            lat = np.NaN
            lon = np.NaN

        return (lat, lon)
