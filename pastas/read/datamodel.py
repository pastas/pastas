from __future__ import print_function, division

import warnings

import numpy as np
import pandas as pd


class DataModel():
    def __init__(self, series=pd.Series(), x=None, y=None, latlon = None, metadata=None):
        """This is the standard datamodel class that is returned by all import
         methods.

        """
        self.series = series
        self.x = x
        self.y = y
        self.latlon = latlon
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

        if self.latlon is None and self.x is not None and self.y is not None:
            self.latlon = self.rd2wgs(self.x, self.y)

        if self.x is None and self.y is None and self.latlon is not None:
            self.x, self.y = self.wgs2rd(self.latlon)

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
            inProj = Proj(init='epsg:28992')
            outProj = Proj(init='epsg:4326')
            lon, lat = transform(inProj, outProj, x, y)
        except ImportError:
            warnings.warn('The module pyproj could not be imported. Please '
                          'install through:'
                          '>>> pip install requests'
                          'or ... conda install requests', ImportWarning)
            lat = np.NaN
            lon = np.NaN
        except:
            # Otherwise just return none's
            lat = np.NaN
            lon = np.NaN

        return (lat, lon)

    def wgs2rd(self, latlon):
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
            inProj = Proj(init='epsg:4326')
            outProj = Proj(init='epsg:28992')
            x, y = transform(inProj, outProj, latlon[1], latlon[0])
        except ImportError:
            warnings.warn('The module pyproj could not be imported. Please '
                          'install through:'
                          '>>> pip install requests'
                          'or ... conda install requests', ImportWarning)
            x = np.NaN
            y = np.NaN
        except:
            # Otherwise just return none's
            x = np.NaN
            y = np.NaN

        return (x, y)
