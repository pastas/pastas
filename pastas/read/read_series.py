"""

@author: ruben
"""

from pastas.read.dinodata import DinoGrondwaterstand
from pastas.read.knmidata import KnmiStation
import numpy as np


class ReadSeries:
    def __init__(self, fname, filetype, variable=None):
        """
        Read in time series.

        Parameters
        ----------
        fname: str
            filename inlcuding the relative path to the file
        filetype: str
            Type of the file. dedicated import modules have to be written and
            imported for it.
        variable: str
            string with the variable name to import

        """

        if filetype == 'dino':
            dino = DinoGrondwaterstand(fname)
            self.series = dino.stand
            self.xy = (dino.x, dino.y)
            self.latlon = self.rd2wgs(self.xy)
            self.meta = dino.meta[-1]
        elif filetype == 'knmi':
            knmi = KnmiStation.fromfile(fname)
            self.series = knmi.data[variable]
            if knmi.stations is not None:
                self.latlon = (knmi.stations['LAT_north'][0],
                               knmi.stations['LON_east'][0])
                self.meta = knmi.stations
            else:
                self.latlon = (np.NaN, np.NaN)
                self.meta = {}

        elif filetype == 'usgs':
            # not implemented yet
            pass
        elif filetype == 'csv':
            # not implemented yet
            pass
        else:
            raise Exception('Unknown filetype')

    def rd2wgs(self, xy):
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
        except ImportError:
            raise ImportWarning(
                'The module pyproj could not be imported. Please '
                'install through:'
                '>>> pip install requests'
                'or ... conda install requests')
        outProj = Proj(init='epsg:4326')
        inProj = Proj(init='epsg:28992')
        lon, lat = transform(inProj, outProj, xy[0], xy[1])
        return (lat, lon)
