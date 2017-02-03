"""

@author: ruben
"""

import warnings

import numpy as np
import pandas as pd

from pastas.read.dinodata import DinoGrondwaterstand
from pastas.read.knmidata import KnmiStation
from pastas.read.menydata import MenyData


def dinodata(fname):
    """This method can be used to import files from Dinoloket that contain
     groundwater level measurements (https://www.dinoloket.nl/)

    Parameters
    ----------
    fname: str
        Filename and path to a Dino file.

    Returns
    -------
    DataModel: object
        returns a standard Pastas DataModel object.

    """
    data = DataModel()
    dino = DinoGrondwaterstand(fname)

    data.series = dino.data
    data.x = dino.x
    data.y = dino.y
    data.latlon = rd2wgs(data.x, data.y)
    data.metadata = dino.meta[-1]

    return data


def knmidata(fname):
    """This method can be used to import KNMI data.
    ()

    Parameters
    ----------
    fname: str
        Filename and path to a Dino file.

    Returns
    -------
    DataModel: object
        returns a standard Pastas DataModel object.

    """
    data = DataModel()
    knmi = KnmiStation.fromfile(fname)

    data.data = knmi.data
    if knmi.stations is not None and not knmi.stations.empty:
        data.x = knmi.stations['LAT_north'][0]
        data.y = knmi.stations['LON_east'][0]
        data.metadata = knmi.stations

    return data


def menydata(fname, get_data='all'):
    """This method can be used to import a menyanthes project file.

    Parameters
    ----------
    fname: str
        Filename and path to a Dino file.

    Returns
    -------
    DataModel: object
        returns a standard Pastas DataModel object.

    """

    meny = MenyData(fname, get_data=get_data)

    # H_list = []
    # for H in meny.H:
    #     data = DataModel()
    #
    #     H_list.append(data)
    #
    # if type == 'all':
    #     data.data = meny
    # else:
    #     data.data = meny[type]

    return meny


class DataModel():
    def __init__(self):
        """This is the standard datamodel class that is returned by all import
         methods.

        """
        self.data = pd.DataFrame()
        self.x = None
        self.y = None
        self.metadata = {}


def rd2wgs(x, y):
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
