# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:33:13 2016

@author: ruben
"""

from gwtsa.imports.dinodata import DinoGrondwaterstand
from gwtsa.imports.knmidata import KnmiStation
from pyproj import Proj, transform

class ImportSeries:
    def __init__(self, fname, filetype, variable = None):
        if filetype=='dino':
            dino = DinoGrondwaterstand(fname)
            self.series=dino.stand
            
            self.xy=(dino.x, dino.y)
            self.latlon=self.rd2wgs(self.xy)
            self.meta=dino.meta[-1]
        elif filetype=='knmi':
            knmi = KnmiStation.fromfile(fname)
            self.series=knmi.data[variable]
            self.latlon=(knmi.stations['LAT_north'][0],
                         knmi.stations['LON_east'][0])
            names=knmi.stations.dtype.names
            self.meta=dict(zip(names,knmi.stations[0]))
            
        elif filetype=='usgs':
            # not implemented yet
            pass
        elif filetype=='csv':
            # not implemented yet
            pass
        else:
            raise Exception('Unknown filtype')
    def rd2wgs(self,xy):
        outProj = Proj(init='epsg:4326')
        inProj = Proj(init='epsg:28992')
        lon,lat = transform(inProj,outProj,xy[0],xy[1])
        return (lat,lon)