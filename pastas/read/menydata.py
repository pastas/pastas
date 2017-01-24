"""
Contains the class MenyData, which represents the data in a Menyanthes file

Currently only the observations (H) and stresses (IN) are supported. Models (M) are ignored.
"""

import scipy.io as sio
import os.path
import numpy as np
import pandas as pd
import datetime as dt


class MenyData:
    def __init__(self, fname):
        # Check if file is present
        if not (os.path.isfile(fname)):
            print('Could not find file ', fname)

        mat = sio.loadmat(fname, struct_as_record=False, squeeze_me=True,
                          chars_as_strings=True)

        # Groundwater level and surface water level - meter with respect to a reference level (e.g. NAP, TAW)
        # Precipitation and evaporation - meters(flux summed starting from date of previous data entry)
        # Extraction - cubic meters(flux summed starting from date of previous data entry)
        self.H = []
        if not isinstance(mat['H'], np.ndarray):
            mat['H'] = [mat['H']]
        for H in mat['H']:
            tindex = [self.matlab2datetime(tval) for tval in
                      H.values[:, 0]]
            # measurement is used as is
            series = pd.Series(H.values[:, 1], index=tindex)

            # round on seconds, to get rid of conversion milliseconds
            series.index = series.index.round('s')

            # add to self.H
            self.H.append(MenyH(series, H.Name))

        self.IN = []
        if not isinstance(mat['IN'], np.ndarray):
            mat['IN'] = [mat['IN']]
        for IN in mat['IN']:
            tindex = [self.matlab2datetime(tval) for tval in
                      IN.values[:, 0]]
            series = pd.Series(IN.values[:, 1], index=tindex)

            # round on seconds, to get rid of conversion milliseconds
            series.index = series.index.round('s')

            if IN.type == 'EVAP' or IN.type == 'PREC' or IN.type == 'WELL':
                # in menyanthes, the flux is summed over the time-step, so devide by the timestep now
                step = series.index.to_series().diff() / pd.offsets.Day(1)
                step = step.values.astype(np.float)
                series = series / step
                if series.values[0] != 0:
                    series = series[1:]

            # add to self.IN
            self.IN.append(MenyIN(series, IN.Name, IN.type))


    def matlab2datetime(self, matlab_datenum):
        """
        Transform a matlab time to a datetime, rounded to seconds
        """
        day = dt.datetime.fromordinal(int(matlab_datenum))
        dayfrac = dt.timedelta(days=float(matlab_datenum) % 1) - dt.timedelta(
            days=366)
        return day + dayfrac

class MenyH:
    def __init__(self, series, name):
        self.series = series
        self.name = name


class MenyIN:
    def __init__(self, series, name, type):
        self.series = series
        self.name = name
        self.type = type
