"""
Contains the class MenyData, which represents the data in a Menyanthes file
"""

import scipy.io as sio
import os.path
import numpy as np
import pandas as pd
import datetime as dt

class MenyData:
    def __init__(self, fname):
        if (os.path.isfile(fname)):
            mat = sio.loadmat(fname, struct_as_record=False, squeeze_me=True, chars_as_strings=True)

            self.H=[]
            if not isinstance(mat['H'],np.ndarray):
                mat['H']=[mat['H']]
            for H in mat['H']:
                tindex = [self.matlab2datetime(tval) for tval in H.values[:,0]]
                series=pd.Series(H.values[:, 1], index=tindex)
                self.H.append(MenyH(series,H.Name))

            self.IN = []
            if not isinstance(mat['IN'], np.ndarray):
                mat['IN'] = [mat['IN']]
            for IN in mat['IN']:
                tindex = [self.matlab2datetime(tval) for tval in IN.values[:, 0]]
                series = pd.Series(IN.values[:, 1], index=tindex)
                self.IN.append(MenyIN(series, IN.Name))

        else:
            print 'Could not find file ', fname

    def matlab2datetime(self,matlab_datenum):
        day = dt.datetime.fromordinal(int(matlab_datenum))
        dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)
        return day + dayfrac

class MenyH:
    def __init__(self,series,name):
        self.series=series
        self.name=name

class MenyIN:
    def __init__(self,series,name):
        self.series=series
        self.name=name

