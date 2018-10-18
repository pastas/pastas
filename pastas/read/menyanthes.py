"""
Contains the class MenyData, which represents the data in a Menyanthes file

Currently only the observations (H) and stresses (IN) and model results are supported.


# Groundwater level and surface water level - meter with respect to a
# reference level (e.g. NAP, TAW)
# Precipitation and evaporation - meters(flux summed starting from date of
# previous data entry)
# Extraction - cubic meters(flux summed starting from date of previous data
# entry)

"""

from os import path

import numpy as np
from pandas import Series, offsets
from scipy.io import loadmat

from ..timeseries import TimeSeries
from ..utils import matlab2datetime


def read_meny(fname, locations=None, type='H'):
    meny = MenyData(fname, data=type)
    if type == 'H':
        data = meny.H
    elif type == 'IN':
        data = meny.IN
    elif type == 'M':
        data = meny.M
    else:
        raise NotImplementedError('type ' + type + ' not supported (yet)')
    if locations is None:
        locations = data.keys()

    ts = []
    for location in locations:
        metadata = {}
        metadata['x'] = data[location]['xcoord']
        metadata['y'] = data[location]['ycoord']
        metadata['z'] = np.mean(
            (data[location]['upfiltlev'], data[location]['lowfiltlev']))
        metadata['projection'] = 'epsg:28992'
        if type == 'H':
            kind = 'oseries'
        else:
            if data[location]['Type'] == 'prec':
                kind = 'prec'
            elif data[location]['Type'] == 'evap':
                kind = 'evap'
            elif data[location]['Type'] == 'well':
                kind = 'well'
            elif data[location]['Type'] == 'river':
                kind = 'waterlevel'
            else:
                kind = None
        if type == 'M':
            kind = None
        ts.append(TimeSeries(data[location]['values'], name=location,
                             metadata=metadata, settings=kind))
    if len(ts) == 1:
        ts = ts[0]
    return ts


class MenyData:
    def __init__(self, fname, data='all'):
        """This class reads a menyanthes file.

        Parameters
        ----------
        fname: str
            String with the filename and path to a menyanthes file.


        """

        mat = self.read_file(fname)

        # Figure out which data to collect from the file.
        if data == 'all':
            data = ['H', 'IN', 'M']
        elif type(data) is str:
            data = [data]

        if 'IN' in data:
            self.IN = dict()
            self.read_in(mat)

        if 'H' in data:
            self.H = dict()
            self.read_h(mat)

        if 'M' in data:
            self.M = dict()
            self.read_m(mat)

        del mat  # Delete the mat file from memory again

    def read_file(self, fname):
        """This method is used to read the file.

        """

        # Check if file is present
        if not (path.isfile(fname)):
            print('Could not find file ', fname)

        mat = loadmat(fname, struct_as_record=False, squeeze_me=True,
                      chars_as_strings=True)

        return mat

    def read_in(self, mat):
        """Read the input part.

        """

        # Check if more then one time series model is present
        if not isinstance(mat['IN'], np.ndarray):
            mat['IN'] = [mat['IN']]

        # Read all the time series models
        for i, IN in enumerate(mat['IN']):
            data = dict()

            for name in IN._fieldnames:
                if name != 'values':
                    data[name] = getattr(IN, name)
                else:
                    tindex = [matlab2datetime(tval) for tval in
                              IN.values[:, 0]]
                    series = Series(IN.values[:, 1], index=tindex)

                    # round on seconds, to get rid of conversion milliseconds
                    series.index = series.index.round('s')

                    if hasattr(IN, 'type'):
                        IN.Type = IN.type

                    if IN.Type in ['EVAP', 'PREC', 'WELL']:
                        # in menyanthes, the flux is summed over the
                        # time-step, so divide by the timestep now
                        step = series.index.to_series().diff() / offsets.Day(
                            1)
                        step = step.values.astype(np.float)
                        series = series / step
                        if series.values[0] != 0:
                            series = series[1:]

                    data['values'] = series

            # add to self.IN
            if not hasattr(IN, 'Name') and not hasattr(IN, 'name'):
                IN.Name = 'IN' + str(i)
            if hasattr(IN, 'name'):
                IN.Name = IN.name

            self.IN[IN.Name] = data

    def read_h(self, mat):
        """Read the dependent variable part.

        """

        # Check if more then one time series model is present
        if not isinstance(mat['H'], np.ndarray):
            mat['H'] = [mat['H']]

        # Read all the time series models
        for i, H in enumerate(mat['H']):
            data = dict()

            for name in H._fieldnames:
                if name != 'values':
                    data[name] = getattr(H, name)
                else:
                    if H.values.size == 0:
                        # when diver-files are used, values will be empty
                        series = Series()
                    else:
                        tindex = [matlab2datetime(tval) for tval in
                                  H.values[:, 0]]
                        # measurement is used as is
                        series = Series(H.values[:, 1], index=tindex)
                        # round on seconds, to get rid of conversion milliseconds
                        series.index = series.index.round('s')
                    data['values'] = series

            # add to self.H
            if not hasattr(H, 'Name') and not hasattr(H, 'name'):
                H.Name = 'H' + str(i)  # Give it the index name
            if hasattr(H, 'name'):
                H.Name = H.name
            if len(H.Name) == 0:
                H.Name = H.tnocode

            self.H[H.Name] = data

    def read_m(self, mat):
        """Read the result part.

        """
        # Check if more then one time series model is present
        if not isinstance(mat['M'], np.ndarray):
            mat['M'] = [mat['M']]

        # Read all the time series models
        for i, M in enumerate(mat['M']):
            data = dict()

            for name in M._fieldnames:
                if name != 'values':
                    data[name] = getattr(M, name)
                else:
                    tindex = [matlab2datetime(tval) for tval in
                              M.values[:, 0]]
                    # measurement is used as is
                    series = Series(M.values[:, 1], index=tindex)
                    # round on seconds, to get rid of conversion milliseconds
                    series.index = series.index.round('s')
                    data['values'] = series

            # add to self.H
            if not hasattr(M, 'Name') and not hasattr(M, 'name'):
                M.Name = 'M' + str(i)  # Give it the index name
            if hasattr(M, 'name'):
                M.Name = M.name

            self.M[M.Name] = data
