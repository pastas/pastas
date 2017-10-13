"""
@author: ruben calje
Reads daily meteorological data in a file from stations of the KNMI:
knmi = KnmiStation.fromfile(filename)

Data can be downloaded for the meteorological stations at:
https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
or
http://projects.knmi.nl/klimatologie/daggegevens/selectie.cgi

For the rainfall stations data is available at:
https://www.knmi.nl/nederland-nu/klimatologie/monv/reeksen

Also, data from the meteorological stations can be downloaded directly, for example with
knmi = KnmiStation(stns=260, start=datetime(1970, 1, 1), end=datetime(1971, 1, 1))  # 260 = de bilt
knmi.download()
For now the direct download only works for meteorological stations and daily data (so no rainfall stations or hourly data)


"""

from __future__ import print_function, division

from datetime import date

import pandas as pd
from pastas.timeseries import TimeSeries


def read_knmi(fname, variables='RD'):
    """This method can be used to import KNMI data.

    Parameters
    ----------
    fname: str
        Filename and path to a Dino file.

    Returns
    -------
    DataModel: object
        returns a standard Pastas DataModel object.

    """
    knmi = KnmiStation.fromfile(fname)
    if variables is None:
        variables = knmi.variables.keys()
    if type(variables) == str:
        variables = [variables]

    stn_codes = knmi.data['STN'].unique()

    ts = []
    for code in stn_codes:
        for variable in variables:
            if variable not in knmi.data.keys():
                raise (ValueError("variable %s is not in this dataset. Please use one of "
                                  "the following keys: %s" % (variable, knmi.data.keys())))

            series = knmi.data.loc[knmi.data['STN'] == code, variable]
            # get rid of the hours when data is daily
            if pd.infer_freq(series.index) == 'D':
                series.index = series.index.normalize()

            metadata = {}
            if knmi.stations is not None and not knmi.stations.empty:
                station = knmi.stations.loc[str(code), :]
                metadata['x'] = station.LON_east
                metadata['y'] = station.LAT_north
                metadata['z'] = station.ALT_m
                metadata['projection'] = 'epsg:4326'
                stationname = station.NAME
            else:
                stationname = str(code)
            metadata['description'] = knmi.variables[variable]
            if variable == 'RD' or variable == 'RH':
                kind = 'prec'
            elif variable == 'EV24':
                kind = 'evap'
            else:
                kind = None
            ts.append(TimeSeries(series, name=variable + stationname, metadata=metadata, kind=kind))
    if len(ts) == 1:
        ts = ts[0]
    return ts


class KnmiStation:
    def __init__(self, start=None, end=None, inseason=False, vars='ALL',
                 stns='260'):
        if start is None:
            self.start = date(date.today().year, 1, 1)
        else:
            self.start = start
        if end is None:
            self.end = date.today()
        else:
            self.end = end
        self.inseason = inseason
        self.vars = vars
        self.stns = stns  # de Bilt (zou ook 'ALL' kunnen zijn)

        self.stations = None
        self.variables = dict()
        self.data = None

    # Alternate constructor
    @classmethod
    def fromfile(cls, fname):
        self = cls()
        with open(fname, 'r') as f:
            self.readdata(f)
        f.close()

        return self

    def download(self):
        """

        :return:
        """
        # Import the necessary modules (optional and not included in the
        # installation of pastas).
        try:
            import requests
        except ImportError:
            raise ImportWarning(
                'The module requests could not be imported. '
                'Please install through:'
                '>>> pip install requests'
                'or:'
                '>>> conda install requests')
        try:
            # StringIO changed from py27 to py35
            try:
                from StringIO import StringIO
            except ImportError:
                from io import StringIO
        except ImportError:
            raise ImportWarning(
                'The module requests could not be imported. Please '
                'install through:'
                '>>> pip install StringIO (for python 2)'
                'or: '
                '>>> pip install io (for python 3)')

        url = 'http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi'
        if not isinstance(self.stns, str):
            if isinstance(self.stns, int):
                self.stns = str(self.stns)
            else:
                raise NameError('Meerdere locaties nog niet ondersteund')

        data = {
            'start': self.start.strftime('%Y%m%d'),
            'end': self.end.strftime('%Y%m%d'),
            'inseason': str(int(self.inseason)),
            'vars': self.vars,
            'stns': self.stns,
        }
        self.result = requests.get(url, params=data).text

        f = StringIO(self.result)
        self.readdata(f)

    def readdata(self, f):
        isLocations = False
        line = f.readline()
        isMeteo = line.startswith('# ')

        # Process the header information (Everything < 'STN,')
        while 'STN,' not in line:
            # Pre-format the line
            line = line.strip('\n')
            line = line.lstrip('# ')

            # If line is empty, skipline
            if line.strip() == '':
                pass
            # If line contains station info (can only happen for meteorological stations)
            elif isMeteo and line.startswith('STN '):
                isLocations = True
                line = line.strip()
                titels = line.split()
                titels = [x.replace('(', '_') for x in titels]
                titels = [x.replace(r')', '') for x in titels]

                # Create pd.DataFrame for station data
                if not self.stations:
                    self.stations = pd.DataFrame(columns=titels)
                    self.stations.set_index(['STN'], inplace=True)

            # If line contains variables
            elif ' = ' in line:
                isLocations = False
                varDes = line.split(' = ')
                self.variables[varDes[0].strip()] = varDes[1].strip()
            # If location data is recognized in the previous line
            elif isLocations:
                # Format line. Ensure delimiter is two spaces to read the
                # location correctly
                line = line.strip()
                line = line.replace(':', '')
                line = line.replace('         ', '  ')
                line = line.replace('        ', '  ')
                line = line.replace('       ', '  ')
                line = line.replace('      ', '  ')
                line = line.replace('     ', '  ')
                line = line.replace('    ', '  ')
                line = line.replace('   ', '  ')
                # Add station location data
                line = line.split('  ')
                self.stations.loc[line[0]] = line[1:]

            # Read in a new line and start over
            line = f.readline()

        # The header information of the datablock
        line = line.strip('\n')
        line = line.lstrip('# ')
        header = line.split(',')
        header = [item.lstrip().rstrip() for item in header]
        line = f.readline()  # Skip empty line after header

        # Process the datablock
        if False:
            # older method, is much slower
            string2datetime = lambda x: pd.to_datetime(x, format='%Y%m%d')

            data = pd.read_csv(f, header=None, names=header,
                               parse_dates=['YYYYMMDD'], index_col='YYYYMMDD',
                               na_values='     ', converters={1: string2datetime})
        else:
            # newer method, calculating the date afterwards is much faster
            data = pd.read_csv(f, header=None, names=header, na_values='     ')
            data.set_index(pd.to_datetime(data.YYYYMMDD, format='%Y%m%d'), inplace=True)
            data = data.drop('YYYYMMDD', axis=1)

        # convert the hours if provided
        if 'HH' in data.keys():
            # hourly data, Hourly division 05 runs from 04.00 UT to 5.00 UT
            data.index = data.index + pd.to_timedelta(data['HH'], unit='h')
            data.pop('HH')
        else:
            # daily data
            if 'RD' in data.keys():
                # daily precipitation amount in 0.1 mm over the period 08.00 preceding day - 08.00 UTC present day
                data.index = data.index + pd.to_timedelta(8, unit='h')
            else:
                # add a full day for meteorologiscal data, so that the timestamp is at the end of the period that the data represenets
                data.index = data.index + pd.to_timedelta(1, unit='d')

        # from UT to UT+1 (standard-time in the Netherlands)
        data.index = data.index + pd.to_timedelta(1, unit='h')

        # Delete empty columns
        if '' in data.columns:
            data.drop('', axis=1, inplace=True)

        # Adjust the unit of the measurements
        for key, value in self.variables.items():
            # test if key existst in data
            if key not in data.keys():
                if key == 'YYYYMMDD' or key == 'HH':
                    pass
                elif key == 'T10N':
                    self.variables.pop(key)
                    key = 'T10'
                else:
                    raise NameError(key + ' does not exist in data')
            if ' (-1 for <0.05 mm)' in value or ' (-1 voor <0.05 mm)' in value:
                # set 0.025 mm where data == -1
                data.loc[data[key] == -1, key] = 0.25  # unit is still 0.1 mm
                value = value.replace(' (-1 for <0.05 mm)', '')
                value = value.replace(' (-1 voor <0.05 mm)', '')
            if '0.1 ' in value:
                # transform 0.1 to 1
                data[key] = data[key] * 0.1
                value = value.replace('0.1 ', '')
            if ' tiende ' in value:
                # transform 0.1 to 1
                data[key] = data[key] * 0.1
                value = value.replace(' tiende ', ' ')
            if ' mm' in value:
                # transform mm to m
                data[key] = data[key] * 0.001
                value = value.replace(' mm', ' m')
            if ' millimeters' in value:
                # transform mm to m
                data[key] = data[key] * 0.001
                value = value.replace(' millimeters', ' m')
            if '(in percents)' in value:
                # do not adjust (yet)
                pass
            if 'hPa' in value:
                # do not adjust (yet)
                pass
            if 'J/cm2' in value:
                # do not adjust (yet)
                pass
            # Store new variable
            self.variables[key] = value

        # Close file
        f.close()

        self.data = data
