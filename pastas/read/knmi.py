"""
@author: ruben calje

"""

import pandas as pd
import warnings

from ..timeseries import TimeSeries


def read_knmi(fname, variables='RD'):
    """This method can be used to import KNMI data.

    Parameters
    ----------
    fname: str
        Filename and path to a Dino file.
    variables: str
        String with the variable name to extract.

    Returns
    -------
    ts: pastas.TimeSeries
        returns a Pastas TimeSeries object or a list of objects.

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
                raise (ValueError(
                    "variable %s is not in this dataset. Please use one of "
                    "the following keys: %s" % (variable, knmi.data.keys())))

            series = knmi.data.loc[knmi.data['STN'] == code, variable]
            # get rid of the hours when data is daily
            if pd.infer_freq(series.index) == 'D':
                series.index = series.index.normalize()

            metadata = {}
            if knmi.stations is not None and not knmi.stations.empty:
                station = knmi.stations.loc[code, :]
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
            ts.append(TimeSeries(series, name=variable + ' ' + stationname,
                                 metadata=metadata, settings=kind))
    if len(ts) == 1:
        ts = ts[0]
    return ts


class KnmiStation:
    """
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

    Hourly data can be downloaded with the 'interval'keyword set to 'hour' or 'hourly':
    knmi = KnmiStation(stns=260, start='2017', end='2018', interval='hourly')

    Data from rainfall-stations can be downloaded by asking for the variable 'RD' (the stns variable now describes codes for rainfall-stations):
    knmi = KnmiStation(stns=550, start='2018', end='2019', vars='RD') # rainfall-station in de bilt

    Times are recalculated to UT+1 (standard-time in the Netherlands), from UT.
    Also the datetime-index of the data is set at the end of the period that the data describes.
    So the rainfall between 2018-01-01 09:00:00 (08:00:00 UT) and 2018-01-02 09:00:00 (08:00:00 UT) gets the timestamp of 2018-01-02 09:00:00

    Units in the data of the knmi are recalculated to more basic SI-units. So mm are transformed to m, and a factor of 0,1 is transformed to 1.

    A description of the variables is found in knmi.variables.
    Information about the measurement-station(s) is found in knmi.stations.
    The measurement-data itself is found in knmi.data
    """

    def __init__(self, *args, **kwargs):
        self.stations = pd.DataFrame()
        self.variables = dict()
        self.data = pd.DataFrame()
        if len(args) > 0 or len(kwargs) > 0:
            warnings.warn("In the future use KnmiStation.download(**kwargs) "
                          "instead "
                          "of KnmiStation(**kwargs)",
                          FutureWarning)
            self._download(*args, **kwargs)
            # diable download method, as old code will call this again
            self.download = lambda *args, **kwargs: None
        else:
            # change download method to the instance-method
            self.download = self._download

    # Construct KnmiStation from file
    @classmethod
    def fromfile(cls, fname):
        self = cls()
        with open(fname, 'r') as f:
            self.readdata(f)
        f.close()
        return self

    # Construct KnmiStation from download
    @classmethod
    def download(cls, start=None, end=None, inseason=False, vars='ALL',
                 stns=260, interval='daily'):
        """

        """
        self = cls()
        self._download(start=start, end=end, inseason=inseason, vars=vars,
                       stns=stns, interval=interval)
        return self

    def _download(self, start=None, end=None, inseason=False, vars='ALL',
                  stns=260, interval='daily'):
        # Import the necessary modules (optional and not included in the
        # installation of pastas).
        try:
            import requests
        except ImportError:
            raise ImportError(
                'The module requests could not be imported. '
                'Please install through:'
                '>>> pip install requests'
                'or:'
                '>>> conda install requests')

        from io import StringIO

        if start is None:
            start = pd.Timestamp(pd.Timestamp.today().year, 1, 1)
        else:
            start = pd.to_datetime(start)
        if end is None:
            end = pd.Timestamp.today()
        else:
            end = pd.to_datetime(end)

        if interval.startswith('hour') and vars == 'RD':
            raise (
                ValueError('Interval can not be hourly for rainfall-stations'))

        if interval.startswith('hour'):
            # hourly data from meteorological stations
            url = 'http://projects.knmi.nl/klimatologie/uurgegevens/getdata_uur.cgi'
        elif vars == 'RD':
            # daily data from rainfall-stations
            url = 'http://projects.knmi.nl/klimatologie/monv/reeksen/getdata_rr.cgi'
        else:
            # daily data from meteorological stations
            url = 'http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi'
        if not isinstance(stns, str):
            if isinstance(stns, int):
                stns = str(stns)
            else:
                stns = [str(i) for i in stns]
                stns = ":".join(stns)

        if interval.startswith('hour'):
            data = {
                'start': start.strftime('%Y%m%d') + '01',
                'end': end.strftime('%Y%m%d') + '24',
                'vars': vars,
                'stns': stns,
            }
        else:
            data = {
                'start': start.strftime('%Y%m%d'),
                'end': end.strftime('%Y%m%d'),
                'inseason': str(int(inseason)),
                'vars': vars,
                'stns': stns,
            }
        result = requests.get(url, params=data).text

        f = StringIO(result)
        self.readdata(f)

    def readdata(self, f):
        stations = pd.DataFrame()
        variables = dict()

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
                stations = pd.DataFrame(columns=titels)
                stations.set_index(['STN'], inplace=True)

            # If line contains variables
            elif ' = ' in line:
                isLocations = False
                varDes = line.split(' = ')
                variables[varDes[0].strip()] = varDes[1].strip()
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
                stn = int(line[0])

                def maybe_float(s):
                    try:
                        return float(s)
                    except (ValueError, TypeError):
                        return s

                line = [maybe_float(v) for v in line[1:]]
                stations.loc[stn] = line

            # Read in a new line and start over
            line = f.readline()

        # The header information of the datablock
        line = line.strip('\n')
        line = line.lstrip('# ')
        header = line.split(',')
        header = [item.lstrip().rstrip() for item in header]
        pos = f.tell()
        line = f.readline()  # Skip empty line after header
        if line not in ["\n", "\r\n", "# \n", '# \r\n']:
            # sometimes there is no empty line between the header and the data
            f.seek(pos)
        # Process the datablock
        if False:
            # older method, is much slower
            string2datetime = lambda x: pd.to_datetime(x, format='%Y%m%d')

            data = pd.read_csv(f, header=None, names=header,
                               parse_dates=['YYYYMMDD'], index_col='YYYYMMDD',
                               na_values='     ',
                               converters={1: string2datetime})
        else:
            # newer method, calculating the date afterwards is much faster
            data = pd.read_csv(f, header=None, names=header, na_values='     ')
            data.set_index(pd.to_datetime(data.YYYYMMDD, format='%Y%m%d'),
                           inplace=True)
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
        for key, value in variables.items():
            # test if key existst in data
            if key not in data.keys():
                if key == 'YYYYMMDD' or key == 'HH':
                    pass
                elif key == 'T10N':
                    variables.pop(key)
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
            variables[key] = value

        # Close file
        f.close()

        self.stations = stations
        self.variables = variables
        self.data = data
