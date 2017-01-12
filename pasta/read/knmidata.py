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

import requests
import numpy as np
from datetime import date, datetime
# from io import StringIO
from StringIO import StringIO
import pandas as pd
import numpy.lib.recfunctions as rfn


class KnmiStation:
    def __init__(self, start=None, end=None, inseason=None, vars=None, stns=None):
        if start is None:
            self.start = date(date.today().year, 1, 1)
        else:
            self.start = start

        if end is None:
            self.end = date.today()
        else:
            self.end = end

        if inseason is None:
            self.inseason = False
        else:
            self.inseason = inseason

        if vars is None:
            self.vars = 'ALL'
        else:
            self.vars = vars

        if stns is None:
            self.stns = '260'  # de Bilt (zou ook 'ALL' kunnen zijn)
        else:
            self.stns = stns

        self.stations = None
        self.variables = None
        self.data = None

    # Alternate constructor
    @classmethod
    def fromfile(cls, fname):
        self = cls()
        with open(fname, 'U') as f:
            self.readdata(f)
        f.close()

        return self

    def readdata(self, f):
        stations = None
        variables = dict()
        isLocations = False
        isVariables = False
        line = f.readline()
        if line.startswith('# '):
            # data from meteorological station
            while line != '':
                if line.startswith('# '):
                    line = line[2:]
                if line.strip() == '':
                    # doe niets
                    pass

                elif line.startswith('STN '):
                    isLocations = True
                    isFirstLocation = True
                    line = line.strip()
                    titels = line.split()
                    titels = [x.replace('(', '_') for x in titels]
                    titels = [x.replace(r')', '') for x in titels]
                    titels = [x.encode('utf8') for x in titels]

                elif line.startswith('YYYYMMDD'):
                    isVariables = True
                    isLocations = False
                    varDes = line.split(' = ')
                    variables[varDes[0].strip()] = varDes[1].strip()

                elif line.startswith('STN,'):
                    #
                    header = line.split(',')
                    header = map(str.strip, header)
                    header = [x.encode('utf8') for x in header]
                    # header=[x.strip().lower() for x in header]
                    # header=[x.lower() for x in ["A","B","C"]]
                    break

                elif isLocations:
                    line = line.strip()
                    line = line.replace(':', '')

                    # zorg dat delimiter twee spaties is, zodat 'de bilt' als 1 string
                    # wordt ingelezen
                    line = line.replace('         ', '  ')
                    line = line.replace('        ', '  ')
                    line = line.replace('       ', '  ')
                    line = line.replace('      ', '  ')
                    line = line.replace('     ', '  ')
                    line = line.replace('    ', '  ')
                    line = line.replace('   ', '  ')
                    s = StringIO(line)

                    data = np.genfromtxt(s, dtype=None, delimiter='  ', names=titels)
                    data = np.atleast_1d(data)

                    if isFirstLocation:
                        stations = data
                        isFirstLocation = False
                    else:
                        # raise NameError('Meerdere locaties nog niet ondersteund')
                        stations = rfn.stack_arrays((stations, data), autoconvert=True, usemask=False)

                elif isVariables:
                    line = line.encode('utf-8')
                    varDes = line.split(' = ')
                    variables[varDes[0].strip()] = varDes[1].strip()

                line = f.readline()

            # lees nog een lege regel
            line = f.readline()
        else:
            # data from precipitation station
            if line.startswith('# '):
                line = line[2:]
            while not line.startswith('STN,') and line != '':
                if ' = ' in line:
                    line = line.encode('utf-8')
                    varDes = line.split(' = ')
                    variables[varDes[0].strip()] = varDes[1].strip()
                line = f.readline()
                if line.startswith('# '):
                    line = line[2:]
            header = line.split(',')
            header = map(str.strip, header)
            header = [x.encode('utf8') for x in header]
            #header.remove('')
        # read the measurements
        if True:
            # use pandas with the right datatypes
            dtype = [np.float64] * (len(variables) + 1)
            dtype[0] = np.int  # station id
            dtype[1] = 'S8'
            if True:
                # do not do anything with hours while reading
                dtype = zip(header, dtype)
                data = pd.read_csv(f, header=None, names=header, parse_dates=['YYYYMMDD'], index_col='YYYYMMDD',
                                   dtype=dtype, na_values='     ')
                if True:
                    # add hours to the index
                    if 'HH' in data.keys():
                        data.index = data.index+pd.to_timedelta(data['HH'], unit='h')
                        data.pop('HH')
            else:
                # convert the hours right away
                if header.__contains__('HH'):
                    dtype[header.index('HH')] = 'S5'
                    parse_dates=[['YYYYMMDD', 'HH']]
                    date_parser = self.parse_day_and_hour
                else:
                    parse_dates = ['YYYYMMDD']
                    date_parser = None
                dtype = zip(header, dtype)
                data = pd.read_csv(f, header=None, names=header, parse_dates=parse_dates, index_col=0,
                                   dtype=dtype, na_values='     ', date_parser=date_parser)

            # sometimes an empty column is added at the end off the file (every line ends with ,)
            if '' in data.columns:
                data.drop('', axis=1, inplace=True)
        elif True:
            # old method: read everything as string, and later transform to numeric
            data = pd.read_csv(f, names=header, parse_dates=['YYYYMMDD'],
                               index_col='YYYYMMDD')
            for key, value in variables.iteritems():
                if key not in ['YYYYMMDD', 'STN']:
                    # reken om naar floats
                    # data.loc[data[key]=='     ', key]=''
                    data[key] = pd.to_numeric(data[key], errors='coerce')
        else:
            # old method: use genfromtxt, and then transfrom to dataframe
            dtype = [np.float] * (len(variables) + 1)
            dtype[0] = np.int  # station id
            dtype[1] = 'datetime64[s]'  # datum in YYYYMMDD-formaat
            dtype = zip(header, dtype)
            # verander de datum naar een datetime
            # string2datetime = lambda x: datetime.strptime(x, '%Y%m%d')
            string2datetime = lambda x: pd.to_datetime(x, format='%Y%m%d')

            data = np.genfromtxt(
                f,
                delimiter=',',  # tab separated values
                dtype=dtype,
                converters={1: string2datetime})
            data = pd.DataFrame(data, index=data['YYYYMMDD'])
            data = data.drop('YYYYMMDD', 1)

        # %% van pas de eenheid aan van de metingen
        for key, value in variables.iteritems():
            # test if key existst in data (YYYYMMDD and possibly HH are allready replaced by the index
            if key not in data.keys():
                if key == 'YYYYMMDD' or key == 'HH':
                    pass
                elif key == 'T10N':
                    variables.pop(key)
                    key = 'T10'
                else:
                    raise NameError(key + ' does not exist in data')
            if ' (-1 for <0.05 mm)' in value or ' (-1 voor <0.05 mm)' in value:
                # erin=data[key]==-1
                # data[key][erin]=data[key][erin]=0.25 # eenheid is nog 0.1 mm
                # data[key][erin]=0.25 # eenheid is nog 0.1 mm
                data.loc[data[key] == -1, key] = 0.25
                value = value.replace(' (-1 for <0.05 mm)', '')
                value = value.replace(' (-1 voor <0.05 mm)', '')
            if '0.1 ' in value:
                # reken om van 0.1 naar 1
                data[key] = data[key] * 0.1
                value = value.replace('0.1 ', '')
            if ' tiende ' in value:
                # reken om van 0.1 naar 1
                data[key] = data[key] * 0.1
                value = value.replace(' tiende ', ' ')
            if ' mm' in value:
                # reken mm om naar m
                data[key] = data[key] * 0.001
                value = value.replace(' mm', ' m')
            if ' millimeters' in value:
                # reken mm om naar m
                data[key] = data[key] * 0.001
                value = value.replace(' millimeters', ' m')
            if '(in percents)' in value:
                # reken procent om naar deel
                # data[key]=data[key]*0.01
                # reken (nog) niet om
                pass
            if 'hPa' in value:
                # reken (nog) niet om
                pass
            if 'J/cm2' in value:
                # reken (nog) niet om
                pass
            # bewaar aangepaste variabele
            variables[key] = value

        # %% sluit bestand
        f.close()

        self.stations = stations
        self.variables = variables
        self.data = data

    def parse_day_and_hour(self,day_hour):
        d=datetime.strptime(day_hour, '%Y%m%d %H')
        return d


    def download(self):
        url = 'http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi'
        if not isinstance(self.stns, basestring):
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
        self.result = self.result.encode('utf8')

        f = StringIO(self.result)
        self.readdata(f)
