"""
@author: ruben

"""

import requests
import numpy as np
from datetime import date
import cStringIO
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
            self.stns = '260' # de Bilt (zou ook 'ALL' kunnen zijn)
        else:
            self.stns = stns
             
        self.stations=None
        self.variables=None
        self.data=None
        
    # Alternate constructor
    @classmethod
    def fromfile(cls,fname):
        self = cls()
        with open(fname, 'U') as f:
            self.readdata(f)
        f.close()
        
        return self
        
    def readdata(self,f):
        stations=None
        isLocations=False
        isVariables=False
        line=f.readline()
        while line!='':
            if line.startswith('# '):
                line=line[2:]
            if line.strip()=='':
                # doe niets                
                pass
                
            elif line.startswith('STN '):
                isLocations=True
                isFirstLocation=True;
                line=line.strip()
                titels=line.split();
                titels=[x.replace('(','_') for x in titels]
                titels=[x.replace(r')','') for x in titels]
                titels=[x.encode('utf8') for x in titels]
                
            elif line.startswith('YYYYMMDD'):
                isVariables=True
                isLocations=False
                variables=dict();
                varDes=line.split(' = ')
                variables[varDes[0].strip()]=varDes[1].strip()
                
            elif line.startswith('STN,'):
                # 
                header=line.split(',')
                header=map(str.strip, header)
                header=[x.encode('utf8') for x in header]
                #header=[x.strip().lower() for x in header]
                #header=[x.lower() for x in ["A","B","C"]]
                break
            
            elif isLocations:
                line=line.strip()
                line=line.replace(':','')
                
                # zorg dat delimiter twee spaties is, zodat 'de bilt' als 1 string
                # wordt ingelezen
                line=line.replace('         ','  ')
                line=line.replace('        ','  ')
                line=line.replace('       ','  ')
                line=line.replace('      ','  ')
                line=line.replace('     ','  ')
                line=line.replace('    ','  ')
                line=line.replace('   ','  ')
                s = cStringIO.StringIO(line)

                data = np.genfromtxt(s, dtype=None, delimiter='  ', names=titels)
                data = np.atleast_1d(data)

                if isFirstLocation:
                    stations=data
                    isFirstLocation=False
                else:
                    #raise NameError('Meerdere locaties nog niet ondersteund')
                    stations=rfn.stack_arrays((stations,data),autoconvert=True,usemask=False)
                
            elif isVariables:
                line=line.encode('utf-8')
                varDes=line.split(' = ')
                variables[varDes[0].strip()]=varDes[1].strip()
            
            line=f.readline()
                
        #%% read measurements
        # lees nog een lege regel
        line=f.readline()
        # lees alle metingen in
        if True:
            # omzetten van datatype werkt niet goed
            dtype=[np.float64]*(len(variables)+1)
            dtype[0]=np.int # station id
            dtype[1]='S8'
            dtype=zip(header,dtype);
            data=pd.read_csv(f,header=None,names=header,parse_dates=['YYYYMMDD'],index_col='YYYYMMDD',
                             dtype=dtype, na_values='     ')
        elif True:
            data=pd.read_csv(f,names=header,parse_dates=['YYYYMMDD'],
                             index_col='YYYYMMDD')
            for key, value in variables.iteritems():
                if key not in ['YYYYMMDD','STN']:
                    # reken om naar floats
                    #data.loc[data[key]=='     ', key]=''
                    data[key]=pd.to_numeric(data[key], errors='coerce')
        else:
            dtype=[np.float]*(len(variables)+1)
            dtype[0]=np.int # station id
            dtype[1]='datetime64[s]' # datum in YYYYMMDD-formaat
            dtype=zip(header,dtype);
            # verander de datum naar een datetime
            #string2datetime = lambda x: datetime.strptime(x, '%Y%m%d')
            string2datetime = lambda x: pd.to_datetime(x,format='%Y%m%d')
            
            data = np.genfromtxt(
                f,
                delimiter = ',', # tab separated values
                dtype = dtype,
                converters = {1: string2datetime})
            data=pd.DataFrame(data,index=data['YYYYMMDD'])
            data = data.drop('YYYYMMDD', 1)
            
        #%% van pas de eenheid aan van de metingen
        for key, value in variables.iteritems():
            if ' (-1 for <0.05 mm)' in value or ' (-1 voor <0.05 mm)' in value:
                #erin=data[key]==-1
                #data[key][erin]=data[key][erin]=0.25 # eenheid is nog 0.1 mm
                #data[key][erin]=0.25 # eenheid is nog 0.1 mm
                data.loc[data[key]==-1, key] = 0.25                
                value=value.replace(' (-1 for <0.05 mm)','')
                value=value.replace(' (-1 voor <0.05 mm)','')
            if '0.1 ' in value:
                # reken om van 0.1 naar 1
                data[key]=data[key]*0.1
                value=value.replace('0.1 ','')
            if ' mm'  in value:
                # reken mm om naar m
                data[key]=data[key]*0.001
                value=value.replace(' mm',' m')
            if '(in percents)'  in value:
                # reken procent om naar deel
                #data[key]=data[key]*0.01
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
            
        #%% sluit bestand
        f.close()
        
        self.stations=stations
        self.variables=variables
        self.data=data
             
    def download(self):
        url='http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi'
        if not isinstance(self.stns, basestring):
            if isinstance(self.stns, int):
                self.stns=str(self.stns)
            else:
                raise NameError('Meerdere locaties nog niet ondersteund')
            
        data= {
            'start': self.start.strftime('%Y%m%d'),
            'end': self.end.strftime('%Y%m%d'),
            'inseason':str(int(self.inseason)),
            'vars': self.vars,
            'stns': self.stns,
            }
        self.result=requests.get(url, params=data).text
        self.result=self.result.encode('utf8')
        #f=StringIO(self.result)
        f=cStringIO.StringIO(self.result)
        self.readdata(f)

