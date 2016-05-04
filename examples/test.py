import pandas as pd
from gwtsa import *

# Import and check the observed groundwater time series
gwdata = pd.read_csv('B58C0698001_0.csv', skiprows=11,
                     parse_dates=['PEIL DATUM TIJD'],
                     index_col='PEIL DATUM TIJD',
                     skipinitialspace=True)
gwdata.rename(columns={'STAND (MV)': 'h'}, inplace=True)
gwdata.index.names = ['date']
gwdata.h *= 0.01
oseries = 30.17 - gwdata.h  # NAP

# Import and check the observed precipitation series
rain = pd.read_csv('Heibloem_rain_data.dat', skiprows=4, delim_whitespace=True,
                   parse_dates=['date'],
                   index_col='date')
rain = rain['1980':]  # cut off everything before 1980
rain = rain.precip
rain /= 1000.0  # Meters

# Import and check the observed evaporation series
evap = pd.read_csv('Maastricht_E_June2015.csv', skiprows=4, sep=';',
                   parse_dates=['DATE'],
                   index_col='DATE')
evap.rename(columns={'VALUE (m-ref)': 'evap'}, inplace=True)
evap = evap['1980':]  # cut off everything before 1980
evap = evap.evap

recharge = rain - 0.8 * evap

#oseries -= oseries.mean()
#recharge -= recharge.mean()

# Create the time series model
ml = Model(oseries)
#ts1 = Tseries(recharge, Gamma(), name='recharge')
ts1 = Tseries2([rain, evap], Gamma(), name='recharge')
ml.addtseries(ts1)
d = Constant(oseries.min())  # Using oseries.min() as initial value slightly
# reduces computation time
ml.addtseries(d)
n = NoiseModel()
ml.addnoisemodel(n)

# Solve the time series model
ml.solve()
ml.plot()

# Solve for a certain period
#ml.solve(tmin='1965', tmax='1990')
#ml.plot_results()  # More advanced plotting option
