"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.

"""
from pastas import *

# read observations
obs = ReadSeries('data/B58C0698001_1.csv', 'dino')

# Create the time series model
ml = Model(obs.series)

# read weather data
#rain = ReadSeries('data/neerslaggeg_HEIBLOEM-L_967-2.txt', 'knmi', variable='RD')
from pandas import read_csv
rain = read_csv('data/Heibloem_rain_data.dat', skiprows=4, sep=' ', skipinitialspace=True, parse_dates='date', index_col='date')
evap = ReadSeries('data/etmgeg_380.txt', 'knmi', variable='EV24')

## Create stress
#ts = Tseries2(rain.series, evap.series[1965:], Gamma, name='recharge')
ts = Tseries2(rain.precip, evap.series[1965:], Gamma, name='recharge')
ml.add_tseries(ts)

## Add noise model
n = NoiseModel()
ml.add_noisemodel(n)

## Solve
ml.solve(tmin='11-1985', tmax='1-2011')
