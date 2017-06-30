"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.

"""
from pastas import *

# read observations
obs = read.dinodata('data/B58C0698001_1.csv')

# Create the time series model
ml = Model(obs.series)

# read weather data
rain = read.knmidata('data/neerslaggeg_HEIBLOEM-L_967-2.txt', variable='RD')
# from pandas import read_csv
# rain = read_csv('data/Heibloem_rain_data.dat', skiprows=4, sep=' ', skipinitialspace=True, parse_dates='date', index_col='date')
evap = read.knmidata('data/etmgeg_380.txt', variable='EV24')

## Create stress
ts = Tseries2(rain.series, evap.series, Gamma, name='recharge')
ml.add_tseries(ts)

## Add noise model
n = NoiseModel()
ml.add_noisemodel(n)

## Solve
ml.solve(noise=True, weights="swsi")

ml.plot()
