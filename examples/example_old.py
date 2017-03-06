"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.

"""
from pastas import *

# read observations
fname = 'data/B32D0136001_1.csv'
obs = read.dinodata(fname)

# Create the time series model
ml = Model(obs.series/100.0)

# read climate data
fname = 'data/KNMI_Bilt.txt'
RH = read.knmidata(fname, variable='RH')
EV24 = read.knmidata(fname, variable='EV24')
#rech = RH.series - EV24.series

# Create stress
#ts = Recharge(RH.series, EV24.series, Gamma, Preferential, name='recharge')
# ts = Recharge(RH.series, EV24.series, Gamma, Combination, name='recharge')
# ts = Tseries2(RH.series, EV24.series, Gamma, name='recharge')
ts = Tseries(RH.series, Gamma, name='precip', freq='D')
ts1 = Tseries(EV24.series, Gamma, name='evap', freq='D')
ml.add_tseries(ts)
ml.add_tseries(ts1)

# Add noise model
n = NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve()
ml.plot()

