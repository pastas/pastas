"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.

"""
#import matplotlib
#matplotlib.use('TkAgg')
from pasta import *

# read observations
fname = 'data/B32D0136001_1.csv'
obs = ReadSeries(fname,'dino')

# Create the time series model
ml = Model(obs.series)

# read climate data
fname = 'data/KNMI_20160522.txt'
RH=ReadSeries(fname,'knmi',variable='RH')
EV24=ReadSeries(fname,'knmi',variable='EV24')
rech = RH.series - EV24.series

# Create stress
#ts = Recharge(RH.series, EV24.series, Gamma, Linear, name='recharge')
#ts = Recharge(RH.series, EV24.series, Gamma, Combination, name='recharge')
ts = Tseries2(RH.series, EV24.series, Gamma, name='recharge')
#ts = Tseries(rech, Gamma, name='recharge')
ml.add_tseries(ts)

# Add drainage level
d = Constant(value=obs.series.min())
ml.add_tseries(d)

# Add noise model
n = NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve()
ml.plot()

#ml.initialize()
#print ml.parameters

# Plot
#ml.plot()