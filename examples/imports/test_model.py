# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:19:46 2016

@author: ruben
"""

import gwtsa
from gwtsa.imports.import_series import ImportSeries

# read observations
fname = 'B32D0136001_1.csv'
obs = ImportSeries(fname,'dino')

# Create the time series model
ml = gwtsa.Model(obs.series)

# read climate data
fname = 'KNMI_20160522.txt'
RH=ImportSeries(fname,'knmi',variable='RH')
EV24=ImportSeries(fname,'knmi',variable='EV24')

# Create stress
ts = gwtsa.Tseries2([RH.series, EV24.series], gwtsa.Gamma(), name='recharge')
ml.addtseries(ts)

# Add drainage level
d = gwtsa.Constant(obs.series.min())
ml.addtseries(d)

# Add noise model
n = gwtsa.NoiseModel()
ml.addnoisemodel(n)

# Solve the time series model
ml.solve()

# show results
ml.plot()