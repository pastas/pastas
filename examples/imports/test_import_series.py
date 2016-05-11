# -*- coding: utf-8 -*-
"""
Created on Wed May 11 20:06:41 2016

@author: ruben
"""
from gwtsa.imports.import_series import ImportSeries
import matplotlib.pyplot as plt

fname = 'B32D0136001_1.csv'
series=ImportSeries(fname,'dino')

plt.figure()
series.series.plot()
plt.show()

fname = 'KNMI_20160504.txt'
series=ImportSeries(fname,'knmi',variable='RH')

plt.figure()
series.series.plot()
plt.show()
