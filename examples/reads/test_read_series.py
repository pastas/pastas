"""
@author: ruben

"""

from pastas import *
import matplotlib.pyplot as plt

fname = '../data/B32D0136001_1.csv'
obs = ReadSeries(fname, 'dino')

fname = '../data/KNMI_Bilt.txt'
stress = ReadSeries(fname, 'knmi', variable='RH')

plt.figure()
plt.subplot(211)
obs.series.plot()
plt.subplot(212)
stress.series.plot()
plt.show()
