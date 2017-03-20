"""
@author: ruben

"""

from pastas import read
import matplotlib.pyplot as plt

fname = '../data/B32D0136001_1.csv'
obs = read.dinodata(fname)

fname = '../data/KNMI_Bilt.txt'
stress = read.knmidata(fname, 'EV24')

obs.series.plot()
stress.series.plot()

