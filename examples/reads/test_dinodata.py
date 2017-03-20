# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:07:44 2016

@author: ruben
"""

import matplotlib.pyplot as plt
from pastas import read

# # How to use it?
fname = '../data/B32D0136001_1.csv'
dino = read.dinodata(fname)

# plot
dino.series.plot()
plt.show()
