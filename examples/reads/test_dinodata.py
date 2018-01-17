# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:07:44 2016

@author: ruben
"""

import matplotlib.pyplot as plt
import pastas as ps

# # How to use it?
fname = '../data/B32D0136001_1.csv'
dino = ps.read_dino(fname)

# plot
dino.plot()
plt.show()
