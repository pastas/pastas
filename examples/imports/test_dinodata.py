# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:07:44 2016

@author: ruben
"""

import matplotlib.pyplot as plt
from gwtsa.read.dinodata import DinoGrondwaterstand

# %% hoe te gebruiken?
fname = '../data/B32D0136001_1.csv'
dino = DinoGrondwaterstand(fname)

# %% teken
plt.figure()
dino.stand.plot()
plt.show()
