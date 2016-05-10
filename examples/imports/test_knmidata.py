# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:58:15 2016

@author: ruben
"""
from datetime import datetime
import matplotlib.pyplot as plt
from knmidata import KnmiStation

#%% hoe te gebruiken?
if True:
    # via een bestand, te downloaden via https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
    knmi=KnmiStation.fromfile('KNMI_20160504.txt')
else:
    # of door direct te downloaden
    knmi=KnmiStation(stns=260, start=datetime(1970,1,1),end=datetime(1971,1,1)) # de bilt
    knmi.download()

#%% teken
f, axarr = plt.subplots(2, sharex=True)
knmi.data['RH'].plot(ax=axarr[0])
axarr[0].set_title(knmi.variables['RH'])
knmi.data['EV24'].plot(ax=axarr[1])
axarr[1].set_title(knmi.variables['EV24'])