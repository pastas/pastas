"""

@author: ruben

"""
from datetime import datetime
import matplotlib.pyplot as plt
from pasta.read.knmidata import KnmiStation

# How to use it?
# data from a meteorological station
if True:
    # use a file
    # with locations:
    knmi = KnmiStation.fromfile('../data/KNMI_20160504.txt')

    # without locations
    knmi = KnmiStation.fromfile('../data/etmgeg_324.txt')

    # hourly data without locations
    knmi = KnmiStation.fromfile('../data/uurgeg_240_2011-2020.txt')
else:
    # or download it directly from
    # https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
    knmi = KnmiStation(stns=260, start=datetime(1970, 1, 1),
                       end=datetime(1971, 1, 1))  # de bilt
    knmi.download() # for now only works for meteorological stataions and daily data (so not hourly)

# plot
f1, axarr = plt.subplots(2, sharex=True)
knmi.data['RH'].plot(ax=axarr[0])
axarr[0].set_title(knmi.variables['RH'])
if 'EV24' in knmi.data.keys():
    knmi.data['EV24'].plot(ax=axarr[1])
    axarr[1].set_title(knmi.variables['EV24'])

if True:
    # use a file of a rainfall station:
    knmi = KnmiStation.fromfile('../data/neerslaggeg_AKKRUM_089.txt')
    # plot
    f2 = plt.figure()
    ax = f2.add_subplot(111)
    knmi.data['RD'].plot(ax=ax)
    ax.set_title(knmi.variables['RD'], fontsize=10)

plt.show()

