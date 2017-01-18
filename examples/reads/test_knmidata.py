"""

@author: ruben

"""
from datetime import datetime
import matplotlib.pyplot as plt
from pastas.read.knmidata import KnmiStation

# How to use it?
# data from a meteorological station
download = False

if not download:
    # use a file with locations:
    knmi = KnmiStation.fromfile('../data/KNMI_Bilt.txt')

    # without locations
    knmi = KnmiStation.fromfile('../data/KNMI_NoLocation.txt')

    # hourly data without locations
    knmi = KnmiStation.fromfile('../data/KNMI_Hourly.txt')
else:
    # or download it directly from
    # https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
    knmi = KnmiStation(stns=260, start=datetime(1970, 1, 1),
                       end=datetime(1971, 1, 1))  # de bilt
    knmi.download() # for now only works for meteorological stations
                    # and daily data (so not hourly)

# plot
f1, axarr = plt.subplots(2, sharex=True)
knmi.data['RH'].plot(ax=axarr[0])
axarr[0].set_title(knmi.variables['RH'])
if 'EV24' in knmi.data.keys():
    knmi.data['EV24'].plot(ax=axarr[1])
    axarr[1].set_title(knmi.variables['EV24'])

if True:
    # use a file of a rainfall station:
    knmi = KnmiStation.fromfile('../data/KNMI_Akkrum.txt')
    # plot
    f2 = plt.figure()
    ax = f2.add_subplot(111)
    knmi.data['RD'].plot(ax=ax)
    ax.set_title(knmi.variables['RD'], fontsize=10)

plt.show()

