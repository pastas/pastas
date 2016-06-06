"""

@author: ruben

"""
from datetime import datetime
import matplotlib.pyplot as plt
from gwtsa.read.knmidata import KnmiStation

# How to use it?

if True:
    # use a file:

    knmi = KnmiStation.fromfile('../data/KNMI_20160504.txt')
else:
    # or download it from
    # https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
    knmi = KnmiStation(stns=260, start=datetime(1970, 1, 1),
                       end=datetime(1971, 1, 1))  # de bilt
    knmi.download()

# draw the figure
f, axarr = plt.subplots(2, sharex=True)
knmi.data['RH'].plot(ax=axarr[0])
axarr[0].set_title(knmi.variables['RH'])
knmi.data['EV24'].plot(ax=axarr[1])
axarr[1].set_title(knmi.variables['EV24'])
plt.show()
