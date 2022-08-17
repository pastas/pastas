"""

@author: ruben

"""
import matplotlib.pyplot as plt
from pastas.read.knmi import KnmiStation
import numpy as np

# How to use it?
# data from a meteorological station
download = True
meteo = True
hourly = False
if hourly and not meteo:
    raise(ValueError('Hourly data is only available in meteorological stations'))

if download:
    # download the data directly from the site of the KNMI
    if meteo:
        if hourly:
            knmi = KnmiStation.download(stns=260, start='2017', end='2018',
                                interval='hourly')  # de bilt
        else:
            # https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
            knmi = KnmiStation.download(stns=260, start='1970',end='1971')
            # de bilt
    else:
        # from a rainfall-station
        knmi = KnmiStation.download(stns=550, start='2018', end='2019',
                                vars='RD') # de bilt
else:
    # import the data from files
    if meteo:
        if hourly:
            # hourly data without locations
            knmi = KnmiStation.fromfile('../data/KNMI_Hourly.txt')
        else:
            # without locations, that was downloaded from the knmi-site
            knmi = KnmiStation.fromfile('../data/KNMI_NoLocation.txt')

            # use a file with locations:
            #knmi = KnmiStation.fromfile('../data/KNMI_Bilt.txt')
    else:
        knmi = KnmiStation.fromfile('../data/KNMI_Akkrum.txt')

# plot
f1, axarr = plt.subplots(2, sharex=True)
if 'RD' in knmi.data.columns and not np.all(np.isnan(knmi.data['RD'])):
    knmi.data['RD'].plot(ax=axarr[0])
    axarr[0].set_title(knmi.variables['RD'])
if 'RH' in knmi.data.columns and not np.all(np.isnan(knmi.data['RH'])):
    knmi.data['RH'].plot(ax=axarr[0])
    axarr[0].set_title(knmi.variables['RH'])
if 'EV24' in knmi.data.columns and not np.all(np.isnan(knmi.data['EV24'])):
    knmi.data['EV24'].plot(ax=axarr[1])
    axarr[1].set_title(knmi.variables['EV24'])

plt.show()
