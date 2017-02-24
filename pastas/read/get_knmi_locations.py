from pastas.read.knmidata import KnmiStation
from datetime import timedelta
from datetime import date
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import dill
import os

fname = 'get_knmi_locations.pkl'
if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        knmi = dill.load(f)
else:
    # download data van het afgelopen jaar
    knmi = KnmiStation(stns='ALL', start=date.today() - timedelta(days=365))
    knmi.download()
    with open(fname, 'wb') as f:
        dill.dump(knmi, f)

inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:28992')

fig = plt.figure()
ax = fig.add_subplot(111)
x=[]
y=[]
for index, station in knmi.stations.iterrows():
    xt, yt = transform(inProj, outProj, station['LON_east'], station['LAT_north'])
    x.append(xt)
    y.append(yt)

points, = plt.plot(x, y, 'o', picker=5)

def onpick(event):

    if event.artist!=points: return True

    N = len(event.ind)
    if not N: return True

    figi = plt.figure()
    for subplotnum, dataind in enumerate(event.ind):
        axi = figi.add_subplot(N,1,subplotnum+1)
        station = knmi.stations.iloc[dataind]
        erin=knmi.data['STN'] == int(station._name)
        if any(erin):
            knmi.data[erin]['RH'].plot(ax=axi, label="RH")
            knmi.data[erin]['EV24'].plot(ax=axi, label='EV24')
            plt.legend()
        axi.text(.5, .98, station.NAME,
                ha='center', va='top',
                transform=axi.transAxes)
    figi.show()
    return True

fig.canvas.mpl_connect('pick_event', onpick)

plt.axis('equal')
plt.grid()
plt.show()

