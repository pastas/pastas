import pandas as pd
import matplotlib.dates as md
import matplotlib.pyplot as plt
from gwtsa import *

# Import and check the observed groundwater time series
parse = lambda x: md.datetime.datetime.strptime(x, '%d-%m-%Y')
gwdata = pd.read_csv('B33A0113001_1.csv', index_col=2, skiprows=15,
                     usecols=[2, 5], parse_dates=True,
                     date_parser=parse)

gwdata.rename(columns={'Stand (cm t.o.v. NAP)': 'h'}, inplace=True)
gwdata.index.names = ['date']
gwdata.h *= 0.01  # In meters
oseries = gwdata.h
#
data = pd.read_csv('KNMI_Apeldoorn.txt', skipinitialspace=True, skiprows=12,
                   delimiter=',', parse_dates=[0], index_col=['Date'], usecols=[
        1, 2, 3], names=['Date', 'P', 'E'])
data = data['1960':]  # cut off everything before 1958
data /= 10000.  # In meters

# Create the time series model with preferential recharge model
ml = Model(oseries)
ts = Tseries3([data.P, data.E], Gamma(), Preferential(), name='recharge')
ml.addtseries(ts)
d = Constant(oseries.min())  # Using oseries.min() as initial value slightly
# reduces computation time
ml.addtseries(d)
n = NoiseModel()
ml.addnoisemodel(n)

# Solve for a certain period
ml.solve(tmin='1970', tmax='2004')

# Create the time series model with linear recharge model
ml1 = Model(oseries)
ts1 = Tseries3([data.P, data.E], Gamma(), Linear(), name='recharge')
ml1.addtseries(ts1)
d1 = Constant(oseries.min())  # Using oseries.min() as initial value slightly
# reduces computation time
ml1.addtseries(d)
n1 = NoiseModel()
ml1.addnoisemodel(n1)

# Solve for a certain period
ml1.solve(tmin='1970', tmax='2004')

plt.figure()
plt.plot(ml1.oseries, 'ko', markersize=2)
plt.plot(ml.simulate())
plt.plot(ml1.simulate())
plt.legend(['observed', 'preferential', 'linear'], loc='best')
