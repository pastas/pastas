"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.

"""
import pastas as ps
import pandas as pd

# read observations
obs = ps.read_dino('data/B58C0698001_1.csv')

# Create the time series model
ml = ps.Model(obs)

# read weather data
rain = ps.read.KnmiStation.fromfile('data/neerslaggeg_HEIBLOEM-L_967-2.txt')
rain = ps.TimeSeries(rain.data.RD,settings='prec')

evap = ps.read_knmi('data/etmgeg_380.txt', variables='EV24')
evap = pd.Series(evap.values,evap.index+pd.to_timedelta(9,unit='h'))
evap = ps.TimeSeries(evap,settings='evap')

# Create stress
sm = ps.StressModel2(stress=[rain, evap], rfunc=ps.Exponential,
                     name='recharge')
ml.add_stressmodel(sm)

## Solve
ml.solve()
ml.plot()
