"""
In this example a daily simulation is conducted from 9:00 until 9:00 (dutch standard time)
This is the time at which precipitation is logged in dutch KNMI-stations.

"""
import pastas as ps
import pandas as pd

# read observations
obs = ps.read_dino('data/B58C0698001_1.csv')

# Create the time series model
ml = ps.Model(obs)

# read weather data
knmi = ps.read.knmi.KnmiStation.fromfile(
    'data/neerslaggeg_HEIBLOEM-L_967-2.txt')
rain = ps.TimeSeries(knmi.data['RD'], settings='prec')

evap = ps.read_knmi('data/etmgeg_380.txt', variables='EV24')
if True:
    # also add 9 hours to the evaporation
    s = evap.series_original
    s.index = s.index + pd.to_timedelta(9, 'h')
    evap.series_original = s

# Create stress
sm = ps.RechargeModel(rain, evap, rfunc=ps.Exponential,
                      recharge=ps.rch.Linear(), name='recharge')
ml.add_stressmodel(sm)

## Solve
ml.solve()
ml.plots.decomposition()
