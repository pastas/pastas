"""
This test file is meant for developing purposes, providing an easy method to
test the functioning of tmin in the solve method. This code is mostly similar
to example_recharge.py.

Author: O.N. Ebbens, Artesia.

"""
import pastas as ps
import pandas as pd

ps.set_log_level("ERROR")

# read observations
obs = ps.read_dino('data/B58C0698001_1.csv')

# Create the time series model
ml = ps.Model(obs, name="groundwater head")

# read weather data
rain = ps.read_knmi('data/neerslaggeg_HEIBLOEM-L_967-2.txt', variables='RD')
rain.multiply(1000)
evap = ps.read_knmi('data/etmgeg_380.txt', variables='EV24')
evap.multiply(1000)

# Create stress
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential,
                      recharge="Linear", name='recharge')
ml.add_stressmodel(sm)

# Set tmin
tmin = pd.Timestamp('2010-1-1')
ml.settings['tmin'] = tmin

# Solve
ml.solve()

assert ml.settings['tmin']==tmin


ml.plot()
