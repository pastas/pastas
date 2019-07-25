"""
This is an example with a step in the observations, which we add artificially.
We model this step through a StepModel.

"""
import pastas as ps

# read observations
obs = ps.read_dino('data/B58C0698001_1.csv')

# add 10 cm to the series from 2007
s = obs.series_original
s['2007':] = s['2007':]+0.1
obs.series_original = s

# Create the time series model
ml = ps.Model(obs)

# read weather data
rain = ps.read_knmi('data/neerslaggeg_HEIBLOEM-L_967-2.txt', variables='RD')
evap = ps.read_knmi('data/etmgeg_380.txt', variables='EV24')

# create stress
sm = ps.StressModel2(stress=[rain, evap], rfunc=ps.Exponential,
                     name='recharge')
ml.add_stressmodel(sm)

# add a stepmodel with an exponential response
sm = ps.stressmodels.StepModel('2007','Step',rfunc=ps.Exponential)
ml.add_stressmodel(sm)

# solve
ml.solve()
ml.plots.decomposition()
