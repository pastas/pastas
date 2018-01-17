"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTAS during development.

"""
import pandas as pd
import pastas as ps

# Read observations
obs = ps.read_dino('data/B58C0698001_1.csv')
obs = obs.iloc[::5]
obs = obs[obs.index > pd.to_datetime('1-1-2010')]

# Create the time series model
ml = ps.Model(obs)

# Read weather data
prec = ps.read_knmi('data/neerslaggeg_HEIBLOEM-L_967-2.txt', variables='RD')
evap = ps.read_knmi('data/etmgeg_380.txt', variables='EV24')

# Create stress
if False:
    sm = ps.StressModel2(stress=[prec, evap], rfunc=ps.Exponential,
                         name='recharge')
    ml.add_stressmodel(sm)
elif False:
    sm = ps.StressModel(prec, rfunc=ps.Exponential, name='prec')
    ml.add_stressmodel(sm)
    sm = ps.StressModel(evap, rfunc=ps.Exponential, name='evap', up=False)
    ml.add_stressmodel(sm)
else:
    sm = ps.stressmodels.NoConvModel(prec, rfunc=ps.Exponential,
                                     name='prec_no_conv')
    ml.add_stressmodel(sm)
    sm = ps.stressmodels.NoConvModel(evap, rfunc=ps.Exponential,
                                     name='evap_no_conv', up=False)
    ml.add_stressmodel(sm)

# Solve and plot
ml.solve(noise=False, warmup=0)
# Plotting takes the longest
ml.plot()
