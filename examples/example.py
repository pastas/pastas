"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.

"""
import pastas as ps

# read observations
obs = ps.read_dino('data/B58C0698001_1.csv')

# Create the time series model
ml = ps.Model(obs)

# read weather data
rain = ps.read_knmi('data/neerslaggeg_HEIBLOEM-L_967-2.txt', variables='RD')
evap = ps.read_knmi('data/etmgeg_380.txt', variables='EV24')

## Create stress
sm = ps.StressModel2(stress=[rain,evap], rfunc=ps.Exponential,name='recharge')
ml.add_stressmodel(sm)

# Add noise model
n = ps.noisemodels.NoiseModel()
ml.add_noisemodel(n)

## Solve
ml.solve()
ml.plot()
