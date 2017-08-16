"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.

"""
import pastas as ps

# read observations
obs = ps.read.dinodata('data/B58C0698001_1.csv')

# Create the time series model
ml = ps.Model(obs.series, metadata={"name":"Test"})

# read weather data
rain = ps.read.knmidata('data/neerslaggeg_HEIBLOEM-L_967-2.txt', variable='RD')
evap = ps.read.knmidata('data/etmgeg_380.txt', variable='EV24')

## Create stress
ts = ps.Tseries2(rain.series, evap.series, ps.Gamma, name='recharge')
ml.add_tseries(ts)

## Add noise model
n = ps.NoiseModel()
ml.add_noisemodel(n)

## Solve
ml.solve(noise=True, weights="swsi", freq="W", warmup=10000)
ml.plot()

#ml.export("test.pas")
# from pastas.io.iolib import import_model as imp
# ml = imp("test.pas")
# ml.plot()

