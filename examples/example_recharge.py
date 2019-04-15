"""
This test file is meant for developing purposes, providing an easy method to
test the functioning of Pastas recharge module during development.

Author: R.A. Collenteur, University of Graz.

"""
import pastas as ps

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

## Solve
ml.solve()
ml.plot()
