"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.

"""
import pastas as ps

# read observations
fname = 'data/B32D0136001_1.csv'
obs = ps.read_dino(fname)

# Create the time series model
ml = ps.Model(obs)

# read climate data
fname = 'data/KNMI_Bilt.txt'
RH = ps.read_knmi(fname, variables='RH')
EV24 = ps.read_knmi(fname, variables='EV24')
#rech = RH.series - EV24.series

# Create stress
#sm = ps.Recharge(RH, EV24, ps.Gamma, ps.Linear, name='recharge')
#sm = Recharge(RH, EV24, Gamma, Combination, name='recharge')
sm = ps.StressModel2([RH, EV24], ps.Gamma, name='recharge')
#sm = ps.StressModel(RH, ps.Gamma, name='precip')
#sm1 = ps.StressModel(EV24, ps.Gamma, name='evap')
ml.add_stressmodel(sm)
#ml.add_tseries(sm1)

# Add noise model
n = ps.NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve(freq="W")
ml.plot()

