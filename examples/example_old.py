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
#ts = ps.Recharge(RH.series, EV24.series, ps.Gamma, ps.Linear, name='recharge')
#ts = Recharge(RH.series, EV24.series, Gamma, Combination, name='recharge')
ts = ps.StressModel2([RH.series, EV24.series], ps.Gamma, name='recharge')
#ts = ps.StressModel(RH.series, ps.Gamma, name='precip')
#ts1 = ps.StressModel(EV24.series, ps.Gamma, name='evap')
ml.add_stressmodel(ts)
#ml.add_tseries(ts1)

# Add noise model
n = ps.NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve(freq="W")
ml.plot()

