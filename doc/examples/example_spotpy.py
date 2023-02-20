"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""
import pandas as pd
import spotpy

import pastas as ps

ps.set_log_level("ERROR")

head = pd.read_csv(
    "data/B32C0639001.csv", parse_dates=["date"], index_col="date"
).squeeze()

# Make this millimeters per day
evap = pd.read_csv("data/evap_260.csv", index_col=0, parse_dates=[0]).squeeze()
rain = pd.read_csv("data/rain_260.csv", index_col=0, parse_dates=[0]).squeeze()

ml = ps.Model(head)

# Select a recharge model
rch = ps.rch.FlexModel()

rm = ps.RechargeModel(rain, evap, recharge=rch, rfunc=ps.Gamma(), name="rch")
ml.add_stressmodel(rm)

ml.solve(noise=False, tmin="1990")

# Now run with spotpy
s = ps.SpotpySolve(
    algorithm=spotpy.algorithms.mle,
    obj_func=spotpy.likelihoods.logLikelihood,
    parallel=True,
)
ml.solve(
    solver=s, initial=False, noise=False,
    repetitions=5000#, runs_after_convergence=1000
)
ml.plot()

posterior = spotpy.analyser.get_posterior(ml.fit.result, percentage=10)
spotpy.analyser.plot_parameterInteraction(posterior)

spotpy.analyser.plot_fast_sensitivity(ml.fit.result, number_of_sensitiv_pars=3)
