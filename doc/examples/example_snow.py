import pandas as pd

import pastas as ps

ps.set_log_level("WARNING")

head = pd.read_csv("data/heby_head.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)
evap = pd.read_csv("data/heby_evap.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)
prec = pd.read_csv("data/heby_prec.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)
temp = pd.read_csv("data/heby_temp.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)

tmin = "1985"  # Needs warmup
tmax = "2018"

ml = ps.Model(head)
sm = ps.RechargeModel(
    prec,
    evap,
    recharge=ps.rch.FlexModel(snow=True),
    rfunc=ps.Gamma(),
    name="rch",
    temp=temp,
)
ml.add_stressmodel(sm)

# In case of the non-linear model, change some parameter settings
ml.set_parameter("rch_kv", vary=False)

# Solve the Pastas model
ml.solve(tmin=tmin, tmax=tmax, noise=False, fit_constant=False, report=False)
ml.set_parameter("rch_ks", vary=False)
ml.solve(tmin=tmin, tmax=tmax, noise=True, fit_constant=False, initial=False)

ml.plots.results()

df = ml.stressmodels["rch"].get_water_balance(ml.get_parameters("rch"))
df.plot(subplots=True, figsize=(20, 10))
