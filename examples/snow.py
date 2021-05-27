import pandas as pd
import pastas as ps
import matplotlib.pyplot as plt

ps.set_log_level("WARNING")

# head = pd.read_csv("notebooks/data_wagna/head_wagna.csv", index_col=0,
#                    parse_dates=True, squeeze=True, skiprows=2).loc["2006":]
# evap = pd.read_csv("notebooks/data_wagna/evap_wagna.csv", index_col=0,
#                    parse_dates=True, squeeze=True, skiprows=2)
# rain = pd.read_csv("notebooks/data_wagna/rain_wagna.csv", index_col=0,
#                    parse_dates=True, squeeze=True, skiprows=2)
# temp = pd.read_csv("notebooks/data_wagna/temp.csv", index_col=0,
#                    parse_dates=True, squeeze=True, skiprows=2)

head = pd.read_csv("data/heby/head.csv", index_col=0, usecols=["Date", "3"],
                   parse_dates=True, squeeze=True)
evap = pd.read_csv("data/heby/evap.csv", index_col=0,
                   parse_dates=True, squeeze=True, usecols=["time", "3"],)
rain = pd.read_csv("data/heby/rain.csv", index_col=0,
                   parse_dates=True, squeeze=True, usecols=["time", "3"],)
temp = pd.read_csv("data/heby/TG_STAID000426.txt", index_col=0,
                   usecols=["DATE", "TG"], skipinitialspace=True,
                   parse_dates=True, squeeze=True, skiprows=19) / 10

#temp = temp.asfreq("D").fillna(0)
tmin = pd.Timestamp("1990-01-01")  # Needs warmup
tmax = pd.Timestamp("2010-12-31")

ml = ps.Model(head)
sm = ps.RechargeModel(rain, evap, recharge=ps.rch.FlexSnowModel(snow=True),
                      rfunc=ps.Exponential, name="rch", temp=temp)
ml.add_stressmodel(sm)

# In case of the non-linear model, change some parameter settings
ml.set_parameter("rch_srmax", vary=False)
ml.set_parameter("rch_k", vary=True)

#ml.set_parameter("constant_d", vary=True, initial=262, pmax=head.min())

# Solve the Pastas model
ml.solve(tmin=tmin, tmax=tmax, noise=False, report="basic")

ml.plots.results()

df = ml.stressmodels["rch"].get_water_balance(ml.get_parameters("rch"))
df.plot(subplots=True, figsize=(20, 10))
