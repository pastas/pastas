import pandas as pd

import pastas as ps

ps.set_log_level("ERROR")

# read observations and create the time series model
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True,
                  squeeze=True)
ml = ps.Model(obs, name="groundwater head")

# read weather data and create stressmodel
rain = pd.read_csv("data/rain_nb1.csv", index_col=0, parse_dates=True,
                   squeeze=True)
evap = pd.read_csv("data/evap_nb1.csv", index_col=0, parse_dates=True,
                   squeeze=True)
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential,
                      recharge="Linear", name='recharge')
ml.add_stressmodel(sm)

# Solve
ml.solve()

# Plot some results
axes = ml.plots.results()

df = ml.uncertainty.prediction_interval()
axes[0].fill_between(df.index, df.iloc[:, 0], df.iloc[:, 1], color="gray",
                     zorder=-1, alpha=0.5, label="95% Prediction interval")
axes[0].legend()
df = ml.uncertainty.contribution("recharge")
axes[3].fill_between(df.index, df.iloc[:, 0], df.iloc[:, 1], color="gray",
                     zorder=-1, alpha=0.5, label="95% confidence")

df = ml.uncertainty.step_response("recharge", alpha=0.05, n=1000)
axes[4].fill_between(df.index, df.iloc[:, 0], df.iloc[:, 1], color="gray",
                     zorder=-1, alpha=0.5, label="95% confidence")

axes[0].set_xlim(["2010", "2015"])
