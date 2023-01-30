import numpy as np
import pandas as pd

import pastas as ps

ps.set_log_level("ERROR")

# read observations and create the time series model
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True).squeeze("columns")
ml = ps.Model(obs, name="groundwater head")

# read weather data and create stressmodel
rain = pd.read_csv("data/rain_nb1.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)
evap = pd.read_csv("data/evap_nb1.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential(), name="recharge")
ml.add_stressmodel(sm)

# Solve
ml.solve()
#
df = ml.fit.prediction_interval()
inside = (obs > df.loc[obs.index, 0.025]) & (obs < df.loc[obs.index, 0.975])
print("percentage inside:", np.count_nonzero(inside) / len(inside) * 100)

# # Plot some results
axes = ml.plots.results(tmin="2010", tmax="2015", figsize=(10, 6))
axes[0].fill_between(
    df.index,
    df.iloc[:, 0],
    df.iloc[:, 1],
    color="gray",
    zorder=-1,
    alpha=0.5,
    label="95% Prediction interval",
)
axes[0].legend(ncol=3)
df = ml.fit.ci_contribution("recharge", tmin="2010", tmax="2015")
axes[2].fill_between(
    df.index,
    df.iloc[:, 0],
    df.iloc[:, 1],
    color="gray",
    zorder=-1,
    alpha=0.5,
    label="95% confidence",
)

df = ml.fit.ci_step_response("recharge", alpha=0.05, n=1000)
axes[3].fill_between(
    df.index,
    df.iloc[:, 0],
    df.iloc[:, 1],
    color="gray",
    zorder=-1,
    alpha=0.5,
    label="95% confidence",
)
