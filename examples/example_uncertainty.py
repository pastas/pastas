import matplotlib.pyplot as plt
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

df = ml.uncertainty.block_response("recharge", n=1000)
ax = ml.get_block_response("recharge").plot(color="C1")
df.plot(color="k", linestyle="--", ax=ax)
df = ml.uncertainty.block_response("recharge", n=1000, alpha=0.01)
df.plot(color="gray", linestyle="--", ax=ax)


df = ml.uncertainty.step_response("recharge", n=1000)
ax = ml.get_step_response("recharge").plot(color="C1")
df.plot(color="k", linestyle="--", ax=ax)
df = ml.uncertainty.step_response("recharge", n=1000, alpha=0.01)
df.plot(color="gray", linestyle="--", ax=ax)