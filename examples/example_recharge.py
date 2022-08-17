"""
This test file is meant for developing purposes, providing an easy method to
test the functioning of Pastas recharge module during development.

Author: R.A. Collenteur, University of Graz.

"""
import pandas as pd
import hydropandas as hpd

import pastas as ps

ps.set_log_level("WARNING")

# read observations and create the time series model and make meters
obs = pd.read_csv("data/B32C0639001.csv", parse_dates=['date'],
                  index_col='date').squeeze("columns")

# Create the time series model
ml = ps.Model(obs, name="head")

# read weather data and make mm/d !
rain = hpd.PrecipitationObs.from_knmi(260, "meteo", 
                                      startdate=obs.index[0],
                                      enddate=obs.index[-1]) * 1e3

evap = hpd.EvaporationObs.from_knmi(260, "EV24", 
                                    startdate=obs.index[0],
                                    enddate=obs.index[-1]) * 1e3

# Initialize recharge model and create stressmodel
rch = ps.rch.FlexModel(interception=True)
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential, recharge=rch,
                      name="rch")

ml.add_stressmodel(sm)

ml.solve(noise=True, tmin="1990")

ml.plots.results()

df = ml.stressmodels["rch"].get_water_balance(ml.get_parameters("rch"))
df.plot(subplots=True)

