"""
In this example a daily simulation is conducted from 9:00 until 9:00 (dutch standard time)
This is the time at which precipitation is logged in dutch KNMI-stations.

"""
import pastas as ps
import pandas as pd
import hydropandas as hpd

# read observations
obs = hpd.GroundwaterObs.from_dino('data/B58C0698001_1.csv')

# Create the time series model
ml = ps.Model(obs['stand_m_tov_nap'])

# read weather data
rain = hpd.PrecipitationObs.from_knmi(967, "precipitation", 
                                      startdate=obs.index[0],
                                      enddate=obs.index[-1])

evap = hpd.EvaporationObs.from_knmi(380, "EV24", 
                                    startdate=obs.index[0],
                                    enddate=obs.index[-1])

if True:
    # also add 8 hours to the evaporation
    evap.index = evap.index + pd.to_timedelta(8, 'h')

# Create stress
sm = ps.StressModel2(stress=[rain, evap], rfunc=ps.Exponential,
                     name='recharge')
ml.add_stressmodel(sm)

## Solve
ml.solve()
ml.plots.decomposition()