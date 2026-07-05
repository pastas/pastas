# %%
import pastas as ps

# load a model with precipitation, evaporation and a well
ml = ps.io.load("data/B28H1808_2_pastas-0-23-0.pas")
if ml.transform is not None:
    ml.del_transform()
ml.del_noisemodel()

# first solve and plot to see the model-performance
ml.solve()
ax = ml.plots.results()

# get the precipitation and evaporation
sm = ml.stressmodels["Recharge"]
prec = sm.prec.series_original
evap = sm.evap.series_original

# delete all the stressmodels and the constant
ml.del_stressmodel("Recharge")
ml.del_stressmodel("Extraction")
ml.del_constant()

# then add a TarsoModel
sm = ps.TarsoModel(
    prec, evap, dmin=ml.oseries.series.min(), dmax=ml.oseries.series.max()
)
ml.add_stressmodel(sm)

# and solve and plot again
ml.solve()
ax = ml.plots.results()
