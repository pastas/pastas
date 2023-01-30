# %%
import pastas as pt

# load a model with precipitation, evaporation and a well
ml = pt.io.load("data/B28H1808_2_pastas-0-22-0.pas")

# first solve and plot to see the model-performance
ml.solve(noise=False)
ax = ml.plots.results()

# get the precipitation and evaporation
sm = ml.stressmodels["Recharge"]
prec = sm.stress[0].series_original
evap = sm.stress[1].series_original

# delete all the stressmodels and the constant
ml.del_stressmodel("Recharge")
ml.del_stressmodel("Extraction")
ml.del_constant()

# then add a TarsoModel
sm = pt.TarsoModel(
    prec, evap, dmin=ml.oseries.series.min(), dmax=ml.oseries.series.max()
)
ml.add_stressmodel(sm)

# and solve and plot again
ml.solve(noise=False)
ax = ml.plots.results()
