import pastas as pt

# load a model with precipitation, evaporation and a well
ml = pt.io.load('data/B28H1808_2.pas')

# first solve and plot to see the model-performance
ml.solve(noise=False)
ax = ml.plots.results()

# get the precipitation and evaporation
sm = ml.stressmodels['Recharge']
prec = sm.stress[0]
evap = sm.stress[1]

# delete all the stressmodels, the constant and the transform
ml.del_stressmodel('Recharge')
ml.del_stressmodel('Extraction')
ml.del_constant()

# then add a TarsoModel
sm = pt.TarsoModel(prec, evap, ml.oseries)
ml.add_stressmodel(sm)

# and solve and plot again
ml.solve(noise=False)
ax = ml.plots.results()
