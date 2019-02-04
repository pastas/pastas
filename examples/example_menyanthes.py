"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.
"""

import pastas as ps

fname = 'data/MenyanthesTest.men'
meny = ps.read.MenyData(fname)

# Create the time series model
H = meny.H['Obsevation well']
ml = ps.Model(H['values'])

# Add precipitation
IN = meny.IN['Precipitation']['values']
IN.index = IN.index.round("D")
IN.name = 'Precipitation'
IN2 = meny.IN['Evaporation']['values']
IN2.index = IN2.index.round("D")
IN2.name = 'Evaporation'
sm = ps.StressModel2([IN, IN2], ps.Gamma, 'Recharge')
ml.add_stressmodel(sm)

# Add well extraction 1
IN = meny.IN['Extraction 1']
well = ps.TimeSeries(IN["values"], freq_original="M", settings="well")
# extraction amount counts for the previous month
sm = ps.StressModel(well, ps.Hantush, 'Extraction_1', up=False)

# Add well extraction 2
IN = meny.IN['Extraction 2']
well = ps.TimeSeries(IN["values"], freq_original="M", settings="well")
# extraction amount counts for the previous month
sm1 = ps.StressModel(well, ps.Hantush, 'Extraction_2', up=False)

# Add well extraction 3
IN = meny.IN['Extraction 3']
well = ps.TimeSeries(IN["values"], freq_original="M", settings="well")
# extraction amount counts for the previous month
sm2 = ps.StressModel(well, ps.Hantush, 'Extraction_3', up=False)

# add_stressmodels also allows addings multiple stressmodels at once
ml.add_stressmodel(sm, sm1, sm2)

# Solve
ml.solve()

# make a decomposition-plot
ax = ml.plots.decomposition(ytick_base=1.)
