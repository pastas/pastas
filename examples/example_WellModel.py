# In this example the MultiWell StressModel is showcased and tested.
# R.A. Collenteur - Artesia Water 2018

import numpy as np
import pastas as ps
from pastas.stressmodels import WellModel

fname = 'data/MenyanthesTest.men'
meny = ps.read.MenyData(fname)

# Create the time series model
H = meny.H['Obsevation well']
ml = ps.Model(H['values'])

# Add precipitation
IN = meny.IN['Precipitation']['values']
IN.index = IN.index.round("D")
IN2 = meny.IN['Evaporation']['values']
IN2.index = IN2.index.round("D")
sm = ps.StressModel2([IN, IN2], ps.Gamma, 'Recharge')
ml.add_stressmodel(sm)

stresses = [meny.IN['Extraction 1']["values"],
            meny.IN['Extraction 2']["values"],
            meny.IN['Extraction 3']["values"]]

# Get distances from metadata
xo = meny.H["Obsevation well"]['xcoord']
yo = meny.H["Obsevation well"]['ycoord']
distances = []
for extr in ['Extraction 1', 'Extraction 2', 'Extraction 3']:
    xw = meny.IN[extr]["xcoord"]
    yw = meny.IN[extr]["ycoord"]
    distances.append(np.sqrt((xo-xw)**2 + (yo-yw)**2))

w = WellModel(stresses, ps.Hantush, distances=distances, name="Wells")

ml.add_stressmodel(w)
ml.solve()
ml.plots.decomposition()
