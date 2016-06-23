"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.

"""
#import matplotlib
#matplotlib.use('TkAgg')
from pasta import *

# read observations
fname = 'data/B32D0136001_1.csv'
obs = ReadSeries(fname,'dino')

# Create the time series model
ml = Model(obs.series)

# read climate data
fname = 'data/KNMI_20160522.txt'
RH=ReadSeries(fname,'knmi',variable='RH')
EV24=ReadSeries(fname,'knmi',variable='EV24')

# Create stress
ts = Recharge(RH.series, EV24.series, Gamma(), Linear(), name='recharge')
ml.addtseries(ts)

# Add drainage level
d = Constant(value=obs.series.mean(), pmin=obs.series.mean() - 5, pmax=obs.series.mean() + 5)
ml.addtseries(d)

# Add noise model
n = NoiseModel()
ml.addnoisemodel(n)

# Initialize model but don't solve
#ml.solve(initialize=True, solve=False)

# Solve model to get good starting value
ml.solve()

import numpy as np

class PastaTest:
    def __init__(self, model):
        self.model = model
        self.dim = self.model.nparam
        self.xlow = []
        self.xup = []
        # TODO: this should be in the Model class of Pasta
        for ts in ml.tserieslist:
            self.xlow.extend(ts.parameters['pmin'].tolist())
            self.xup.extend(ts.parameters['pmax'].tolist())
        if ml.noisemodel is not None:
            self.xlow.extend(ml.noisemodel.parameters['pmin'].tolist())
            self.xup.extend(ml.noisemodel.parameters['pmax'].tolist())
        self.xlow = np.array(self.xlow)
        self.xup = np.array(self.xup)
        # option 2
        #faclow = 0.5 * np.ones(self.dim)
        #facup = 2 * np.ones(self.dim)
        #faclow[ml.parameters < 0] = 2
        #facup[ml.parameters < 0] = 0.5
        #self.xlow = faclow * ml.parameters
        #self.xup = facup * ml.parameters
        self.info = "Pasta test"
        self.integer = [] # integer variables MUST be specified
        self.continuous = np.arange(self.dim) # continuos variables
        
    def objfunction(self, x):
        print '.',
        return self.model.sse(x, noise=True)
    
data = PastaTest(ml)
    
from scipy.optimize import differential_evolution
result = differential_evolution(data.objfunction, zip(data.xlow, data.xup))