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
ml.solve(initialize=True, solve=False)


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
        self.info = "Pasta test"
        #self.integer = np.array([0]) # integer variables
        self.continuous = np.arange(self.dim) # continuos variables
        
    def objfunction(self, x):
        return self.model.sse(x)
    
# Import the necessary modules
from pySOT import *
from poap.controller import SerialController, BasicWorkerThread
import numpy as np

# Decide how many evaluations we are allowed to use
maxeval = 500

# (1) Optimization problem
data = PastaTest(ml)

# (2) Experimental design
# Use a symmetric Latin hypercube with 2d + 1 samples
exp_des = SymmetricLatinHypercube(dim=data.dim, npts=2 * data.dim + 1)

# (3) Surrogate model
# Use a cubic RBF interpolant, with the domain scaled to the unit box
surrogate = RSUnitbox(RBFInterpolant(surftype=CubicRBFSurface, maxp=maxeval), data)

# (4) Adaptive sampling
# Use DYCORS with 100d candidate points
adapt_samp = CandidateDYCORS(data=data, numcand=100 * data.dim)