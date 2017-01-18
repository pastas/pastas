"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""
#import matplotlib
#matplotlib.use('TkAgg')
from pastas import *

# read observations
fname = 'data/B32D0136001_1.csv'
obs = ReadSeries(fname,'dino')

# Create the time series model
ml = Model(obs.series)

# read climate data
fname = 'data/KNMI_Bilt.txt'
RH=ReadSeries(fname,'knmi',variable='RH')
EV24=ReadSeries(fname,'knmi',variable='EV24')

# Create stress
ts = Recharge(RH.series, EV24.series, Gamma(), Linear(), name='recharge')
ml.add_tseries(ts)

# Add noise model
n = NoiseModel()
ml.add_noisemodel(n)

# Initialize model but don't solve
#ml.solve(initialize=True, solve=False)

# Solve model to get good starting value
ml.solve()


class PastasTest:
    def __init__(self, model):
        self.model = model
        self.dim = self.model.nparam
        self.xlow = []
        self.xup = []
        # TODO: this should be in the Model class of Pastas
        for ts in ml.tserieslist:
            self.xlow.extend(ts.parameters['pmin'].tolist())
            self.xup.extend(ts.parameters['pmax'].tolist())
        if ml.noisemodel is not None:
            self.xlow.extend(ml.noisemodel.parameters['pmin'].tolist())
            self.xup.extend(ml.noisemodel.parameters['pmax'].tolist())
        self.xlow = np.array(self.xlow)
        self.xup = np.array(self.xup)
        #
        #faclow = 0.5 * np.ones(self.dim)
        #facup = 2 * np.ones(self.dim)
        #faclow[ml.parameters < 0] = 2
        #facup[ml.parameters < 0] = 0.5
        #self.xlow = faclow * ml.parameters
        #self.xup = facup * ml.parameters
        self.info = "Pastas test"
        self.integer = [] # integer variables MUST be specified
        self.continuous = np.arange(self.dim) # continuos variables
        
    def objfunction(self, x):
        print('.')
        return self.model.sse(x, noise=True)
    
# Import the necessary modules
from pySOT import *
from poap.controller import SerialController, BasicWorkerThread
import numpy as np

# Decide how many evaluations we are allowed to use
maxeval = 800

# (1) Optimization problem
data = PastasTest(ml)

# (2) Experimental design
# Use a symmetric Latin hypercube with 2d + 1 samples
exp_des = SymmetricLatinHypercube(dim=data.dim, npts=2 * data.dim + 1)

# (3) Surrogate model
# Use a cubic RBF interpolant, with the domain scaled to the unit box
surrogate = RSUnitbox(RBFInterpolant(surftype=CubicRBFSurface, maxp=maxeval), data)

# (4) Adaptive sampling
# Use DYCORS with 100d candidate points
adapt_samp = CandidateDYCORS(data=data, numcand=100 * data.dim)

# Use the serial controller (uses only one thread)
controller = SerialController(data.objfunction)

# (5) Use the sychronous strategy without non-bound constraints
strategy = SyncStrategyNoConstraints(
        worker_id=0, data=data, maxeval=maxeval, nsamples=1,
        exp_design=exp_des, response_surface=surrogate, 
        sampling_method=adapt_samp)
controller.strategy = strategy

# Run the optimization strategy
result = controller.run()

# Print the final result
print('Best value found: {0}'.format(result.value))
print('Best solution found: {0}'.format(
    np.array_str(result.params[0], max_line_width=np.inf,
                precision=5, suppress_small=True)))