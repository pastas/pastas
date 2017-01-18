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
ts = Recharge(RH.series, EV24.series, Gamma, Linear(), name='recharge')
ml.add_tseries(ts)

# Add noise model
n = NoiseModel()
ml.add_noisemodel(n)

# Initialize model but don't solve
#ml.solve(initialize=True, solve=False)

# Solve model to get good starting value
ml.initialize()
import numpy as np

class PastasTest:
    def __init__(self, model):
        self.model = model
        self.dim = self.model.nparam
        self.xlow = []
        self.xup = []
        self.xvalue = []
        # TODO: this should be in the Model class of Pastas
        for ts in ml.tserieslist:
            self.xlow.extend(ts.parameters['pmin'].tolist())
            self.xup.extend(ts.parameters['pmax'].tolist())
            self.xvalue.extend(ts.parameters['value'].tolist())
        if ml.noisemodel is not None:
            self.xlow.extend(ml.noisemodel.parameters['pmin'].tolist())
            self.xup.extend(ml.noisemodel.parameters['pmax'].tolist())
            self.xvalue.extend(ml.noisemodel.parameters['value'].tolist())
        self.xlow = np.array(self.xlow)
        self.xup = np.array(self.xup)
        self.xvalue = np.array(self.xvalue)
        # option 2
        #faclow = 0.5 * np.ones(self.dim)
        #facup = 2 * np.ones(self.dim)
        #faclow[ml.parameters < 0] = 2
        #facup[ml.parameters < 0] = 0.5
        #self.xlow = faclow * ml.parameters
        #self.xup = facup * ml.parameters
        self.info = "Pastas test"
        self.integer = [] # integer variables MUST be specified
        self.continuous = np.arange(self.dim) # continuos variables
    
    def checkbounds(self, **kwargs):
        x = kwargs["x_new"]
        testmax = bool(np.all(x <= self.xup))
        testmin = bool(np.all(x >= self.xlow))
        #print 'x', x
        #print 'testmin', testmin
        #print 'testmax', testmax
        return testmax and testmin
        
    def objfunction(self, x):
        print('.')
        return self.model.sse(x, noise=True)
    
data = PastasTest(ml)
    
from scipy.optimize import differential_evolution
result = differential_evolution(data.objfunction, zip(data.xlow, data.xup))

#from scipy.optimize import least_squares
#result = least_squares(data.objfunction, data.xvalue, bounds=(data.xlow, data.xup))

#from scipy.optimize import basinhopping
#def print_fun(x, f, accepted):
#    print("at minimum %.4f accepted %d" % (f, int(accepted)))

#result = basinhopping(data.objfunction, data.xvalue, niter=50, callback=print_fun, accept_test=data.checkbounds)