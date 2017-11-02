"""
@author: ruben

"""

import pastas as ps

fname = '../data/B32D0136001_1.csv'
obs = ps.read_dino(fname)

fname = '../data/KNMI_Bilt.txt'
stress = ps.read_knmi(fname, 'EV24')

obs.plot()
stress.plot()

