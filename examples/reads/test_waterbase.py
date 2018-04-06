"""

@author: ruben

"""

import pastas as ps

fname = '../data/20180405_010.csv'
wb = ps.read_waterbase(fname)

wb.plot()