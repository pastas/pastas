"""

@author: ruben

"""

import matplotlib.pyplot as plt
from pasta.read.menydata import MenyData

# how to use it?
fname = '../data/MenyanthesTest.men'
meny = MenyData(fname)

# plot some series
f1, axarr = plt.subplots(2, sharex=True)
meny.H[0].series.plot(ax=axarr[0])
axarr[0].set_title(meny.H[0].name)
meny.IN[0].series.plot(ax=axarr[1])
axarr[1].set_title(meny.IN[0].name)
plt.show()