"""

@author: ruben

"""

import matplotlib.pyplot as plt
from pastas import read
import scipy.io as sio

# how to use it?
fname = '../data/MenyanthesTest.men'
mat = sio.loadmat(fname, struct_as_record=False, squeeze_me=True,
                  chars_as_strings=True)
meny = read.menydata(fname, 'IN')



# plot some series
f1, axarr = plt.subplots(len(meny.IN)+1, sharex=True)
meny.H[0].series.plot(ax=axarr[0])
axarr[0].set_title(meny.H[0].name)
for i in range(0,len(meny.IN)):
    meny.IN[i].series.plot(ax=axarr[i+1])
    axarr[i+1].set_title(meny.IN[i].name)
plt.tight_layout()
plt.show()