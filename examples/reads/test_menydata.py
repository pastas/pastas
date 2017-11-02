"""

@author: ruben

"""

import matplotlib.pyplot as plt
from pastas.read import MenyData

# how to use it?
fname = '../data/MenyanthesTest.men'
meny = MenyData(fname)

# plot some series
f1, axarr = plt.subplots(len(meny.IN)+1, sharex=True)
meny.H['Obsevation well']['values'].plot(ax=axarr[0])
axarr[0].set_title('Obsevation well')
for i, name in enumerate(meny.IN):
    meny.IN[name]['values'].plot(ax=axarr[i+1])
    axarr[i+1].set_title(name)
plt.tight_layout()
plt.show()