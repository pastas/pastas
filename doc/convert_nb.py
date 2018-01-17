""" This script is called by sphinxdoc to convert all Jupyter notebooks in the
Examples folder to ReStructured Text files and stores them in the documentation
folder.

Author: R.A. Collenteur 2016
"""

import nbconvert
from nbconvert import RSTExporter
import glob
import io
import os

notebooks = glob.iglob('../examples/notebooks/*.ipynb')
for nb in notebooks:
    (x,resources) = nbconvert.export(RSTExporter, nb)
    with io.open('../doc/examples/%s.rst' % os.path.basename(nb)[:-6], 'w',
                 encoding='utf-8') as f:
        f.write(x[:])
    for fname in resources['outputs'].keys():
        with io.open('../doc/examples/%s' % fname, 'wb') as f:
            f.write(resources['outputs'][fname])
        

print("example notebooks successfully converted to .rst files")