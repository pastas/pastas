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
    name = os.path.basename(nb)[:-6]
    pathname = '../doc/examples/{0}'.format(name)
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    fname = os.path.join(pathname,'{}.rst'.format(name))
    with io.open(fname, 'w', encoding='utf-8') as f:
        f.write(x[:])
    for output in resources['outputs'].keys():
        fname = os.path.join(pathname,output)
        with io.open(fname, 'wb') as f:
            f.write(resources['outputs'][output])

print("example notebooks successfully converted to .rst files")