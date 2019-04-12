""" This script is called by sphinxdoc, to add a link to all Jupyter notebooks
in the Examples folder, so they are converted to HTML for the documentation.
"""

import os

from_path = '../examples/notebooks'
to_path = 'examples'
# first delete existing links in examples directory
for file in os.listdir(to_path):
    if file.endswith('.ipynb.nblink'):
        os.remove(os.path.join(to_path,file))
# then add new linkt to jupyter notebooks
for file in os.listdir(from_path):
    if file.endswith('.ipynb'):
        fname = os.path.join(to_path,'{}.nblink'.format(file))
        with open(fname,'w') as f:
            f.write('{{\n    "path": "../{}/{}"\n}}'.format(from_path,file))
print("Links successfully placed to jupyter notebooks")