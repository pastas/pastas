"""
Testing the Jupyter notebooks on TravisCI

adapted from https://github.com/ghego/travis_anaconda_jupyter

R.A. Collenteur, June, 2019

"""

import subprocess
import tempfile
import os

nbdir = os.path.join("examples", "notebooks")


def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        pth  = os.path.join(nbdir, )
        args = ["jupyter", "nbconvert", "--to", "notebook",
                "--execute", "{}".format(pth),
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
        subprocess.check_call(args)


def test():
    _exec_notebook('example.ipynb')


def test_basic_model():
    _exec_notebook("1_basic_model.ipynb")


def test_timeseries_objects():
    _exec_notebook("2_timeseries_objects.ipynb")
