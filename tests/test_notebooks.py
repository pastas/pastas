# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:10:05 2019

@author: Artesia
"""
import os
import shutil

nbdir = os.path.join('examples', 'notebooks')

tempdir = 'build'
if os.path.isdir(tempdir):
    shutil.rmtree(tempdir)
os.mkdir(tempdir)

testdir = os.path.join(tempdir, 'Notebooks')
if os.path.isdir(testdir):
    shutil.rmtree(testdir)
os.mkdir(testdir)

def test_notebooks():
    # get list of notebooks to run
    files = [f for f in os.listdir(nbdir) if f.endswith('.ipynb')]
    
    # run each notebook
    for file in files:
        # run autotest on each notebook
        fname = os.path.join(nbdir, file)
        cmd = 'jupyter ' + 'nbconvert ' + \
              '--ExecutePreprocessor.timeout=600 ' + \
              '--to ' + 'notebook ' + \
              '--execute ' + '"{}" '.format(fname) + \
              '--output-dir ' + '{} '.format(testdir) + \
              '--output ' + '"{}"'.format(file)
        ival = os.system(cmd)
        msg = 'could not run {}'.format(file)
        assert ival == 0 and os.path.isfile(os.path.join(testdir,file)), msg

if __name__ == '__main__':
    test_notebooks()
