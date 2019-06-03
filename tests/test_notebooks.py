# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:10:05 2019

@author: Artesia
"""
import os
import shutil

def test_notebooks():
    # get list of notebooks to run
    
    cwd = os.getcwd()
    
    nbdir = os.path.join('examples', 'notebooks')
    os.chdir(nbdir)
    
    testdir = 'build'
    if os.path.isdir(testdir):
        shutil.rmtree(testdir)
    os.mkdir(testdir)
    
    # run each notebook
    files = [f for f in os.listdir() if f.endswith('.ipynb')]
    for file in files:
        if file not in ['10_pastas_project.ipynb']:
            try:
                # run autotest on each notebook
                cmd = 'jupyter ' + 'nbconvert ' + \
                      '--ExecutePreprocessor.timeout=600 ' + \
                      '--to ' + 'notebook ' + \
                      '--execute ' + '"{}" '.format(file) + \
                      '--output-dir ' + '{} '.format(testdir)
                ival = os.system(cmd)
                msg = 'could not run {}'.format(file)
                assert ival == 0, msg
                assert os.path.isfile(os.path.join(testdir,file)), msg
            except Exception as e:
                os.chdir(cwd)
                raise Exception(e)
    os.chdir(cwd)

if __name__ == '__main__':
    test_notebooks()
