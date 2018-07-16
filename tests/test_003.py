import os

import matplotlib.pyplot as plt


def test_example():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example.py").read())
    plt.close('all')
    os.chdir(cwd)
    return


def test_example_menyanthes():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_menyanthes.py").read())
    plt.close('all')
    os.chdir(cwd)
    return


def test_example_project():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_project.py").read())
    plt.close('all')
    os.chdir(cwd)
    return


def test_example_docs():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_docs.py").read())
    plt.close('all')
    os.chdir(cwd)
    return


def test_example_stats():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_stats.py").read())
    plt.close('all')
    os.chdir(cwd)
    return

# TODO Make faster.Too slow for testing now
# def test_example_no_conv():
#     plt.ion()
#     cwd = os.getcwd()
#     os.chdir("examples")
#     exec(open("example_no_conv.py").read())
#     plt.close('all')
#     os.chdir(cwd)
#     return

# TODO Fix WellModel before testing again
# def test_example_WellModel():
#     plt.ion()
#     cwd = os.getcwd()
#     os.chdir("examples")
#     exec(open("example_WellModel.py").read())
#     plt.close('all')
#     os.chdir(cwd)
#     return


def test_example_timestep_weighted_resample():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_timestep_weighted_resample.py").read())
    plt.close('all')
    return os.chdir(cwd)
