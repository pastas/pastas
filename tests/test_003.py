import os

import matplotlib.pyplot as plt


def test_example():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example.py").read())
    plt.close('all')
    return os.chdir(cwd)


def test_example_menyanthes():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_menyanthes.py").read())
    plt.close('all')
    return os.chdir(cwd)


def test_example_project():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_project.py").read())
    plt.close('all')
    return os.chdir(cwd)


def test_example_docs():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_docs.py").read())
    plt.close('all')
    return os.chdir(cwd)


def test_example_stats():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_stats.py").read())
    plt.close('all')
    return os.chdir(cwd)


# TODO Fix WellModel before testing again
# def test_example_WellModel():
#     plt.ion()
#     cwd = os.getcwd()
#     os.chdir("examples")
#     exec(open("example_WellModel.py").read())
#     plt.close('all')
#     return os.chdir(cwd)


def test_example_timestep_weighted_resample():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_timestep_weighted_resample.py").read())
    plt.close('all')
    return os.chdir(cwd)

def test_example_transform():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_transform.py").read())
    plt.close('all')
    return os.chdir(cwd)

def test_example_900():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_900.py").read())
    plt.close('all')
    return os.chdir(cwd)

def test_example_step():
    plt.ion()
    cwd = os.getcwd()
    os.chdir("examples")
    exec(open("example_step.py").read())
    plt.close('all')
    return os.chdir(cwd)
