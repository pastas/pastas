import os
import matplotlib.pyplot as plt


def test_examples():
    # run all the examples in the following folders
    pathnames = ['examples', 'examples/reads']
    # pathnames = ['examples/reads']
    cwd = os.getcwd()
    # Turn interactive mode on, so that the figures do not block the main thread
    plt.ion()
    for pathname in pathnames:
        os.chdir(cwd)
        files = os.listdir(pathname)
        os.chdir(pathname)
        for file in files:
            if file.endswith('.py'):
                print('testing example ' + file)
                exec (open(file).read())
                # close the figures again
                plt.close('all')
    os.chdir(cwd)
    return 'all examples work!'
