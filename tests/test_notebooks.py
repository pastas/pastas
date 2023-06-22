import os
import shutil

import pytest

pathname = os.path.join("doc", "examples")
# get list of notebooks to run
files = [f for f in os.listdir(pathname) if f.endswith(".ipynb")]

testdir = "build"
if os.path.isdir(os.path.join(pathname, testdir)):
    shutil.rmtree(os.path.join(pathname, testdir))
os.mkdir(os.path.join(pathname, testdir))


@pytest.mark.notebooks
@pytest.mark.parametrize("file", files)
def test_notebook(file) -> None:
    cwd = os.getcwd()

    os.chdir(pathname)
    if file not in [
        "prepare_timeseries.ipynb",
        "emcee_uncertainty.ipynb",
    ]:
        try:
            # run autotest on each notebook
            cmd = (
                "jupyter "
                + "nbconvert "
                + "--ExecutePreprocessor.timeout=600 "
                + "--to "
                + "notebook "
                + "--execute "
                + '"{}" '.format(file)
                + "--output-dir "
                + "{} ".format(testdir)
            )
            ival = os.system(cmd)
            msg = "could not run {}".format(file)
            assert ival == 0, msg
            assert os.path.isfile(os.path.join(testdir, file)), msg
        except Exception as e:
            os.chdir(cwd)
            raise Exception(e)
    os.chdir(cwd)


if __name__ == "__main__":
    for file in files:
        test_notebook(file)
