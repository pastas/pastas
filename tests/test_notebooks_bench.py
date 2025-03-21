import os
import shutil
from pathlib import Path

import pytest

pathname = Path(__file__).parent.parent / "doc/benchmarks"
files = list(pathname.glob("*.ipynb"))

testdir = "build"
if os.path.isdir(os.path.join(pathname, testdir)):
    shutil.rmtree(os.path.join(pathname, testdir))
os.mkdir(os.path.join(pathname, testdir))


@pytest.mark.bnotebooks
@pytest.mark.parametrize("file", files)
def test_notebook(file) -> None:
    cwd = os.getcwd()

    os.chdir(pathname)
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
