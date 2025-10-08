import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

pathname = Path(__file__).parent.parent / "doc/examples"
files = list(pathname.glob("*.py"))


@pytest.mark.parametrize("file", files)
def test_example(file) -> None:
    cwd = os.getcwd()
    os.chdir(pathname)
    try:
        # run each example
        with open(file) as f:
            exec(compile(f.read(), file, "exec"))
            # Report success
            print(f"Example {file} ran successfully.")
        plt.close("all")
    except Exception as e:
        os.chdir(cwd)
        raise Exception(f"could not run {file}") from e
    os.chdir(cwd)


if __name__ == "__main__":
    for file in files:
        test_example(file)
