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
            exec(compile(f.read(), file, "exec"))  # noqa: S102
            # Report success
            print(f"Example {file} ran successfully.")
        plt.close("all")
    except Exception as e:  # noqa: BLE001
        os.chdir(cwd)
        raise RuntimeError(f"could not run {file}, error: {e}")
    os.chdir(cwd)


if __name__ == "__main__":
    for file in files:
        test_example(file)
