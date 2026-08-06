import os
import shutil
from pathlib import Path

import papermill as pm
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
    if file.name not in [
        "pastas_uncertainty_benchmark.ipynb",
    ]:
        try:
            # Report which notebook is being tested
            print(f"\nTesting notebook: {file}")

            # Use papermill to execute the notebook
            output_path = os.path.join(testdir, file)
            pm.execute_notebook(
                str(file),
                output_path,
                timeout=600,
            )

            msg = f"could not run {file}"
            assert os.path.isfile(output_path), msg
            # Report success
            print(f"Notebook {file} ran successfully.")
        except Exception as e:  # noqa: BLE001
            os.chdir(cwd)
            raise RuntimeError(f"Could not run notebook {file}, error: {e}")
    else:
        print(f"Skipping notebook: {file}")
    os.chdir(cwd)


if __name__ == "__main__":
    for file in files:
        test_notebook(file)
