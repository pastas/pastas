import shutil
from pathlib import Path

import papermill as pm
import pytest

pathname = Path(__file__).parent.parent / "doc/examples"
files = sorted(pathname.glob("*.ipynb"))

testdir = pathname / "build"
if testdir.is_dir():
    shutil.rmtree(testdir)
testdir.mkdir()

PARAMETERS = {
    "uncertainty_ls_mcmc.ipynb": {"NUM_STEPS": 10},
}

@pytest.mark.notebooks
@pytest.mark.parametrize("file", files)
def test_notebook(file) -> None:
    if file.name in ["prepare_timeseries.ipynb"]:
        pytest.xfail("This notebook checks if errors are raised")

    # Report which notebook is being tested
    print(f"\nTesting notebook: {file}")

    # Use papermill to execute the notebook
    output_path = testdir / file.name
    pm.execute_notebook(
        file,
        output_path,
        parameters=PARAMETERS.get(file.name),
        timeout=600,
        cwd=pathname,
    )

    # Report success
    print(f"Notebook {file} ran successfully.")


if __name__ == "__main__":
    for file in files:
        test_notebook(file)
