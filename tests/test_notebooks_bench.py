import shutil
from pathlib import Path

import papermill as pm
import pytest

pathname = Path(__file__).parent.parent / "doc/benchmarks"
files = list(pathname.glob("*.ipynb"))

testdir = pathname / "build"
if testdir.is_dir():
    shutil.rmtree(testdir)
testdir.mkdir()


PARAMETERS = {
    "pastas_uncertainty_benchmark.ipynb": {"NUMBER_EXPERIMENTS": 10},
}


@pytest.mark.bnotebooks
@pytest.mark.parametrize("file", files)
def test_notebook(file) -> None:
    # Report which notebook is being tested
    print(f"\nTesting notebook: {file}")

    # Use papermill to execute the notebook
    pm.execute_notebook(
        input_path=file,
        output_path=testdir / file.name,
        timeout=600,
        parameters=PARAMETERS.get(file.name),
        cwd=pathname,
    )
    # Report success
    print(f"Notebook {file} ran successfully.")


if __name__ == "__main__":
    for file in files:
        test_notebook(file)
