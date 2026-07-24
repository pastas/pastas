from pathlib import Path

import papermill as pm
import pytest

pathname = Path(__file__).parent.parent / "doc/examples"
files = list(pathname.glob("*.ipynb"))

EXCLUDED_NOTEBOOKS = {"prepare_timeseries.ipynb"}


@pytest.mark.notebooks
@pytest.mark.parametrize("file", files, ids=lambda f: f.name)
def test_notebook(file: Path) -> None:
    if file.name in EXCLUDED_NOTEBOOKS:
        pytest.skip(f"Skipping excluded notebook: {file.name}")

    # Executes the notebook in-place and populates output cells
    pm.execute_notebook(
        input_path=file,
        output_path=file,  # Overwrites file in-place
        execution_timeout=600,
        kernel_name="python3",
    )
