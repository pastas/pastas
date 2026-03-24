"""Tests for loading .pas files.

Add a version to VERSIONS to generate and test additional pas files for that Pastas
version. Generation is skipped for versions for which a directory already exists. Newly
generated directories are removed after the test session finishes.

    pytest tests/test_pas_files.py
"""

import atexit
import shutil
import subprocess
from pathlib import Path

import pytest

import pastas as ps

# Pastas versions to generate and test — extend as needed.
PASTAS_VERSIONS = ["1.13.2"]

DATADIR = Path(__file__).parent / "data"
_GENERATED_DIRS: list[Path] = []


def generate_pas_files(version: str) -> Path:
    output_dir = DATADIR / f"pas_files_{version}"
    # Generate if directory doesn't exist or is empty
    if not output_dir.exists() or not list(output_dir.glob("*.pas")):
        subprocess.run(
            [
                "uv",
                "run",
                "--with",
                f"pastas=={version}",
                "generate_pas_files.py",
            ],
            check=True,
            cwd=str(Path(__file__).parent),
        )
        _GENERATED_DIRS.append(output_dir)
    return output_dir


def _get_pas_files() -> list[Path]:
    """Get list of generated pas files."""
    return [
        p
        for version in PASTAS_VERSIONS
        for p in sorted(DATADIR.glob(f"pas_files_{version}/*.pas"))
    ]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate pas files and parametrize test.

    `metafunc` is a Pytest object that represents a test function during the
    test collection phase. We use it here to inspect the function's arguments
    (`metafunc.fixturenames`) to check if it needs the `pas_file` parameter,
    and if so, we dynamically generate a separate test case for each file using
    `metafunc.parametrize()`.
    """
    if "pas_file" in metafunc.fixturenames:
        # Generate files before parametrization
        for version in PASTAS_VERSIONS:
            generate_pas_files(version)

        # Now get the generated files
        pas_files = _get_pas_files()
        metafunc.parametrize(
            "pas_file",
            pas_files,
            ids=[f"{p.parent.name}/{p.stem}" for p in pas_files],
        )


XFAIL = {
    "ChangeModel.pas": "Known issue with ChangeModel in <=1.13.2",
}


@pytest.mark.pasfiles
def test_load_pas_file(pas_file: Path) -> None:
    """Load and test a .pas file."""
    if str(pas_file.name) in XFAIL:
        pytest.xfail(XFAIL[pas_file.name])
    ps.io.load(pas_file)


def cleanup_generated_pas_files():
    for directory in _GENERATED_DIRS:
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
    _GENERATED_DIRS.clear()


atexit.register(cleanup_generated_pas_files)
