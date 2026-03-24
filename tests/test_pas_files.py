"""Tests for loading .pas files.

Add version strings to VERSIONS to generate and test additional galleries.
Generation is skipped for versions whose gallery directory already exists.
Newly generated directories are removed after the test class finishes.

    pytest tests/test_pas_files.py
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import pastas as ps

# Pastas versions to generate and test — extend as needed.
PASTAS_VERSIONS = ["1.13.2"]

DATADIR = Path(__file__).parent / "data"

# Tracks directories created during this session so teardown_class can clean up.
GENERATED_DIRS: list[Path] = []


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
        GENERATED_DIRS.append(output_dir)
    return output_dir


PAS_FILES = (
    [
        p
        for version in PASTAS_VERSIONS
        for p in sorted(generate_pas_files(version).glob("*.pas"))
    ]
    if os.environ.get("CI_PASFILES", "1") == "1"
    else []
)

XFAIL = {
    "ChangeModel.pas": "Known issue with ChangeModel in <=1.13.2",
}


class TestPasFiles:
    @pytest.mark.pasfiles
    @pytest.mark.parametrize(
        "pas_file",
        PAS_FILES,
        ids=[f"{p.parent.name}/{p.stem}" for p in PAS_FILES],
    )
    def test_load(self, pas_file: Path) -> None:
        if str(pas_file.name) in XFAIL:
            pytest.xfail(XFAIL[pas_file.name])
        ps.io.load(pas_file)

    @classmethod
    def teardown_class(cls) -> None:
        for directory in GENERATED_DIRS:
            shutil.rmtree(directory, ignore_errors=True)
        GENERATED_DIRS.clear()
