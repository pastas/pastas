"""Tests for loading .pas files.

Add version strings to PASTAS_VERSIONS to generate and test additional
galleries. Generation is skipped for versions whose gallery directory already
exists. Newly generated directories are removed after the test class finishes.
Versions that cannot run on the Python interpreter running the tests (e.g. an
old Pastas version on a new Python) are skipped.

    pytest tests/test_pas_files.py
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import pastas as ps

# Pastas versions to generate and test — extend as needed.
PASTAS_VERSIONS = ["1.13.2"]

DATADIR = Path(__file__).parent / "data"

# Tracks directories created during this session so teardown_class can clean up.
GENERATED_DIRS: list[Path] = []

# Pastas versions that cannot be run with the current Python interpreter,
# mapped to the reason why. Filled while collecting the .pas files.
UNSUPPORTED_VERSIONS: dict[str, str] = {}


def _uv_run_with_pastas(version: str, *args: str) -> subprocess.CompletedProcess:
    """Run a command in an ephemeral environment with pastas==version installed.

    The current interpreter (sys.executable) is requested explicitly, so the
    result reflects whether pastas==version runs on the Python that is running
    the tests.
    """
    return subprocess.run(
        [
            "uv",
            "run",
            "--python",
            sys.executable,
            "--with",
            f"pastas=={version}",
            *args,
        ],
        cwd=str(Path(__file__).parent),
        check=False,
        capture_output=True,
    )


def unsupported_reason(version: str) -> str | None:
    """Check whether pastas==version can be imported with the current Python.

    Returns None if the version runs, otherwise a human-readable reason.
    """
    try:
        result = _uv_run_with_pastas(version, "python", "-c", "import pastas")
    except OSError as exc:  # e.g. uv is not installed
        return f"cannot run pastas=={version} with uv: {exc}"
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip().splitlines()
        reason = (
            f"pastas=={version} does not support Python {platform.python_version()}"
        )
        if detail:
            reason = f"{reason} ({detail[-1]})"
        return reason
    return None


def generate_pas_files(version: str) -> Path:
    output_dir = DATADIR / f"pas_files_{version}"
    # Generate if directory doesn't exist or is empty
    if not output_dir.exists() or not list(output_dir.glob("*.pas")):
        subprocess.run(
            [
                "uv",
                "run",
                "--python",
                sys.executable,
                "--with",
                f"pastas=={version}",
                "generate_pas_files.py",
            ],
            check=True,
            cwd=str(Path(__file__).parent),
        )
        GENERATED_DIRS.append(output_dir)
    return output_dir


def collect_pas_files() -> list[Path]:
    """Collect the .pas files of all Pastas versions supported on this Python."""
    pas_files: list[Path] = []
    for version in PASTAS_VERSIONS:
        reason = unsupported_reason(version)
        if reason is not None:
            UNSUPPORTED_VERSIONS[version] = reason
            continue
        pas_files.extend(sorted(generate_pas_files(version).glob("*.pas")))
    return pas_files


PAS_FILES = collect_pas_files() if os.environ.get("CI_PASFILES", "1") == "1" else []

XFAIL = {
    "ChangeModel.pas": "Known issue with ChangeModel in <=1.13.2",
    "Ireson.pas": "No Ireson recharge model in <=1.14.0",
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

    @pytest.mark.pasfiles
    @pytest.mark.parametrize("version", PASTAS_VERSIONS)
    def test_version_supported(self, version: str) -> None:
        if version in UNSUPPORTED_VERSIONS:
            pytest.skip(UNSUPPORTED_VERSIONS[version])

    @classmethod
    def teardown_class(cls) -> None:
        for directory in GENERATED_DIRS:
            shutil.rmtree(directory, ignore_errors=True)
        GENERATED_DIRS.clear()
