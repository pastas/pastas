from typing import Any

from pastas.version import get_versions, show_versions


def test_get_versions_basic() -> None:
    get_versions()


def test_get_versions_optional() -> None:
    get_versions(optional=True)


def test_show_versions(capsys: Any) -> None:
    show_versions()
