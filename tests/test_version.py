from unittest.mock import patch

from pastas.version import check_numba_scipy, get_versions, show_versions


def test_check_numba_scipy_not_installed():
    with patch("importlib.import_module", side_effect=ImportError):
        assert check_numba_scipy() is False


def test_check_numba_scipy_version_mismatch():
    with (
        patch("importlib.import_module") as mock_import_module,
        patch("importlib.metadata.version") as mock_version,
        patch("importlib.metadata.requires") as mock_requires,
    ):
        mock_import_module.return_value = None
        mock_version.side_effect = lambda x: "1.6.0" if x == "scipy" else None
        mock_requires.return_value = ["scipy<=1.5.0"]

        assert check_numba_scipy() is False


def test_get_versions_basic():
    get_versions()


def test_get_versions_optional():
    get_versions(optional=True)


def test_show_versions(capsys):
    show_versions()
