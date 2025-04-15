from unittest.mock import patch

from pastas.version import check_numba_scipy, get_versions, show_versions


def test_check_numba_scipy_installed():
    with patch("importlib.import_module") as mock_import_module, \
         patch("importlib.metadata.version") as mock_version, \
         patch("importlib.metadata.requires") as mock_requires:
        mock_import_module.return_value = None
        mock_version.side_effect = lambda x: "1.5.0" if x == "scipy" else None
        mock_requires.return_value = ["scipy<=1.5.0"]

        assert check_numba_scipy() is True


def test_check_numba_scipy_not_installed():
    with patch("importlib.import_module", side_effect=ImportError):
        assert check_numba_scipy() is False


def test_check_numba_scipy_version_mismatch():
    with patch("importlib.import_module") as mock_import_module, \
         patch("importlib.metadata.version") as mock_version, \
         patch("importlib.metadata.requires") as mock_requires:
        mock_import_module.return_value = None
        mock_version.side_effect = lambda x: "1.6.0" if x == "scipy" else None
        mock_requires.return_value = ["scipy<=1.5.0"]

        assert check_numba_scipy() is False


def test_get_versions_basic():
    with patch("platform.python_version", return_value="3.9.0"), \
         patch("importlib.metadata.version") as mock_version:
        mock_version.side_effect = lambda x: {
            "numpy": "1.21.0",
            "pandas": "1.3.0",
            "scipy": "1.7.0",
            "matplotlib": "3.4.2",
            "numba": "0.54.0",
            "custom-package": "0.1.0"
        }.get(x, None)

        expected_versions = {
            "Pastas version": "1.9.0",
            "Python version": "3.9.0",
            "NumPy version": "1.21.0",
            "Pandas version": "1.3.0",
            "SciPy version": "1.7.0",
            "Matplotlib version": "3.4.2",
            "Numba version": "0.54.0",
            "Custom-Package version": "0.1.0"
        }

        result = get_versions()
        for key, value in expected_versions.items():
            assert f"{key}: {value}" in result


def test_get_versions_optional():
    with patch("importlib.metadata.version") as mock_version, \
         patch("importlib.import_module") as mock_import_module:
        mock_version.side_effect = lambda x: {
            "requests": "2.25.1",
            "lmfit": "1.0.2",
            "emcee": "3.0.2",
            "bokeh": "2.3.3",
            "plotly": "5.1.0",
            "latexify-py": "0.0.1"
        }.get(x, None)
        mock_import_module.side_effect = lambda x: None

        result = get_versions(optional=True)
        assert "Requests version: 2.25.1" in result
        assert "lmfit" in result
        assert "emcee" in result
        assert "bokeh" in result
        assert "plotly" in result
        assert "latexify" in result


def test_show_versions(capsys):
    with patch("platform.python_version", return_value="3.9.0"), \
         patch("importlib.metadata.version") as mock_version:
        mock_version.side_effect = lambda x: {
            "numpy": "1.21.0",
            "pandas": "1.3.0",
            "scipy": "1.7.0",
            "matplotlib": "3.4.2",
            "numba": "0.54.0"
        }.get(x, None)

        show_versions()
        captured = capsys.readouterr()
        assert "Pastas version: 1.9.0" in captured.out
        assert "Python version: 3.9.0" in captured.out
        assert "NumPy version: 1.21.0" in captured.out
        assert "Pandas version: 1.3.0" in captured.out
        assert "SciPy version: 1.7.0" in captured.out
        assert "Matplotlib version: 3.4.2" in captured.out
        assert "Numba version: 0.54.0" in captured.out