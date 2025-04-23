"""Tests for utility functions in pastas.utils."""

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Use test utilities from the dedicated file
from pastas import utils


@pytest.fixture
def test_logger() -> logging.Logger:
    """Create a test logger."""
    logger = logging.getLogger("test_pastas_logger")
    yield logger
    # Clean up after test
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.handlers = []


class TestGetStressTminTmax:
    """Test get_stress_tmin_tmax function."""

    def test_with_valid_model(self, ml_solved: Any) -> None:
        """Test with a valid model."""
        # Test function with real model from conftest
        tmin, tmax = utils.get_stress_tmin_tmax(ml_solved)

        # Verify correct types
        assert isinstance(tmin, pd.Timestamp)
        assert isinstance(tmax, pd.Timestamp)

        # Verify tmin and tmax make sense relative to stress data
        for sm_name in ml_solved.stressmodels:
            for stress in ml_solved.stressmodels[sm_name].stress:
                stress_tmin = stress.series_original.index.min()
                stress_tmax = stress.series_original.index.max()
                # tmin should be <= each stress's max time
                # tmax should be >= each stress's min time
                # This ensures there's at least some overlap in the time ranges
                assert tmin <= stress_tmax
                assert tmax >= stress_tmin

    def test_with_invalid_model(self) -> None:
        """Test with an invalid model type."""
        invalid_model = "Not a model"

        with pytest.raises(TypeError):
            utils.get_stress_tmin_tmax(invalid_model)


class TestLoggerFunctions:
    """Test logger-related utility functions."""

    def test_initialize_logger(self, test_logger: logging.Logger) -> None:
        """Test initializing a logger."""
        utils.initialize_logger(test_logger, level=logging.DEBUG)

        assert test_logger.level == logging.DEBUG
        assert any(
            isinstance(handler, logging.StreamHandler)
            for handler in test_logger.handlers
        )
        assert not any(
            isinstance(handler, logging.handlers.RotatingFileHandler)
            for handler in test_logger.handlers
        )

    def test_set_console_handler(self, test_logger: logging.Logger) -> None:
        """Test setting a console handler."""
        # First ensure no console handlers
        utils.remove_console_handler(test_logger)
        assert not any(
            isinstance(handler, logging.StreamHandler)
            for handler in test_logger.handlers
        )

        # Add console handler
        utils.set_console_handler(test_logger, level=logging.WARNING, fmt="%(message)s")

        # Check handler was added
        console_handlers = [
            h for h in test_logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(console_handlers) == 1
        assert console_handlers[0].level == logging.WARNING
        assert console_handlers[0].formatter._fmt == "%(message)s"

    def test_remove_console_handler(self, test_logger: logging.Logger) -> None:
        """Test removing console handlers."""
        # Add console handler
        utils.set_console_handler(test_logger)
        assert any(
            isinstance(handler, logging.StreamHandler)
            for handler in test_logger.handlers
        )

        # Remove console handler
        utils.remove_console_handler(test_logger)
        assert not any(
            isinstance(handler, logging.StreamHandler)
            for handler in test_logger.handlers
        )

    @patch("pastas.utils.handlers.RotatingFileHandler")
    def test_add_file_handlers(
        self, mock_handler: Any, test_logger: logging.Logger
    ) -> None:
        """Test adding file handlers."""
        utils.add_file_handlers(
            test_logger,
            filenames=("test1.log", "test2.log"),
            levels=(logging.INFO, logging.ERROR),
        )

        # Check that RotatingFileHandler was called twice with correct args
        assert mock_handler.call_count == 2
        call_args_list = mock_handler.call_args_list
        assert call_args_list[0][0][0] == "test1.log"
        assert call_args_list[1][0][0] == "test2.log"

    def test_remove_file_handlers(self, test_logger: logging.Logger) -> None:
        """Test removing file handlers."""
        # Mock a file handler
        fh = logging.handlers.RotatingFileHandler("dummy.log")
        test_logger.addHandler(fh)

        assert any(
            isinstance(handler, logging.handlers.RotatingFileHandler)
            for handler in test_logger.handlers
        )

        # Remove file handlers
        utils.remove_file_handlers(test_logger)

        assert not any(
            isinstance(handler, logging.handlers.RotatingFileHandler)
            for handler in test_logger.handlers
        )

    @patch("pastas.utils.logging.getLogger")
    def test_set_log_level(self, mock_get_logger: Any) -> None:
        """Test setting log level."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        utils.set_log_level("ERROR")

        mock_get_logger.assert_called_once_with("pastas")
        mock_logger.setLevel.assert_called_once_with("ERROR")


class TestValidateName:
    """Test validate_name function."""

    @patch("pastas.utils.platform", return_value="Linux")
    def test_valid_name_linux(self, _: Any) -> None:
        """Test with valid name on Linux."""
        name = "valid_name-123"
        result = utils.validate_name(name)
        assert result == name

    @patch("pastas.utils.platform", return_value="Windows")
    def test_valid_name_windows(self, _: Any) -> None:
        """Test with valid name on Windows."""
        name = "valid_name-123"
        result = utils.validate_name(name)
        assert result == name

    @patch("pastas.utils.platform", return_value="Linux")
    def test_invalid_name_linux(self, _: Any) -> None:
        """Test with invalid name on Linux platform."""
        name = "invalid/name with space"

        with patch("pastas.utils.logger") as mock_logger:
            result = utils.validate_name(name)
            assert result == name
            assert mock_logger.warning.call_count == 2  # Both '/' and ' ' are invalid

    @patch("pastas.utils.platform", return_value="Windows")
    def test_invalid_name_windows(self, _: Any) -> None:
        """Test with invalid name on Windows platform."""
        name = "invalid:name"

        with patch("pastas.utils.logger") as mock_logger:
            result = utils.validate_name(name)
            assert result == name
            mock_logger.warning.assert_called_once()

    @patch("pastas.utils.platform", return_value="Linux")
    def test_invalid_name_raise_error(self, _: Any) -> None:
        """Test with invalid name and raise_error=True."""
        name = "invalid/name"

        with pytest.raises(Exception) as excinfo:
            utils.validate_name(name, raise_error=True)

        assert "contains illegal character" in str(excinfo.value)

    def test_non_string_name(self) -> None:
        """Test with non-string name."""
        name = 12345
        result = utils.validate_name(name)
        assert result == "12345"  # Converted to string
