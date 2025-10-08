"""Tests for the timer module."""

import time
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from pastas.timer import ExceededMaxSolveTime, SolveTimer


class TestSolveTimer:
    """Test class for SolveTimer."""

    def test_init(self) -> None:
        """Test initialization of SolveTimer."""
        # Test default initialization
        timer = SolveTimer()
        assert timer.max_time is None
        assert timer.desc == "Optimization progress"
        assert timer.total is None

        # Test custom initialization
        timer = SolveTimer(max_time=60, desc="Custom progress", total=100)
        assert timer.max_time == 60
        assert timer.desc == "Custom progress"
        assert timer.total == 100

    @patch("pastas.timer.tqdm.update")
    def test_timer_callback(self, mock_update: Any) -> None:
        """Test the timer callback function."""
        # Setup mock return value for parent update method
        mock_update.return_value = True

        # Create SolveTimer with no max_time
        timer = SolveTimer()

        # Call timer callback and verify update was called
        result = timer.timer(None)
        mock_update.assert_called_once_with(1)
        assert result is True

        # Reset mock and test with custom n value
        mock_update.reset_mock()
        timer.timer(None, n=5)
        mock_update.assert_called_once_with(5)

    @patch("pastas.timer.tqdm.__init__")
    def test_custom_kwargs_passed_to_parent(self, mock_init: Any) -> None:
        """Test that custom kwargs are passed to parent class."""
        mock_init.return_value = None

        # Initialize with custom kwargs
        SolveTimer(max_time=30, leave=False, position=1, colour="green")

        # Check if kwargs were passed to parent
        _, kwargs = mock_init.call_args
        assert kwargs.get("leave") is False
        assert kwargs.get("position") == 1
        assert kwargs.get("colour") == "green"


@pytest.mark.skipif(
    "tqdm" not in pytest.importorskip("tqdm").__file__,
    reason="Test requires tqdm to be installed directly, not as a vendored package",
)
def test_real_usage() -> None:
    """Test SolveTimer in a scenario closer to real usage."""

    # Create a mock function to simulate model.solve
    def mock_solve(callback: Callable | None = None) -> None:
        for i in range(5):
            if callback:
                callback(None)
            time.sleep(0.01)

    # Test with a small max_time that won't be exceeded
    with SolveTimer(max_time=1) as timer:
        mock_solve(callback=timer.timer)

    # Test with a max_time that will be exceeded
    with pytest.raises(ExceededMaxSolveTime):
        with SolveTimer(max_time=0.02) as timer:
            mock_solve(callback=timer.timer)
