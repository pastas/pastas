"""This module contains a timer for model optimization.

The timer prints the time elapsed and number of iterations. Optionally, a maximum solve
time can be specified, to abort long optimizations. This class is not automatically
imported in pastas and requires the tqdm module (pip install tqdm).

Usage::

    from pastas.timer import SolveTimer

    with SolveTimer(max_time=60) as t:  # max time in seconds
        ml.solve(callback=t.timer)

This will print the following to the console::

    Optimization progress: 73it [00:01, 67.68it/s]

"""

try:
    from tqdm.auto import tqdm
except ImportError:
    msg = "SolveTimer requires 'tqdm' to be installed."
    raise ImportError(msg) from None

# Type Hinting
from typing import Optional


class ExceededMaxSolveTime(Exception):
    """Custom Exception when model optimization exceeds threshold."""


class SolveTimer(tqdm):
    """Progress indicator for model optimization.

    Examples
    --------
    Print timer and number of iterations in console while running `ml.solve()`::

        with SolveTimer() as t:
            ml.solve(callback=t.timer)

    This prints the following to the console, for example::

        Optimization progress: 73it [00:01, 67.68it/s]

    Set maximum allowable time (in seconds) for solve, otherwise raise
    ExceededMaxSolveTime exception::

        with SolveTimer(max_time=60) as t:
            ml.solve(callback=t.timer)

    Notes
    -----
    If the logger is also printing messages to the console the timer will not be
    updated quite as nicely.
    """

    def __init__(self, max_time: Optional[float] = None, *args, **kwargs) -> None:
        """Initialize SolveTimer.

        Parameters
        ----------
        max_time : float, optional
            maximum allowed time spent in solve(), by default None, which does
            not impose a limit. If time is exceeded, raises
            ExceededMaxSolveTime Exception.
        """
        if "total" not in kwargs:
            kwargs["total"] = None
        if "desc" not in kwargs:
            kwargs["desc"] = "Optimization progress"
        self.max_time = max_time
        super().__init__(*args, **kwargs)

    def timer(self, _, n: int = 1):
        """Callback method for ps.Model.solve()."""
        displayed = super().update(n)
        if self.max_time is not None:
            if self.format_dict["elapsed"] > self.max_time:
                raise ExceededMaxSolveTime(
                    "Model solve time exceeded" f" {self.max_time} seconds!"
                )
        return displayed
