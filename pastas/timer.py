"""This module contains a timer for model optimization.

The timer prints the time elapsed and number of iterations. Optionally, a maximum solve
time can be specified, to abort long optimizations. This class is not automatically
imported in pastas and requires the tqdm module (``pip install tqdm``).

Examples
--------
Usage::

    from pastas.timer import SolveTimer

    with SolveTimer(max_time=60) as t:  # max time in seconds
        ml.solve(callback=t.timer)

This will print the following to the console::

    Optimization progress: 73it [00:01, 67.68it/s]

"""

from pastas.decorators import PastasDeprecationWarning
from pastas.stats import metrics


@PastasDeprecationWarning(
    version="2.0.0",
    reason="The ExceededMaxSolveTime exception has been renamed to TimeoutError.",
)
class ExceededMaxSolveTime(Exception):
    pass


try:
    from tqdm.auto import tqdm
except ImportError:
    msg = "SolveTimer requires 'tqdm' to be installed."
    raise ImportError(msg) from None


class SolveTimer(tqdm):
    """Progress indicator for model optimization.

    Examples
    --------
    Print timer and number of iterations in console while running `ml.solve()`::

        with SolveTimer() as t:
            ml.solve(callback=t.timer)

    This prints the following to the console, for example::

        Optimization progress: 73it [00:01, 67.68it/s]


    Notes
    -----
    If the logger is also printing messages to the console the timer will not be
    updated quite as nicely.
    """

    def __init__(self, *args, max_time: float | None = None, **kwargs) -> None:
        """Initialize SolveTimer.

        Parameters
        ----------
        max_time : float, optional
            maximum allowed time spent in solve(), by default None, which does
            not impose a limit. If time is exceeded, raises RunTimeError.
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
                raise TimeoutError(
                    f"Model solve time exceeded {self.max_time} seconds!"
                )
        return displayed


class StatTimer(SolveTimer):
    """StatTimer that updates a user-specified solve statistic every N iterations."""

    def __init__(
        self, ml, *args, statistic="rmse", update_interval: int | None = None, **kwargs
    ) -> None:
        """
        Parameters
        ----------
        ml : pastas.Model
            The model being solved, used to compute residuals.
        statistic : str, optional
            The statistic to compute and display, by default "rmse". Must be a valid
            statistic in pastas.stats.metrics that accepts ``res=`` as an argument.
        update_interval : int, optional
            Number of iterations between RMSE updates. If None (default), the
            RMSE is updated when iteration % number of varying parameters == 0.
        """
        self.ml = ml
        if update_interval is not None:
            self.update_interval = update_interval
        else:
            self.update_interval = self.ml.parameters.vary.sum()
        self.statistic = statistic
        self.func = getattr(metrics, self.statistic)
        super().__init__(*args, **kwargs)

    def timer(self, p, n: int = 1):
        """Callback method that updates RMSE in the progress bar."""
        # extra overhead to compute residuals again, though with caching
        # this will be faster
        if (self.n % self.update_interval) == 0:
            if self.ml.noisemodel is not None:
                rv = self.ml.noise(p) * self.ml.noise_weights(p)
            else:
                rv = self.ml.residuals(p)
            stat = self.func(res=rv)
            self.set_postfix(**{f"{self.statistic.upper()}": f"{stat:.4e}"})
        return super().timer(p, n)
