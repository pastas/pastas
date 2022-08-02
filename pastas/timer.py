try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    raise ModuleNotFoundError("SolveTimer requires 'tqdm' to be installed.")


class ExceededMaxSolveTime(Exception):
    """Custom Exception when model optimization exceeds threshold.
    """
    pass


class SolveTimer(tqdm):
    """Progress indicator for model optimization.

    Usage
    ----- 
    Print timer and number of iterations in console while running
    `ml.solve()`::

        with SolveTimer() as t:
            ml.solve(callback=t.timer)

    This prints the following to the console, for example::

        Optimization progress: 73it [00:01, 67.68it/s]

    Set maximum allowable time (in seconds) for solve, otherwise raise 
    ExceededMaxSolveTime exception::

        with SolveTimer(max_time=60) as t:
            ml.solve(callback=t.timer)

    Note
    ----
    If the logger is also printing messages to the console the timer will not
    be updated quite as nicely.
    """

    def __init__(self, max_time=None, *args, **kwargs):
        """Initialize SolveTimer.

        Parameters
        ----------
        max_time : float, optional
            maximum allowed time spent in solve(), by default None, which does
            not impose a limit. If time is exceeded, raises
            ExceededMaxSolveTime Exception.
        """
        if "total" not in kwargs:
            kwargs['total'] = None
        if "desc" not in kwargs:
            kwargs["desc"] = "Optimization progress"
        self.max_time = max_time
        super(SolveTimer, self).__init__(*args, **kwargs)

    def timer(self, _, n=1):
        """Callback method for ps.Model.solve().
        """
        displayed = super(SolveTimer, self).update(n)
        if self.max_time is not None:
            if self.format_dict["elapsed"] > self.max_time:
                raise ExceededMaxSolveTime("Model solve time exceeded"
                                           f" {self.max_time} seconds!")
        return displayed
