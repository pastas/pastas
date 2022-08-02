try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    raise ModuleNotFoundError("SolveTimer requires 'tqdm' to be installed.")


class SolveTimer(tqdm):
    """Progress indicator for model optimization.

    Usage
    ----- 
    Print timer and number of iterations in console while running
    `ml.solve()`::

    >>> with SolveTimer() as t:
            ml.solve(callback=t.update)

    This prints the following to the console, for example::

        Optimization progress: 73it [00:01, 67.68it/s]

    Note
    ----
    If the logger is also printing messages to the console the timer will not
    be updated quite as nicely.
    """

    def __init__(self, *args, **kwargs):
        if "total" not in kwargs:
            kwargs['total'] = None
        if "desc" not in kwargs:
            kwargs["desc"] = "Optimization progress"
        super(SolveTimer, self).__init__(*args, **kwargs)

    def update(self, _, n=1):
        displayed = super(SolveTimer, self).update(n)
        return displayed
