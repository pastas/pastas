"""This module contains plottings methods for Pastas projects.

Raoul Collenteur, 2018 - Artesia Water.

"""
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, mls):
        """

        Parameters
        ----------
        mls: pastas.Project
            Pastas project

        """
        self.mls = mls

    def stresses(self, kind=None, cols=2, **kwargs):
        """Make plots of the stresses in different subplots.

        Parameters
        ----------
        kind: str
            string with one of the types of the stresses.
        cols: int
            Number of columns to divide the plots over.

        Returns
        -------
        ax: matplotlib.axes
            returns a list of matplotlib axes instances.

        """
        if isinstance(kind, str):
            kinds = [kind]
        else:
            kinds = list(kind)

        stresses = self.mls.stresses.index[self.mls.stresses.kind.isin(kinds)]
        num = len(stresses)
        rows = -(-num // cols)  # round up with out additional import

        # Automatically adjust figsize
        if "figsize" not in kwargs.keys():
            norm_size = plt.rcParams["figure.figsize"]
            kwargs["figsize"] = [cols / 2 * norm_size[0],
                                 rows / 5 * norm_size[1]]

        _, ax = plt.subplots(rows, cols, **kwargs)

        if hasattr(ax, "flatten"):
            ax = ax.flatten()
        else:
            ax = [ax]

        for i, key in enumerate(stresses):
            self.mls.stresses.loc[key, "series"].series.plot(ax=ax[i],
                                                             x_compat=True)
            ax[i].legend([key], loc=2)
        return ax
