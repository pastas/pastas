"""The stats sub-package contains statistical methods for Pastas.

"""

import pastas.stats.metrics as metrics
import pastas.stats.signatures as signatures

from .core import acf, ccf, mean, std, var
from .dutch import ghg, glg, gvg, q_ghg, q_glg, q_gvg
from .metrics import aic, bic, evp, kge_2012, mae, nse, pearsonr, rmse, rsq, sse
from .sgi import sgi
from .tests import (
    diagnostics,
    durbin_watson,
    ljung_box,
    plot_acf,
    plot_cum_frequency,
    plot_diagnostics,
    runs_test,
    stoffer_toloi,
)
