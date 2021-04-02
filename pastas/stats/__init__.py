"""The stats sub-package contains statistical methods for Pastas.

.. autosummary::
   :nosignatures:
   :toctree: ./generated
   :noindex:

    pastas.stats.core
    pastas.stats.metrics
    pastas.stats.tests
    pastas.stats.sgi
    pastas.stats.dutch

"""

from .core import ccf, acf, mean, std, var
from .dutch import q_ghg, q_glg, q_gvg, ghg, glg, gvg
from .metrics import mae, evp, nse, rmse, sse, rsq, aic, bic, pearsonr, \
    kge_2012
from .tests import runs_test, ljung_box, durbin_watson, stoffer_toloi, \
    diagnostics, plot_acf, plot_diagnostics, plot_cum_frequency
from .sgi import sgi
