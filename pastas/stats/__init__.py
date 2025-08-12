# ruff: noqa: F401
"""The stats sub-package contains statistical methods for Pastas.

.. currentmodule:: pastas.stats

.. autosummary::
    :toctree: ./generated
    :nosignatures:

    core
    metrics
    dutch
    sgi
    signatures
    tests

"""

import pastas.stats.metrics as metrics
import pastas.stats.signatures as signatures

from .core import acf, ccf, mean, std, var
from .dutch import gg, ghg, glg, gvg, q_ghg, q_glg, q_gvg
from .metrics import (
    aic,
    bic,
    evp,
    kge,
    kge_2012,
    mae,
    nnse,
    nse,
    pearsonr,
    picp,
    rmse,
    rsq,
    sse,
)
from .sgi import sgi
from .tests import diagnostics, durbin_watson, ljung_box, runs_test, stoffer_toloi
