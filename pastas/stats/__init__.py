# ruff: noqa: F401
"""Stats sub-package contains statistical methods for Pastas."""

from pastas.stats import metrics, signatures

from .core import acf, ccf, mean, moment, std, var
from .dutch import gg, ghg, glg, gvg, q_ghg, q_glg, q_gvg
from .metrics import (
    aic,
    bic,
    evp,
    kge,
    kge_2012,
    mae,
    me,
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
