import logging

from pastas.extensions.accessor import register_model_accessor

logger = logging.getLogger(__name__)


def register_plotly():
    """Register Plotly extension for pastas.Model class for interactive plotting."""
    from pastas.plotting.plotly import Plotly

    logger.info("Registered plotly extension in Model class, e.g. `ml.plotly.plot()`.")
