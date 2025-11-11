"""Immutable datatypes for Pastas models."""

from logging import getLogger

import pandas as pd

logger = getLogger(__name__)


class SettingsDict(dict):
    """A dict subclass that warns when modifications are attempted.

    This class is used to return a copy of settings that warns users
    when they try to modify it, alerting them that changes won't affect
    the original model.
    """

    def __setitem__(self, key, value):
        msg = (
            f"Modifying settings['{key}'] on a copy of the settings dictionary. This"
            "change will not affect the model. Model settings are managed internally "
            "and updated through methods like ml.solve() and ml.initialize()."
        )
        logger.warning(msg)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        msg = (
            f"Deleting settings['{key}'] on a copy of the settings dictionary. This"
            "change will not affect the model."
        )
        logger.warning(msg)
        super().__delitem__(key)

    def update(self, *args, **kwargs):
        msg = (
            "Updating settings on a copy of the settings dictionary. These changes will"
            "not affect the model."
        )
        logger.warning(msg)
        super().update(*args, **kwargs)


class ParameterDataFrame(pd.DataFrame):
    """A DataFrame subclass that warns when modifications are attempted.

    This class is used to return a copy of parameters that warns users
    when they try to modify it, alerting them that changes won't affect
    the original model and pointing them to use ml.set_parameter() instead.
    """

    def __setitem__(self, key, value):
        msg = (
            f"Modifying parameters['{key}'] on a copy of the parameters DataFrame"
            "This change will not affect the model. Use 'ml.set_parameter(name, ...)' "
            "to modify parameter properties. See the documentation for ml.set_parameter"
            " for more information."
        )
        logger.warning(msg)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        msg = (
            f"Deleting parameters['{key}'] on a copy of the parameters DataFrame. This"
            " change will not affect the model."
        )
        logger.warning(msg)
        super().__delitem__(key)

    @property
    def _constructor(self):
        """Return the constructor for slicing operations."""
        return ParameterDataFrame
