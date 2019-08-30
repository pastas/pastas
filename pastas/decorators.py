from functools import wraps
from logging import getLogger

logger = getLogger(__name__)


def set_parameter(function):
    @wraps(function)
    def _set_parameter(self, name, value, **kwargs):
        if name not in self.parameters.index:
            logger.error("Parameter name {} does not exist, please choose "
                         "from {}".format(name, self.parameters.index))
        else:
            return function(self, name, value, **kwargs)

    return _set_parameter


def get_stressmodel(function):
    @wraps(function)
    def _get_stressmodel(self, name, **kwargs):
        if name not in self.stressmodels.keys():
            logger.error("The stressmodel name you provided is not in the "
                         "stressmodels dict. Please select from the "
                         "following list: {}".format(self.stressmodels.keys()))
        else:
            return function(self, name, **kwargs)

    return _get_stressmodel


def model_tmin_tmax(function):
    @wraps(function)
    def _model_tmin_tmax(self, tmin=None, tmax=None, *args, **kwargs):
        if tmin is None:
            tmin = self.ml.settings["tmin"]
        if tmax is None:
            tmax = self.ml.settings["tmax"]

        return function(self, tmin, tmax, *args, **kwargs)

    return _model_tmin_tmax


def PastasDeprecationWarning(function):
    @wraps(function)
    def _function(*args, **kwargs):
        logger.warning("Deprecation warning: method will be deprecated "
                       "in version 0.13.0.")
        return function(*args, **kwargs)

    return _function
