import functools
import logging

logger = logging.getLogger(__name__)


def set_parameter(function):
    @functools.wraps(function)
    def _set_parameter(self, name, value, **kwargs):
        if name not in self.parameters.index:
            logger.warning('Parameter name %s does not exist, please choose '
                           'from %s' % (name, self.parameters.index))
        else:
            return function(self, name, value, **kwargs)

    return _set_parameter


def get_tseries(function):
    @functools.wraps(function)
    def _get_tseries(self, name, **kwargs):
        if name not in self.tseriesdict.keys():
            logger.warning('The tseries name you provided is not in the '
                           'tseriesdict. Please select from the following '
                           'list: %s' % self.tseriesdict.keys())
        else:
            return function(self, name, **kwargs)

    return _get_tseries
