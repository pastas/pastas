"""Module for accessor registration for extending Pastas model classes."""

# copied and adapted from pandas/core/accessor.py
import logging

logger = logging.getLogger(__name__)


class CachedAccessor:
    """Custom property-like object.

    A descriptor for caching accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``ml.foo``.
    accessor : cls
        Class with the extension methods.
    """

    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        """Get the accessor object for the instance.

        Returns the accessor class if accessed from the class, or creates and caches
        an instance of the accessor if accessed from an instance.
        """
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        accessor_obj = self._accessor(obj)
        # Replace the property with the accessor object. Inspired by:
        # https://www.pydanny.com/cached-property.html
        # We need to use object.__setattr__ because we overwrite __setattr__ on
        # NDFrame
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def _register_accessor(name: str, cls):
    """
    Register a custom accessor on objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Returns
    -------
    callable
        A class decorator.

    Notes
    -----
    When accessed, your accessor will be initialized with the pastas object
    the user is interacting with. So the signature must be

    .. code-block:: python

        def __init__(self, pastas_object):
            ...

    Examples
    --------
    In your library code::

        import pastas as ps

        @ps.extensions.register_plotly_accessor("plotly")
        class Plotly:
            def __init__(self, model):
                self._obj = model

            def plot(self):
                # plotly plot function
                pass

    Back in an interactive IPython session:

    .. code-block:: ipython

        In [1]: from pastas.plotting.plotly import Plotly
        In [2]: ml = ps.Model(oseries)
        In [3]: ml.plotly.plot()  # plots interactive figure
    """

    def decorator(accessor):
        if hasattr(cls, name):
            msg = (
                "registration of accessor %s under name %s for type %s is overriding"
                " a preexisting attribute with the same name."
            )
            logger.warning(msg, repr(accessor), repr(name), repr(cls))
        setattr(cls, name, CachedAccessor(name, accessor))
        cls._accessors.add(name)
        return accessor

    return decorator


def register_model_accessor(name: str):
    """Register a custom accessor for Pastas Model classes.

    Parameters
    ----------
    name : str
        The name of the accessor to register.

    Returns
    -------
    decorator : callable
        A decorator to apply to the accessor class.
    """
    from pastas import Model

    return _register_accessor(name, Model)
