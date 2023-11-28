import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PluginRegistrationError(Exception):
    pass


def register(module, target: Optional[str] = "plugins"):
    """Register a plugin module in pastas.

    Parameters
    ----------
    module : module
        Python file containing plugin classes or functions.
    target : Optional[str], optional
        target sub-module for registering plugin, by default "plugins". See usage
        section for more details.

    Usage
    -----
    Register a new type of stressmodel in pastas::

        import pastas as ps

        # a file my_new_stressmodel.py containing a class NewStressModel
        import my_new_stressmodel

        # register NewStressModel in pastas.stressmodels: ps.stressmodels.NewStressModel
        ps.plugins.register(my_new_stressmodel, target="stressmodels")

        # register NewStressModel in pastas root: ps.NewStressModel
        ps.plugins.register(my_new_stressmodel, target=None)

        # register NewStressModel in new sub-module: ps.experimental.NewStressModel
        ps.plugins.register(my_new_stressmodel, target="experimental")

    """
    import pastas as ps

    objs = [o for o in dir(module) if not o.startswith("_")]
    if target is not None:
        try:
            target = getattr(ps, target)
        except AttributeError:
            setattr(ps, target, module)
            target = getattr(ps, target)
    else:
        target = ps
    for obj in objs:
        if obj in dir(target):
            raise PluginRegistrationError(
                f"Failed registering plugin, <{target.__name__}> already "
                f"contains class or method named '{obj}'."
            )
        setattr(target, obj, getattr(module, obj))
        logger.info(f"Registered plugin '{obj}' in <{target.__name__}>.")
