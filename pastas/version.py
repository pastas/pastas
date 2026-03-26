"""Module to get and show the current version of Pastas and its dependencies.

Examples
--------
Show the versions of Pastas and its all dependencies::

    ps.show_versions(optional=True)

"""

import logging
from importlib import import_module, metadata
from platform import python_version

logger = logging.getLogger(__name__)

__version__ = "2.0.0b"


def get_versions(
    optional: bool = False, lmfit: bool = False, latexify: bool = False
) -> str:
    """Method to get the version of dependencies.

    Parameters
    ----------
    optional: bool, optional
        Add the version of optional dependencies if installed.
    lmfit: bool, optional
        Print the version of LMfit if installed. Deprecated since v1.6.0.
    latexify: bool, optional
        Print the version of Latexify if installed. Deprecated since v1.6.0.

    Returns
    -------
    str
        String with the version of the dependencies.

    """
    if lmfit:
        logger.warning(
            "The lmfit argument is deprecated and will be removed in a future version."
        )
    if latexify:
        logger.warning(
            "The latexify argument is deprecated and will be removed in a "
            "future version."
        )

    version_dict = {}
    version_dict["pastas"] = __version__
    version_dict["python"] = python_version()

    required_dependencies = (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "numba",
    )
    for module in required_dependencies:
        version_dict[module] = metadata.version(module)

    if optional:
        optional_dependencies = (
            "requests",
            "lmfit",
            "emcee",
            "bokeh",
            "plotly",
            "latexify",
        )
        for module in optional_dependencies:
            try:
                import_module(module)
                module_name = module if module != "latexify" else "latexify-py"
                version_dict[module] = metadata.version(module_name) + " (optional)"
            except ImportError:
                version_dict[module] = "Not Installed"

    return version_dict


def show_versions(optional: bool = False) -> None:
    """Method to print the version of dependencies.

    Parameters
    ----------
    optional: bool, optional
        Print the version of optional dependencies if installed

    """
    version_dict = get_versions(optional=optional)

    max_len_key = max(len(key) for key in version_dict.keys()) + 1
    msg = ""
    # msg = f"{'Package':<{max_len_key}}: Version\n"
    # msg += "-" * (max_len_key + 9) + "\n"
    for key, value in version_dict.items():
        leftside = f"{key.capitalize()}"
        msg += f"{leftside:<{max_len_key}}: {value}\n"

    print(msg)
