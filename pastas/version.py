import logging
from importlib import import_module, metadata
from platform import python_version

logger = logging.getLogger(__name__)

__version__ = "1.7.0"


def check_numba_scipy() -> bool:
    try:
        import_module("numba_scipy")
    except ImportError:
        logger.warning(
            "numba_scipy is not installed, defaulting to numpy implementation."
        )
        return False

    scipy_version = metadata.version("scipy")
    scipy_version_nsc = [x for x in metadata.requires("numba-scipy") if "scipy" in x]
    scipy_version_nsc = scipy_version_nsc[0].split(",")[0].split("=")[-1]
    if scipy_version > scipy_version_nsc:
        logger.warning(
            "numba_scipy supports SciPy<=%s, found %s", scipy_version_nsc, scipy_version
        )
        return False
    return True


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
            "The lmfit argument is deprecated and will be removed in a "
            "future version."
        )
    if latexify:
        logger.warning(
            "The latexify argument is deprecated and will be removed in a "
            "future version."
        )

    msg = (
        f"Pastas version: {__version__}\n"
        f"Python version: {python_version()}\n"
        f"NumPy version: {metadata.version('numpy')}\n"
        f"Pandas version: {metadata.version('pandas')}\n"
        f"SciPy version: {metadata.version('scipy')}\n"
        f"Matplotlib version: {metadata.version('matplotlib')}\n"
        f"Numba version: {metadata.version('numba')}"
    )

    if optional:
        msg += "\nOptional Dependencies:"

        msg += "\nRequests version: "

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
                msg += f"{metadata.version(module_name)}"
            except ImportError:
                msg += "Not Installed"

    return msg


def show_versions(optional: bool = False) -> None:
    """Method to print the version of dependencies.

    Parameters
    ----------
    optional: bool, optional
        Print the version of optional dependencies if installed

    """
    msg = get_versions(optional=optional)

    return print(msg)
