import logging
from importlib import import_module, metadata
from platform import python_version

logger = logging.getLogger(__name__)

__version__ = "1.0.0"


def check_numba_scipy() -> bool:
    try:
        import_module("numba_scipy")
    except ModuleNotFoundError:
        logger.warning(
            "numba_scipy is not installed, defaulting to numpy implementation."
        )
        return False

    scipy_version = metadata.version("scipy")
    scipy_version_nsc = [x for x in metadata.requires("numba-scipy") if "scipy" in x]
    scipy_version_nsc = scipy_version_nsc[0].split(",")[0].split("=")[-1]
    if scipy_version > scipy_version_nsc:
        logger.warning(
            f"numba_scipy supports SciPy<={scipy_version_nsc}, found {scipy_version}"
        )
        return False
    return True


def show_versions(lmfit: bool = True, latexify: bool = True) -> None:
    """Method to print the version of dependencies.

    Parameters
    ----------
    lmfit: bool, optional
        Print the version of LMfit if installed.
    latexify: bool, optional
        Print the version of Latexify if installed.

    """

    msg = (
        f"Python version: {python_version()}\n"
        f"NumPy version: {metadata.version('numpy')}\n"
        f"Pandas version: {metadata.version('pandas')}\n"
        f"SciPy version: {metadata.version('scipy')}\n"
        f"Matplotlib version: {metadata.version('matplotlib')}\n"
        f"Numba version: {metadata.version('numba')}"
    )

    if lmfit:
        msg += "\nLMfit version: "
        try:
            import_module("lmfit")
            msg += f"{metadata.version('lmfit')}"
        except ModuleNotFoundError:
            msg += "Not Installed"

    if latexify:
        msg += "\nLatexify version: "
        try:
            import_module("latexify")
            msg += f"{metadata.version('latexify-py')}"
        except ModuleNotFoundError:
            msg += "Not Installed"

    msg += f"\nPastas version: {__version__}"

    return print(msg)
