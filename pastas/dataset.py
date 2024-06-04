"""This module contains functions to load datasets from the pastas-data repository on
GitHub. The datasets are used for testing and examples in the documentation. The
load_dataset function can be used to load a single csv file or multiple csv files from
a subfolder in the pastas-data repository.

"""

from functools import lru_cache
from typing import Dict, List, Union

from pandas import DataFrame, read_csv

GITHUB_URL = "https://api.github.com/repos/pastas/pastas-data/contents/"


@lru_cache
def load_dataset(name: str) -> Union[DataFrame, Dict[str, DataFrame]]:
    """Load csv-files from a subfolder in the pastas dataset repository on GitHub.

    Parameters
    ----------
    name : str
        The name of the subfolder, i.e., collenteur_2023. For a list of available
        datasets, see the pastas-data repository on GitHub
        (www.github.com/pastas/pastas-data).

    Returns
    -------
    Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        The loaded dataset(s). If one csv file is found, returns a pandas DataFrame.
        If multiple csv files are found, returns a dictionary with file names as keys
        and dataframes as values.

    Raises
    ------
    Exception: If the request status code is not 200 (OK), an exception is raised. This
    is likely due to an invalid folder name. Check the pastas-data repository on GitHub
    for available datasets.

    Examples
    --------
    >>> ps.load_dataset("collenteur_2021")
    Returns the dataset from the "collenteur_2021" subfolder as a pandas DataFrame.

    >>> ps.load_dataset("collenteur_2023")
    Returns a dictionary with datasets from the "collenteur_2023" subfolder. The keys
    are the file names and the values are pandas DataFrames.

    """
    # Try to import requests, if not installed raise error
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The requests package is required to load datasets from the pastas-data "
            "repository. Install requests using 'pip install requests'."
        )

    # Get the folder from the pastas-data repository
    r = requests.get(f"{GITHUB_URL}/{name}/")

    # Check if requests status is okay, otherwise raise error and return status code
    if not r.status_code == 200:
        raise Exception(f"Error: {r.status_code}. Reason: {r.reason}. ")

    # Get information about the files in the folder
    data = {}

    # Loop over the files in the folder
    rjson = r.json()
    read_csv_kwargs = requests.get(
        [x for x in rjson if x["name"] == "settings.json"][0]["download_url"]
    ).json()
    for file in rjson:
        fname = file["name"]
        if fname.endswith(".csv"):
            df = read_csv(file["download_url"], **read_csv_kwargs[fname])
            data[fname.split(".")[0]] = df

    # Return the data, if only one file is found return the dataframe, otherwise return
    # a dictionary with the dataframes
    if len(data) == 1:
        return list(data.values())[0]
    elif len(data) > 1:
        return data
    else:
        raise Exception(
            f"No csv files found in the folder {name}. Check the pastas-data repository "
            "on GitHub for available datasets."
        )


@lru_cache
def list_datasets(silent: bool = True) -> List[str]:
    """Print a list of available datasets in the pastas-data repository on GitHub.

    Returns
    -------
    list[str]
        A list of available datasets in the pastas-data repository on GitHub.
        Prints a list of available datasets in the pastas-data repository on GitHub.

    Examples
    --------
    >>> ps.list_datasets()
    Prints a list of available datasets in the pastas-data repository on GitHub.

    """
    # Try to import requests, if not installed raise error
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The requests package is required to load datasets from the pastas-data "
            "repository. Install requests using 'pip install requests'."
        )

    # Get the folder from the pastas-data repository
    r = requests.get(GITHUB_URL)

    # Check if requests status is okay, otherwise raise error and return status code
    if not r.status_code == 200:
        raise Exception(f"Error: {r.status_code}. Reason: {r.reason}. ")

    # Get information about the files in the folder
    data = []

    # Loop over the files in the folder
    for file in r.json():
        if file["type"] == "dir":
            data.append(file["name"])

    # Print the list of datasets
    if not silent:
        print("Available datasets in the pastas-data repository on GitHub:")
        for folder in data:
            print(f" - {folder}")
        print(
            "Use ps.load_dataset('folder_name') to load a dataset from the repository."
        )
    return data
