"""This module contains functions to load datasets from the pastas-data repository on
GitHub. The datasets are used for testing and examples in the documentation. The
load_dataset function can be used to load a single csv file or multiple csv files from
a subfolder in the pastas-data repository.

"""

from pandas import read_csv, DataFrame
from typing import Union, Dict

GITHUB_URL = "https://api.github.com/repos/pastas/pastas-data/contents/"


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
    Exception: If the request status code is not 200.

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
    r = requests.api.get(f"{GITHUB_URL}/{name}/")

    # Check if requests status is okay, otherwise raise error and return status code
    if not r.status_code == 200:
        raise Exception(f"Error: {r.status_code}. Reason: {r.reason}. ")

    # Get information about the files in the folder
    data = {}

    # Loop over the files in the folder
    for file in r.json():
        if file["name"].endswith(".csv"):
            # Read file
            df = read_csv(file["download_url"], index_col=0, parse_dates=True)
            data[file["name"].split(".")[0]] = df

    # Return the data, if only one file is found return the dataframe, otherwise return
    # a dictionary with the dataframes
    if len(data) == 1:
        return list(data.values())[0]
    elif len(data) > 1:
        return data
