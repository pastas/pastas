from typing import get_args

import pytest
from pandas import DataFrame
from requests.exceptions import HTTPError

from pastas.dataset import DATASET_NAMES, list_datasets, load_dataset


def test_load_multiple_csv() -> None:
    # Test loading multiple csv files
    try:
        dataset = load_dataset("collenteur_2023")
    except HTTPError as e:
        pytest.skip(f"HTTPError occurred: {e}")
    assert isinstance(dataset, dict)
    assert len(dataset) > 1
    for _, value in dataset.items():
        assert isinstance(value, DataFrame)


def test_invalid_folder_name() -> None:
    # Test loading dataset with invalid folder name
    with pytest.raises(Exception):
        load_dataset("invalid_folder_name")


def test_list_datasets() -> None:
    # Test listing available datasets
    try:
        datasets_list = list_datasets(silent=False)
    except HTTPError as e:
        pytest.skip(f"HTTPError occurred: {e}")
    # Add assertions here to verify the output of the function
    # For example, you can check if the output contains certain dataset names
    assert isinstance(datasets_list, list)
    assert all(isinstance(name, str) for name in datasets_list)
    assert set(datasets_list).issubset(set(DATASET_NAMES.__args__))
