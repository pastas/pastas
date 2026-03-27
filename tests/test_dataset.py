from typing import get_args

import pytest
from pandas import DataFrame

from pastas.dataset import DATASET_NAMES, list_datasets, load_dataset

requests = pytest.importorskip("requests")


def test_load_multiple_csv() -> None:
    # Test loading multiple csv files
    try:
        dataset = load_dataset("collenteur_2023")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            pytest.skip(f"Rate limit exceeded: {e}")
        raise
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
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            pytest.skip(f"Rate limit exceeded: {e}")
        raise
    # Add assertions here to verify the output of the function
    # For example, you can check if the output contains certain dataset names
    assert isinstance(datasets_list, list)
    assert all(isinstance(name, str) for name in datasets_list)
    assert set(get_args(DATASET_NAMES)).issubset(set(datasets_list))
