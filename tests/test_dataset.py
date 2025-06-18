import pytest
from pandas import DataFrame

from pastas.dataset import list_datasets, load_dataset


def test_load_multiple_csv() -> None:
    # Test loading multiple csv files
    dataset = load_dataset("collenteur_2023")
    assert isinstance(dataset, dict)
    assert len(dataset) > 1
    for key, value in dataset.items():
        assert isinstance(value, DataFrame)


def test_invalid_folder_name() -> None:
    # Test loading dataset with invalid folder name
    with pytest.raises(Exception):
        load_dataset("invalid_folder_name")


def test_list_datasets() -> None:
    # Test listing available datasets
    list_datasets(silent=False)
    # Add assertions here to verify the output of the function
    # For example, you can check if the output contains certain dataset names
    # assert "collenteur_2021" in list_datasets()
