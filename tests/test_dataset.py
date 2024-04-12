import pytest
from pastas.dataset import list_datasets, load_dataset
from pandas import DataFrame


def test_load_single_csv():
    # Test loading a single csv file
    dataset = load_dataset("collenteur_2021")
    assert isinstance(dataset, DataFrame)


def test_load_multiple_csv():
    # Test loading multiple csv files
    dataset = load_dataset("collenteur_2023")
    assert isinstance(dataset, dict)
    assert len(dataset) > 1
    for key, value in dataset.items():
        assert isinstance(value, DataFrame)


def test_invalid_folder_name():
    # Test loading dataset with invalid folder name
    with pytest.raises(Exception):
        load_dataset("invalid_folder_name")


def test_list_datasets():
    # Test listing available datasets
    list_datasets()
    # Add assertions here to verify the output of the function
    # For example, you can check if the output contains certain dataset names
    # assert "collenteur_2021" in list_datasets()
