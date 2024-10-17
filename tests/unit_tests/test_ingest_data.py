import os
import tarfile
import unittest
import urllib
from unittest.mock import MagicMock, patch

# Assuming fetch_housing_data is in a module named `housing_module`
from HousePricePrediction.ingest_data import fetch_housing_data


@patch("os.makedirs")
@patch("os.path.join")
@patch("urllib.request.urlretrieve")
@patch("tarfile.open")
def test_fetch_housing_data(
    mock_tarfile_open, mock_urlretrieve, mock_path_join, mock_makedirs
):
    # Mock the behavior of os.path.join to return the correct file paths
    mock_path_join.side_effect = lambda *args: "/".join(args)

    # Mock the behavior of tarfile.open and simulate extracting files
    mock_tarfile = MagicMock()
    mock_tarfile_open.return_value = mock_tarfile

    # Call the function
    housing_url = "http://example.com/housing.tgz"
    housing_path = "some/fake/path"
    fetch_housing_data(housing_url, housing_path)

    # Assert os.makedirs was called with the right path
    mock_makedirs.assert_called_once_with("some/fake/path/raw", exist_ok=True)

    # Assert os.path.join was used correctly
    mock_path_join.assert_any_call(housing_path, "raw")
    mock_path_join.assert_any_call("some/fake/path/raw", "housing.tgz")

    # Assert urllib.request.urlretrieve was called with the correct URL and file path
    mock_urlretrieve.assert_called_once_with(
        housing_url, "some/fake/path/raw/housing.tgz"
    )

    # Assert tarfile.open was called with the correct path
    mock_tarfile_open.assert_called_once_with("some/fake/path/raw/housing.tgz")

    # Assert tarfile.extractall was called
    mock_tarfile.extractall.assert_called_once_with(path="some/fake/path/raw")

    # Assert tarfile.close was called
    mock_tarfile.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
