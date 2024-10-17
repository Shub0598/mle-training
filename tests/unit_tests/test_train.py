import os
import unittest
from unittest.mock import patch

import pandas as pd

# Assuming load_housing_data is in a module named `housing_module`
from HousePricePrediction.train import load_housing_data


@patch("os.path.join")
@patch("pandas.read_csv")
def test_load_housing_data(mock_read_csv, mock_path_join):
    # Mock the behavior of os.path.join to return a file path
    # List of paths for each os.path.join call
    mock_path_join.side_effect = lambda *args: "/".join(args)

    # Create a mock DataFrame to return when pandas.read_csv is called
    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mock_read_csv.return_value = mock_df

    # Call the function
    housing_path = "some/fake/path"
    result = load_housing_data(housing_path)

    # Assert os.path.join was called twice (for "raw" and "housing.csv")
    expected_calls = [
        (housing_path, "raw"),
        ("some/fake/path/raw", "housing.csv")
    ]

    # Check that os.path.join was called with the right arguments in order
    mock_path_join.assert_any_call(*expected_calls[0])  # First call with "raw"
    mock_path_join.assert_any_call(*expected_calls[1])  # Second call with "housing.csv"

    # Assert that pandas.read_csv was called with the correct file path
    mock_read_csv.assert_called_once_with("some/fake/path/raw/housing.csv")

    # Assert the result is the DataFrame we expect
    pd.testing.assert_frame_equal(result, mock_df)

if __name__ == "__main__":
    unittest.main()
