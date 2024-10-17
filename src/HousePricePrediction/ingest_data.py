"""
A module for downloading data and saving into data folder.
"""

import argparse
import os
import tarfile
from datetime import datetime

import pandas as pd
from six.moves import urllib
from HousePricePrediction.utils import configure_logger

# DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
# HOUSING_PATH = os.path.join("datasets", "housing")
# HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url, housing_path):
    """
    Download the data from passed url

    Parameters
    ----------
    housing_url : str
        url where data is present

    housing_path : str
        folder where data is to saved


    Returns
    -------


    """
    raw_file_location = os.path.join(housing_path, "raw")
    os.makedirs(raw_file_location, exist_ok=True)
    tgz_path = os.path.join(raw_file_location, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=raw_file_location)
    housing_tgz.close()


def main():
    """ """

    import logging

    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger = configure_logger(
        log_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ingest_data.log"),
        console=True,
        log_level="INFO")

    parser = argparse.ArgumentParser(
        description="Download data and create training and validation datasets."
    )
    parser.add_argument(
        "--output_folder",
        default="data",
        # nargs='?',
        help="Folder where the output datasets will be saved.",
    )

    args = parser.parse_args()

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    HOUSING_PATH = args.output_folder

    try:
        fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Log message with manually formatted date and time
        logger.info(f"{current_time} - Data Successfully fetched")
    except:
        # logger.info("function didn't working")
        logger.exception("fetch_housing_data function does not worked as expected")


if __name__ == "__main__":
    main()
