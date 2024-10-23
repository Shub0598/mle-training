"""
A module to train the model.

The model can be trained with python train.py data model, where data model are cli input
"""

# A script (train.py) to train the model(s). The script should accept arguments for
# input (dataset) and output folders (model pickles).
import argparse
import os
import pickle

import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from HousePricePrediction.utils import configure_logger
#from utils import *
import tarfile
import urllib
import numpy as np
import pandas as pd
from zlib import crc32
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


def load_housing_data(housing_path: str) -> pd.DataFrame:
    """
    load the data stored in data/raw folder

    Parameters
    ----------
    housing_path: str :
        url link where housing data in present

    Returns
    -------
    pd.DataFrame
    """

    # housing_path --> path enterd by user
    raw_file_location = os.path.join(housing_path, "raw")
    csv_path = os.path.join(raw_file_location, "housing.csv")
    return pd.read_csv(csv_path)


def main():
    """
    Split data into train and test and perform model training

    When this function is called it will create folder named processed folder
    and then save train and test data.Then model training will happen next and
    the trained model will be saved in the folder named model
    """
    import logging

    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger = configure_logger(
        log_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "model_train.log"),
        console=True,   # if the output in console is needed
        log_level="INFO")

    parser = argparse.ArgumentParser(
        description="accepts two argument input (dataset) and output folders (model pickles)"
    )
    parser.add_argument(
        "--input_dataset", default="data", help="location for input dataset"
    )
    parser.add_argument(
        "--model_output_folder", default="model", help="location for output folder"
    )

    args = parser.parse_args()
    input_data = args.input_dataset  # data
    model_folder = args.model_output_folder  # model

    try:
        housing = load_housing_data(input_data)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"{current_time} : load_housing_data function worked as expected")
    except Exception as e:
        logger.exception(
            f"load_housing_data function does not worked as expected.An error occured as {e}",
            exc_info=True,
        )
    # housing.to_csv("./data/raw/housing.csv")
    # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    # housing_labels = train_set["median_house_value"].copy()
    # train_set = train_set.drop(
    #     "median_house_value", axis=1
    # )  # drop labels for training set

    #housing = housing.drop(columns = ["longitude","latitude"])

    #get_feature_names_from_column_transformer(preprocessing)
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    y_train = train_set["median_house_value"].copy()
    X_train = train_set.drop(
        "median_house_value", axis=1
    )

    y_test = test_set["median_house_value"].copy()
    X_test = test_set.drop(
        "median_house_value", axis=1
    )
    def column_ratio(X):
        return X[:, [0]] / X[:, [1]]

    def ratio_name(function_transformer, feature_names_in):
        return ["ratio"] # feature names out

    def ratio_pipeline():
        return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler())

    #cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                            StandardScaler())

    preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
        "households", "median_income"]),
        #("geo", cluster_simil, ["latitude", "longitude"]),
        #("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ("cat", cat_pipeline, ["ocean_proximity"]),
        ],
        remainder=default_num_pipeline) # one column remaining: housing_median_age

    #
    # os.makedirs("./data/processed", exist_ok=True)
    housing_prepared_train = preprocessing.fit_transform(X_train)
    housing_prepared_test = preprocessing.transform(X_test)

    housing_prepared_train = pd.DataFrame(housing_prepared_train,columns = preprocessing.get_feature_names_out())
    housing_prepared_test = pd.DataFrame(housing_prepared_test,columns = preprocessing.get_feature_names_out())

    train = housing_prepared_train
    test = housing_prepared_test

    train["median_house_value"] = y_train
    test["median_house_value"] = y_test

    os.makedirs("./data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared_train, y_train)

    final_model = grid_search.best_estimator_

    os.makedirs(model_folder, exist_ok=True)
    with open(os.path.join(model_folder, "best_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)
        logger.info(f"{current_time} : Model succesfully trained")


if __name__ == "__main__":
    main()
