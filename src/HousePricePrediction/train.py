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
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing_labels = train_set["median_house_value"].copy()
    train_set = train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set

    imputer = SimpleImputer(strategy="median")

    housing_num = train_set.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = train_set[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    train_set = housing_prepared

    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))
    X_test_prepared["median_house_value"] = y_test
    # X_test_prepared = X_test_prepared.drop("median_house_value", axis=1)
    os.makedirs("./data/processed", exist_ok=True)
    train_set.to_csv("data/processed/train.csv", index=False)
    X_test_prepared.to_csv("data/processed/test.csv", index=False)

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
    grid_search.fit(housing_prepared, housing_labels)

    final_model = grid_search.best_estimator_

    os.makedirs(model_folder, exist_ok=True)
    with open(os.path.join(model_folder, "best_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)
        logger.info(f"{current_time} : Model succesfully trained")


if __name__ == "__main__":
    main()
