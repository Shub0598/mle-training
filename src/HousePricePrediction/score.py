import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from HousePricePrediction.utils import configure_logger

def main():

    logger = configure_logger(
        log_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "score.log"),
        console=True,
        log_level="INFO")

    parser = argparse.ArgumentParser(
        description="accepts two argument input (dataset) and output folders (model pickles)"
    )
    parser.add_argument(
        "--input_test_dataset", default="data", help="location for input test dataset"
    )
    parser.add_argument(
        "--saved_model_folder", default="model", help="location for saved model folder"
    )

    args = parser.parse_args()
    input_test_data_folder = args.input_test_dataset  # data/processed
    saved_model_folder = (
        args.saved_model_folder
    )  # model.The folder where saved model is present

    model = os.path.join(saved_model_folder, "best_model.pkl")
    # final_model = joblib.load('model/best_model.pkl')
    # print("model location is ", model)
    final_model = joblib.load(model)

    # Load the test data
    test_data_loc = os.path.join(input_test_data_folder, "processed", "test.csv")
    test_data = pd.read_csv(test_data_loc)
    # test_data = pd.read_csv("data/processed/test.csv")
    # print("test_data location is ",test_data_loc)
    X_test_prepared = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"]

    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("final_rmse = ", final_rmse)
    logger.info(f"final RMSE = {final_rmse}")


if __name__ == "__main__":
    main()
