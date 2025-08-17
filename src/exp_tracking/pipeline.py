import os
import warnings
import sys
from typing import Union
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import dagshub
import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return (rmse, mae, r2)

def train_model(x_train, y_train, alpha, l1_ratio):
    net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    net = net.fit(X=x_train, y=y_train)
    return net

def make_prediction(model: ElasticNet, X):
    pred = model.predict(X)
    return pred


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # Read the wine-quality csv from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your intenet connection. Error: %s", e
        )
    # Split the data into training and test sets. (.75, 025) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is scaler from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_x = test.drop(["quality"], axis=1)
    test_y = test[["quality"]]

    # Perform training

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():

        model = train_model(train_x, train_y, alpha, l1_ratio)
        pred = make_prediction(model=model, X=test_x)
        metrics = eval_metrics(test_y, pred)

        (rmse, mae, r2) = metrics

        print("Elastic model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("   RMSE: %s" % rmse)
        print("   MAE: %s" % mae)
        print("   R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("rmse", rmse)
        mlflow.log_param("mae", mae)
        mlflow.log_param("r2", r2)


