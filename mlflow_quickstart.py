import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### Set the tracking URI ###
mlflow.set_tracking_uri(uri="http://localhost:8080")

### Train a model and prepare metadata for logging ###
#Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split()

