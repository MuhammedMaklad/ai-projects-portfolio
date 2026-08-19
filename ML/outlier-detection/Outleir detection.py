# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% id="3907Gffdh-V4"
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# %% id="Y0QMygXzyeeE" executionInfo={"status": "ok", "timestamp": 1723275775629, "user_tz": -180, "elapsed": 9, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/"} outputId="5979424e-570e-43fc-b37e-8388d0fb30d1"
# Load and prepare the dataset
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names
print(feature_names)

# %% colab={"base_uri": "https://localhost:8080/"} id="lBXKxnDJNGme" executionInfo={"status": "ok", "timestamp": 1723275775629, "user_tz": -180, "elapsed": 7, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a52d1b26-04c4-4ec9-dafe-06cb4c7e6016"
y

# %% colab={"base_uri": "https://localhost:8080/"} id="FspiQIj1MwoP" executionInfo={"status": "ok", "timestamp": 1723275775629, "user_tz": -180, "elapsed": 5, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="26d9c9ac-91cb-4c2d-c0e8-fab35b9a69dc"
print(X.shape, "   ", y.shape)

# %% id="XBc1NNqfMri2"
# Standardize the dataset / Z score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# %% id="pNJIWCNMycu9"
# Function to detect and remove outliers using Z-Score
def remove_outliers_zscore(X, threshold=3):
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], mask


# %% id="FtA4NNvsyVcr"
# Function to detect and remove outliers using IQR
def remove_outliers_iqr(X):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    mask = ((X >= (Q1 - 1.5 * IQR)) & (X <= (Q3 + 1.5 * IQR))).all(axis=1)
    return X[mask], mask


# %% id="FAD5VwndyXCk"
# Function to detect and remove outliers using Isolation Forest
def remove_outliers_isolation_forest(X):
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X)
    mask = yhat != -1
    return X[mask], mask


# %% id="q2bl1Fo0yYcW"
# Function to detect and remove outliers using Local Outlier Factor
def remove_outliers_lof(X):
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X)
    mask = yhat != -1
    return X[mask], mask


# %% id="MJr7OYZNyZyT"
# Function to detect and remove outliers using Elliptic Envelope
def remove_outliers_elliptic_envelope(X):
    envelope = EllipticEnvelope(contamination=0.1)
    yhat = envelope.fit_predict(X)
    mask = yhat != -1
    return X[mask], mask



# %% id="LpyqXx7dyRU3"
# List of outlier detection methods
outlier_methods = {
    'Z-Score': remove_outliers_zscore,
    'IQR': remove_outliers_iqr,
    'Isolation Forest': remove_outliers_isolation_forest,
    'LOF': remove_outliers_lof,
    'Elliptic Envelope': remove_outliers_elliptic_envelope
}


# %% colab={"base_uri": "https://localhost:8080/"} id="ew7Gza6qyPAf" executionInfo={"status": "ok", "timestamp": 1723275777231, "user_tz": -180, "elapsed": 1605, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="af0bf48b-215f-473d-c8a8-c27013939e96"
# Function to train and evaluate the model
def train_evaluate(X_train, X_test, y_train, y_test, model_name):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print(f"\n{model_name} - Accuracy: {accuracy}\n")
    print(f"{model_name} - Classification Report:\n{report}\n")

# Apply outlier detection methods and evaluate the model
for method_name, method_func in outlier_methods.items():
    X_filtered, mask = method_func(X_scaled)
    y_filtered = y[mask]
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)
    train_evaluate(X_train, X_test, y_train, y_test, method_name)
