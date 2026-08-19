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

# %% id="AB3zPdpE-cGS" executionInfo={"status": "ok", "timestamp": 1726817582581, "user_tz": -180, "elapsed": 3635, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN
from collections import Counter

# %% colab={"base_uri": "https://localhost:8080/"} id="1OOh4BfCjrTg" executionInfo={"status": "ok", "timestamp": 1726817583510, "user_tz": -180, "elapsed": 930, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c90edf06-36bb-42b0-8445-773b6ab7b32e"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/lung cancer

# %% id="6-3PT7rCjnMn" executionInfo={"status": "ok", "timestamp": 1726817584685, "user_tz": -180, "elapsed": 1177, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data = pd.read_csv("lung_cancer.csv")

# %% id="CrVuZmOckAYC" executionInfo={"status": "ok", "timestamp": 1726817584685, "user_tz": -180, "elapsed": 4, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['LUNG_CANCER']=encoder.fit_transform(data['LUNG_CANCER'])
data['GENDER']=encoder.fit_transform(data['GENDER'])

# %% id="D70M8GEKj1VD" executionInfo={"status": "ok", "timestamp": 1726817584685, "user_tz": -180, "elapsed": 4, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X=data.drop('LUNG_CANCER',axis=1)
y=data['LUNG_CANCER']

# %% id="p3l9-6ho_uDX" executionInfo={"status": "ok", "timestamp": 1726817584685, "user_tz": -180, "elapsed": 3, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% colab={"base_uri": "https://localhost:8080/"} id="q7DKC8zpjKE-" executionInfo={"status": "ok", "timestamp": 1726817599529, "user_tz": -180, "elapsed": 581, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="adfa5973-d2bd-4918-9d8c-45d5ee675600"
Counter(y_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="35uSvNtnAbj9" executionInfo={"status": "ok", "timestamp": 1726817653158, "user_tz": -180, "elapsed": 409, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="2a8a3ab9-047a-4786-c839-5e209655750d"
# Performing Logistic Regression on the original imbalanced dataset
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Evaluating the performance
print("Logistic Regression on Original Data:")
print(classification_report(y_test, y_pred_lr, digits=4))


# %% colab={"base_uri": "https://localhost:8080/"} id="FZ45su1u_-26" executionInfo={"status": "ok", "timestamp": 1726817735116, "user_tz": -180, "elapsed": 501, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f064d290-37a5-4aa0-e772-257aaeb72af9"
# 1. Random Oversampling
ros = RandomOverSampler(random_state=42)
X_res_ros, y_res_ros = ros.fit_resample(X_train, y_train)
print(Counter(y_res_ros))

model_ros = LogisticRegression(random_state=42)
model_ros.fit(X_res_ros, y_res_ros)
y_pred_ros = model_ros.predict(X_test)
print("Random Oversampling:")
print(classification_report(y_test, y_pred_ros, digits=4))

# %% colab={"base_uri": "https://localhost:8080/"} id="J41SdzpX_fYe" executionInfo={"status": "ok", "timestamp": 1726817772004, "user_tz": -180, "elapsed": 357, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="cf177d09-2af6-4ba1-f52c-358e2579d888"
# 2. Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_res_rus, y_res_rus = rus.fit_resample(X_train, y_train)
print(Counter(y_res_rus))
model_rus = LogisticRegression(random_state=42)
model_rus.fit(X_res_rus, y_res_rus)
y_pred_rus = model_rus.predict(X_test)
print("Random Undersampling:")
print(classification_report(y_test, y_pred_rus))

# %% colab={"base_uri": "https://localhost:8080/"} id="E3foN8yC_hQo" executionInfo={"status": "ok", "timestamp": 1726817796638, "user_tz": -180, "elapsed": 360, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="fa40cb01-37e0-465b-f74e-83715429fb3a"
# 3. SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_res_smote, y_res_smote = smote.fit_resample(X_train, y_train)
print(Counter(y_res_smote))
model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_res_smote, y_res_smote)
y_pred_smote = model_smote.predict(X_test)
print("SMOTE:")
print(classification_report(y_test, y_pred_smote, digits=4))

# %% colab={"base_uri": "https://localhost:8080/"} id="WAcjYjdq_ZDN" executionInfo={"status": "ok", "timestamp": 1726818057616, "user_tz": -180, "elapsed": 390, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="88b0582a-fe24-4da1-9d4f-7f2624887cb0"
# Applying SMOTEENN (SMOTE + Edited Nearest Neighbors)
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# Display resampled class distribution
print(f"Resampled class distribution: {Counter(y_resampled)}")

# Training a logistic regression model on the resampled dataset
model = LogisticRegression(random_state=42)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# Evaluating the model
print("SMOTEENN Results:")
print(classification_report(y_test, y_pred, digits=4))

# %% colab={"base_uri": "https://localhost:8080/"} id="QfrkDQK8nZZM" executionInfo={"status": "ok", "timestamp": 1726818068974, "user_tz": -180, "elapsed": 394, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="5b5d53fb-cb92-4202-d293-3f447ad8b5e0"
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# Display resampled class distribution
print(f"Resampled class distribution: {Counter(y_resampled)}")

# Training a logistic regression model on the resampled dataset
model = LogisticRegression(random_state=42)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# Evaluating the model
print("SMOTETomek Results:")
print(classification_report(y_test, y_pred, digits=4))

# %% colab={"base_uri": "https://localhost:8080/"} id="ZijZIr5xnx81" executionInfo={"status": "ok", "timestamp": 1726818078012, "user_tz": -180, "elapsed": 587, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0a043bcf-bc0e-4d74-bf21-e3e3997ecc9a"
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
# Display resampled class distribution
print(f"Resampled class distribution: {Counter(y_resampled)}")

# Training a logistic regression model on the resampled dataset
model = LogisticRegression(random_state=42)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# Evaluating the model
print("ADASYN Results:")
print(classification_report(y_test, y_pred, digits=4))
