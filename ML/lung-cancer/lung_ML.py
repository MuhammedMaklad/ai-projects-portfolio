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
#     language: python
#     name: python3
# ---

# %% id="5CkQFIxSwzUE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777614452, "user_tz": -180, "elapsed": 3982, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="03d4e0c0-3b2d-4c1e-9d6e-8a0d50662a60"
from google.colab import drive
drive.mount('/content/drive')


# %% id="1z2vkklcwyII" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777614452, "user_tz": -180, "elapsed": 5, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="761f0390-526e-4027-d6f2-54d6a963d5ef"
# cd /content/drive/MyDrive/cancer classification/lung cancer

# %% id="9726d39f"
# **DATA PROCESSING**

import numpy as np # Array Processing
import pandas as pd # Data Processing
import os # Input of Data
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# %% [markdown] id="59b256e6"
# Lets import our main data into the notebook

# %% id="00a9da3c"
data = pd.read_csv("lung_cancer.csv")

# %% id="okNFirBtFUKU"
class_names = ["No", "Yes"]

# %% [markdown] id="f6ed00c4"
# It is a good habit to take a look at the data first. It gives us a lot of knowledge

# %% id="6a2a5fba" colab={"base_uri": "https://localhost:8080/", "height": 243} executionInfo={"status": "ok", "timestamp": 1724777615010, "user_tz": -180, "elapsed": 562, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="03335c19-bf66-4a37-aa68-f9a70015a7e9"
data.head()

# %% id="7c469559" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777615010, "user_tz": -180, "elapsed": 11, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d4f8c9aa-8a00-470c-fcc5-f8dd1265ca9c"
data.info()

# %% id="8LSny3uz4nWk" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777615010, "user_tz": -180, "elapsed": 10, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a7e2aff7-0cb1-4be2-c680-4a612bf3dae2"
#Remove duplicates in the dataset
print(data.duplicated().sum())
data.drop_duplicates(inplace=True)

# %% [markdown] id="412003fa"
# Categorical data is harder for the system to compute and thus making it numerical will be benifical. Lets replace the values of our target with numbers

# %% id="bd3d1f5b"
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['LUNG_CANCER']=encoder.fit_transform(data['LUNG_CANCER'])
data['GENDER']=encoder.fit_transform(data['GENDER'])

# %% id="4bb9cdaa" colab={"base_uri": "https://localhost:8080/", "height": 243} executionInfo={"status": "ok", "timestamp": 1724777615873, "user_tz": -180, "elapsed": 871, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d36e4017-7751-48f9-f6f8-b7737df5d93a"
data.head()

# %% id="VoGTbHiQtUXq" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777615874, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0019c1a5-ccc0-4559-f6f4-90154c5fbafa"
data.shape

# %% id="KvJ5mlYCUBbL" colab={"base_uri": "https://localhost:8080/", "height": 178} executionInfo={"status": "ok", "timestamp": 1724777615874, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e30612d9-2cda-4e3e-82c2-112a28bcd77d"
data.LUNG_CANCER.value_counts()

# %% [markdown] id="cd5aeb7e"
# Now lets divide our datasets into train and test data

# %% id="W_adRT8c6_tz"
# creating features and label

X = data.drop('LUNG_CANCER', axis = 1)
y = data['LUNG_CANCER']

# %% id="1zKsKri87-h7" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777615874, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="2acdca31-91f7-476c-abfa-7b0f612c2b62"
print(X.shape)
print(y.shape)

# %% id="68a5d16e"
# splitting data into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44)

# %% id="ymp7TrhAt7Bz" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777615874, "user_tz": -180, "elapsed": 11, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="de493b16-3aa8-4813-b2e8-c1218fb560c5"
print(X_train.shape, "    " , X_test.shape)
print(y_train.shape, "    " , y_test.shape)


# %% id="RI7sYRU7vKNn" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777615874, "user_tz": -180, "elapsed": 11, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="2c1bd2dc-b406-418c-8fca-db579371ba27"
from collections import Counter
print(Counter(y_train))

# %% id="dmn9ilfR4O6m"
#Random data shuffle
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state=42)
X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

# %% id="3CpHI_7B9LJP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777616710, "user_tz": -180, "elapsed": 5, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d2dbef1f-8944-4a3b-dc5d-6c56d6206b8d"
print(X_train.shape)
print(X_test.shape)

# %% id="rtoAnJ0uu_TN" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777616710, "user_tz": -180, "elapsed": 4, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="5f3b34e1-d448-4834-8230-2134249dc4b7"
# Need to know the number of values for every class after oversampling

from collections import Counter
print(Counter(y_train))


# %% [markdown] id="_B1RZuvS1GYN"
# # Support Vector Classifier (SVC)

# %% id="FB0vu81T1Het" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777624680, "user_tz": -180, "elapsed": 7973, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="12c26e31-6867-47bc-f477-a0e3b9392fcc"
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC()
parameters = {
    'gamma' : [0.0001, 0.001, 0.01, 0.1],
    'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_search = GridSearchCV(svc, parameters)
grid_search.fit(X_train, y_train)
# best parameters

grid_search.best_params_

# %% id="6QZKfK5ovmbD"
best_svc=grid_search.best_estimator_

# %% id="PPvxI8vI1Jic" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777624680, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a083b0d1-65b5-498c-8849-53c139d7b543"
# model predictions

y_pred = best_svc.predict(X_test)
# accuracy score

svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)

# %% id="CMnLjhwT1VLl" colab={"base_uri": "https://localhost:8080/", "height": 397} executionInfo={"status": "ok", "timestamp": 1724777624680, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="adbc6d9a-a110-4122-a272-aeecb0ce309e"
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
class_names = ["No", "Yes"]

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(5,4)) # Adjust figure size as needed
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 15},
            xticklabels=class_names, yticklabels=class_names) # Add class names here
plt.title('', size=24) # Adjust title size as needed
plt.xlabel('Predicted label', size=12) # Adjust label size as needed
plt.ylabel('True label', size=12) # Adjust label size as needed
plt.xticks(rotation=0) # Optional: Rotate x-axis labels if needed
plt.yticks(rotation=90) # Optional: Rotate y-axis labels if needed
plt.show()

# %% id="G1I6h1-I1XVt" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777624680, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="65bdb46c-2833-4744-e238-5a29ffdaca51"
# classification report

print(classification_report(y_test, y_pred,target_names=class_names, digits=4))

# %% id="zdIk2Vbp-QjC" colab={"base_uri": "https://localhost:8080/", "height": 564} executionInfo={"status": "ok", "timestamp": 1724777625678, "user_tz": -180, "elapsed": 1015, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="706252d4-6f47-48f3-e415-cac9c2e1c443"
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVC  (ROC) Curve for lung cancer')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="N5exg45K1xUY"
# # Decision Tree Classifier

# %% id="cjGNFkEk1z2E" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777637559, "user_tz": -180, "elapsed": 11889, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="94072782-7751-4316-b442-ce71726ab38a"
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

parameters = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(2, 32, 5),
    'min_samples_leaf' : range(1, 10, 3),
    'min_samples_split' : range(2, 10, 3),
    'splitter' : ['best', 'random']
}

grid_search_dt = GridSearchCV(dtc, parameters, cv = 5, n_jobs = -1, verbose = 1)
grid_search_dt.fit(X_train, y_train)
print(grid_search_dt.best_params_)

# %% id="JaaGBfjS10EO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777637560, "user_tz": -180, "elapsed": 6, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="43bc6ddb-a7bc-4b96-f210-e2297690f1b7"
best_dtc = grid_search_dt.best_estimator_
y_pred = best_dtc.predict(X_test)
# accuracy score

dtc_acc = accuracy_score(y_test, y_pred)
print(dtc_acc)
# classification report


print(classification_report(y_test, y_pred,target_names=class_names, digits=4))

# %% id="_b3hO74V3pe_" colab={"base_uri": "https://localhost:8080/", "height": 392} executionInfo={"status": "ok", "timestamp": 1724777638158, "user_tz": -180, "elapsed": 603, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="1ead788a-3d11-4ead-9876-961561a3e795"
from sklearn import metrics
class_names = ["Benign", "Malignant"]

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(5,4)) # Adjust figure size as needed
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 15},
            xticklabels=class_names, yticklabels=class_names) # Add class names here
plt.title('', size=24) # Adjust title size as needed
plt.xlabel('Predicted label', size=12) # Adjust label size as needed
plt.ylabel('True label', size=12) # Adjust label size as needed
plt.xticks(rotation=0) # Optional: Rotate x-axis labels if needed
plt.yticks(rotation=90) # Optional: Rotate y-axis labels if needed
plt.show()

# %% id="PpWmISXg-Yc5" colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"status": "ok", "timestamp": 1724777638158, "user_tz": -180, "elapsed": 10, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="cc4365a9-e169-4910-f079-02ac789b4734"
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree (ROC) Curve for lung cancer')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="Yy47W_qR2OTc"
# # Random Forest Classifier

# %% id="HqwgSp7q10Fe" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777662040, "user_tz": -180, "elapsed": 23892, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8e7a384f-d036-4d3b-ce0b-c6b88a366319"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 150, 200],          # Number of trees in the forest
    'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
    'max_depth': [10, 11, 12],                  # Minimum number of samples required to be at a leaf node
    'criterion': ['gini', 'entropy']           # Function to measure the quality of a split
}

# Create a RandomForestClassifier instance
rand_clf = RandomForestClassifier(random_state=42)

# Instantiate GridSearchCV with the RandomForestClassifier, parameter grid, and cross-validation
grid_search = GridSearchCV(estimator=rand_clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', error_score=0)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best accuracy score
best_params = grid_search.best_params_  #Dictionary
best_accuracy = grid_search.best_score_

best_rand_clf=grid_search.best_estimator_

print("Best Parameters: ", best_params)

# %% id="YM2z8y202UWa" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777662041, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="665fbd21-9c40-40c7-e7d2-9de101b1f067"
y_pred = best_rand_clf.predict(X_test)
# accuracy score

ran_clf_acc = accuracy_score(y_test, y_pred)
print(ran_clf_acc)

# classification report

print(classification_report(y_test, y_pred,target_names=class_names, digits=4))

# %% id="B_DvbFiL3vnr" colab={"base_uri": "https://localhost:8080/", "height": 392} executionInfo={"status": "ok", "timestamp": 1724777662888, "user_tz": -180, "elapsed": 858, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="863675b8-970e-4d50-f583-7eb99e46c8b4"
from sklearn import metrics
class_names = ["No", "Yes"]

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(5,4)) # Adjust figure size as needed
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 15},
            xticklabels=class_names, yticklabels=class_names) # Add class names here
plt.title('', size=24) # Adjust title size as needed
plt.xlabel('Predicted label', size=12) # Adjust label size as needed
plt.ylabel('True label', size=12) # Adjust label size as needed
plt.xticks(rotation=0) # Optional: Rotate x-axis labels if needed
plt.yticks(rotation=90) # Optional: Rotate y-axis labels if needed
plt.show()

# %% id="kAitp0Jn-clY" colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"status": "ok", "timestamp": 1724777663725, "user_tz": -180, "elapsed": 842, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b4bcf546-bbc6-4443-aa3e-7c5a4ee5b592"
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest  (ROC) Curve for lung cancer')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="j4f-sCDz3xOk"
# # Extreme Gradient Boosting

# %% id="b2ujQ2S13xgk" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777673783, "user_tz": -180, "elapsed": 10060, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="7408a164-b599-40fe-b49e-699a12b01e55"
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'learning_rate': [0.1, 0.5, 1.0],  # Learning rate
    'max_depth': [3, 4, 5],            # Maximum depth of the tree
    'n_estimators': [150, 180, 200]     # Number of boosting rounds
}

# Create an XGBClassifier instance
xgb = XGBClassifier(objective='binary:logistic', random_state=42)

# Instantiate GridSearchCV with the XGBClassifier, parameter grid, and cross-validation
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', error_score=0)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best accuracy score
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Train the model with the best parameters
best_xgb = XGBClassifier(**best_params, objective='binary:logistic', random_state=42)
best_xgb.fit(X_train, y_train)


print("Best Parameters: ", best_params)

# %% id="GH2ege0T3x2G" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724777673783, "user_tz": -180, "elapsed": 12, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="655a17bc-9c19-4e07-d59f-6d43fe60b355"
y_pred = best_xgb.predict(X_test)
# accuracy score

xgb_acc = accuracy_score(y_test, y_pred)
print(xgb_acc)

# classification report

print(classification_report(y_test, y_pred,target_names=class_names, digits=4))

# %% id="gzOgp-4D3yGR" colab={"base_uri": "https://localhost:8080/", "height": 397} executionInfo={"status": "ok", "timestamp": 1724777674374, "user_tz": -180, "elapsed": 602, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e7f0b14e-bc39-4a21-e9c7-e40f4980f04b"
from sklearn import metrics
class_names = ["Benign", "Malignant"]

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(5,4)) # Adjust figure size as needed
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 15},
            xticklabels=class_names, yticklabels=class_names) # Add class names here
plt.title('', size=24) # Adjust title size as needed
plt.xlabel('Predicted label', size=12) # Adjust label size as needed
plt.ylabel('True label', size=12) # Adjust label size as needed
plt.xticks(rotation=0) # Optional: Rotate x-axis labels if needed
plt.yticks(rotation=90) # Optional: Rotate y-axis labels if needed
plt.show()

# %% id="qc3OlMb8-giA" colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"status": "ok", "timestamp": 1724777674374, "user_tz": -180, "elapsed": 11, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c71d69b7-2fd4-40b0-c8a2-33c51e8c76dc"
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost  (ROC) Curve for lung cancer')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="n622khM74K1X"
# # camparison

# %% colab={"base_uri": "https://localhost:8080/", "height": 174} id="ox1bM_bt3x4z" outputId="d8d4a146-fe1e-43c7-f38e-375d42a13566" executionInfo={"status": "ok", "timestamp": 1724777674697, "user_tz": -180, "elapsed": 333, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
models = pd.DataFrame({
    'Model': ['SVC',  'Decision Tree Classifier', 'Random Forest Classifier',
              'XgBoost'],
    'Score': [ svc_acc, dtc_acc, ran_clf_acc, xgb_acc]
})

models.sort_values(by = 'Score', ascending = False)
