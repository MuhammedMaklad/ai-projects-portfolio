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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2543, "status": "ok", "timestamp": 1724773514240, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="5CkQFIxSwzUE" outputId="03bdcef5-2518-4e5b-ec39-0fa5e2da9c2c"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1724773514241, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1z2vkklcwyII" outputId="50b34643-43c6-461e-908d-479cafbe5519"
# cd /content/drive/MyDrive/cancer classification/Prostate cancer

# %% id="9726d39f"
# **DATA PROCESSING**

import numpy as np # Array Processing
import pandas as pd # Data Processing
import os # Input of Data
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# %% [markdown] id="59b256e6"
# Lets import our main data into the notebook

# %% id="00a9da3c"
data = pd.read_csv("prad_mcspc_mskcc_2020_clinical_data.tsv", delimiter='\t') #tab

# %% id="xHhEhhkbGNDp"
class_names = ["high volume", "low volume"]

# %% [markdown] id="f6ed00c4"
# It is a good habit to take a look at the data first. It gives us a lot of knowledge

# %% colab={"base_uri": "https://localhost:8080/", "height": 498} executionInfo={"elapsed": 28, "status": "ok", "timestamp": 1724773514241, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="6a2a5fba" outputId="27d6f7ea-c91e-49ca-926d-30bf0a99f073"
data.head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1724773514241, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="7c469559" outputId="e69abe37-c1f5-4977-af3a-4677b7252fdc"
data.info()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1724773514241, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="wfHIi89YX64L" outputId="25fa07cb-bea8-45f3-adee-012f8111f6e9"
# prompt: need to know the number of records that conatain null values in it;'s features

null_counts = data.isnull().sum()
print(null_counts)


# %% [markdown] id="85116aa3"
# Seems like `index` and `Patient Id` are unique for every row and will thus deivite the accuracy of our model. So lets remove them

# %% id="09137658"
data.drop(["Study ID", "Patient ID", "Sample ID","MSK Slide ID", "SO comments", "Cancer Type", "Cancer Type Detailed", "Race Category"], axis = 1 , inplace = True)

# %% [markdown] id="3ccc924f"
# And now if we see at our datasets

# %% colab={"base_uri": "https://localhost:8080/", "height": 498} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1724773514241, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="c148e1fc" outputId="0ae37dca-9548-4101-8bc3-b7ad4278fa78"
data.head()

# %% [markdown] id="412003fa"
# Categorical data is harder for the system to compute and thus making it numerical will be benifical. Lets replace the values of our target with numbers

# %% id="W_adRT8c6_tz"
# creating features and label

X = data.drop('Disease volume', axis = 1)
y = data['Disease volume']

# %% id="Ri4_uI9bpp74"
from sklearn.preprocessing import LabelEncoder

# Assuming data['Disease volume'] contains the labels you want to convert
label_encoder = LabelEncoder()

# Fit and transform the labels to 0 and 1
y = label_encoder.fit_transform(y)

# %% id="bd3d1f5b"
import pandas as pd

# Identify object-type columns in the DataFrame
object_columns = X.select_dtypes(include=['object']).columns

# %% id="uRkyBQ-TnOH9"
# Iterate through object columns and fill NaN values with the most frequent value
for col in object_columns:
  most_frequent_value = X[col].mode()[0]  # Get the most frequent value
  X[col].fillna(most_frequent_value, inplace=True)  # Replace NaN with the most frequent value

# %% colab={"base_uri": "https://localhost:8080/", "height": 498} id="Fb5RTDSfdjbl" executionInfo={"status": "ok", "timestamp": 1724773514241, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="46c73ff5-2145-4670-8211-fa06dc346b12"
X.head()

# %% id="dE1FJA6QnJuD"
# Apply one-hot encoding to object-type columns
X = pd.get_dummies(X, columns=object_columns)

# %% colab={"base_uri": "https://localhost:8080/"} id="D6IqJ0cid04B" executionInfo={"status": "ok", "timestamp": 1724773514747, "user_tz": -180, "elapsed": 25, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d4a918fe-949b-4811-8997-c84dc4fa4e1c"
print(X.isnull().sum())

# %% colab={"base_uri": "https://localhost:8080/", "height": 342} id="TXjSmuUHeEz_" executionInfo={"status": "ok", "timestamp": 1724773514747, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="98c82c4e-4da3-4cc3-b00d-6a572c47c7d4"
X.head()

# %% id="xKq3CG0nIwl4"
# Fill NaN values with the mean of each column
X = X.fillna(X.mean())

# %% colab={"base_uri": "https://localhost:8080/"} id="-uU4UcFFnqoJ" executionInfo={"status": "ok", "timestamp": 1724773514747, "user_tz": -180, "elapsed": 22, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="dfa0243c-c4e3-4ac0-be0d-36d66730b9de"
null_counts = X.isnull().sum()
print(null_counts)

# %% [markdown] id="cd5aeb7e"
# Now lets divide our datasets into train and test data

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1724773514747, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1zKsKri87-h7" outputId="8d1e3616-185a-4fa3-fcbb-bf42cd9283d4"
print(X.shape)
print(y.shape)

# %% id="ipyXM96Bf7i3"
# prompt: aply kbest with 50 features with ch2

from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=70)
X_new = selector.fit_transform(X, y)

# %% id="xqaYc2CvzXC6"
# scaling data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X_new)

# %% id="68a5d16e"
# splitting data into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1724773514748, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="6Zl-K88CAHlf" outputId="a7580f31-06d0-4483-80b4-acb1ee3d7963"
# need to know the number of instance s per every class with X_train and y_train
from collections import Counter

# Count the number of instances per class
class_counts = Counter(y)

# Print the class counts
print(class_counts)


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1724773514748, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Z1ZXcuO6Unu_" outputId="9d34cc2b-fb33-4df2-831c-cef948ab15d2"
print(X_train.shape)
print(X_test.shape)

# %% [markdown] id="_B1RZuvS1GYN"
# # Support Vector Classifier (SVC)

# %% id="FB0vu81T1Het" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724773516127, "user_tz": -180, "elapsed": 1394, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d5d93d12-af95-420c-86b7-1f606eedca95"
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC()
parameters = {
    'gamma' : [0.0001, 0.001, 0.01, 0.1],
    'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_searchSVM = GridSearchCV(svc, parameters, cv=5)
grid_searchSVM.fit(X_train, y_train)
# best parameters

grid_searchSVM.best_params_

# %% id="PPvxI8vI1Jic" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724773516128, "user_tz": -180, "elapsed": 6, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ea21c67d-3847-41b9-8d27-20cfd0b947cb"
# model predictions
best_SVM = grid_searchSVM.best_estimator_
y_pred = best_SVM.predict(X_test)
# accuracy score

svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)

# %% id="CMnLjhwT1VLl" colab={"base_uri": "https://localhost:8080/", "height": 397} executionInfo={"status": "ok", "timestamp": 1724773534816, "user_tz": -180, "elapsed": 906, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="cf23e7d7-7784-4e98-90d7-2cebb1c258b9"
from sklearn import metrics
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

# %% id="G1I6h1-I1XVt" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724773540710, "user_tz": -180, "elapsed": 410, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="30f5aa12-6f3d-4759-e95f-a3f06d37c86e"
# classification report

print(classification_report(y_test, y_pred, digits=4))


# %% id="UMqd5loUBAyy" colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"status": "ok", "timestamp": 1724773552431, "user_tz": -180, "elapsed": 466, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a96c2fde-86b7-44dd-ceca-bd381e0823cd"
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
plt.title('SVC  (ROC) Curve for prostate cancer')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="N5exg45K1xUY"
# # Decision Tree Classifier

# %% id="cjGNFkEk1z2E" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724773571502, "user_tz": -180, "elapsed": 10619, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d05fc8dd-038b-4a79-bba4-4c72604597db"
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

# %% id="JaaGBfjS10EO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1724773571502, "user_tz": -180, "elapsed": 5, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="6152011e-ebac-4c25-b996-4904acbc84d2"
best_dt = grid_search_dt.best_estimator_
y_pred = best_dt.predict(X_test)
# accuracy score

dtc_acc = accuracy_score(y_test, y_pred)
print(dtc_acc)

# %% id="tjTAJ3UaYg03" colab={"base_uri": "https://localhost:8080/", "height": 392} executionInfo={"status": "ok", "timestamp": 1724773582989, "user_tz": -180, "elapsed": 934, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e28ad69e-018f-4a46-bddd-08240e8bdeb8"
# confusion matrix
from sklearn import metrics
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

# %% id="RHna6avuBEmh" colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"status": "ok", "timestamp": 1724773587136, "user_tz": -180, "elapsed": 935, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ed316667-0511-464a-ca09-e18ddfa2815d"
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
plt.title('Decision Tree  (ROC) Curve for prostate cancer')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="Yy47W_qR2OTc"
# # Random Forest Classifier

# %% colab={"base_uri": "https://localhost:8080/"} id="HqwgSp7q10Fe" executionInfo={"status": "ok", "timestamp": 1724773617179, "user_tz": -180, "elapsed": 26246, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="72e3dff2-98a7-4aeb-dabe-1762e420eeae"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 150, 200],          # Number of trees in the forest
    'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
    'max_depth': [10, 11, 12],                 # Maximum number of levels in tree
    'criterion': ['gini', 'entropy']           # Function to measure the quality of a split
}

# Create a RandomForestClassifier instance
rand_clf = RandomForestClassifier(random_state=42)

# Instantiate GridSearchCV with the RandomForestClassifier, parameter grid, and cross-validation
grid_search = GridSearchCV(estimator=rand_clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', error_score=0)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best accuracy score
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

# %% colab={"base_uri": "https://localhost:8080/"} id="YM2z8y202UWa" executionInfo={"status": "ok", "timestamp": 1724773617179, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="acd50d4e-6099-445a-8d5f-1902e2da615f"
best_rand_forest=grid_search.best_estimator_

y_pred = best_rand_forest.predict(X_test)
# accuracy score

ran_clf_acc = accuracy_score(y_test, y_pred)
print(ran_clf_acc)

# %% colab={"base_uri": "https://localhost:8080/", "height": 397} id="rZMENLddYqdY" executionInfo={"status": "ok", "timestamp": 1723132203587, "user_tz": -180, "elapsed": 634, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="90289364-4560-4d99-d4d9-185c4afee1d4"
# confusion matrix
from sklearn import metrics
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} id="rOVm9i5zBOaJ" executionInfo={"status": "ok", "timestamp": 1723132203587, "user_tz": -180, "elapsed": 10, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="4f29fa8b-bc2e-4e97-cc97-bceb337b9854"
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
plt.title('Random Forest  (ROC) Curve for prostate cancer')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="j4f-sCDz3xOk"
# # Extreme Gradient Boosting

# %% colab={"base_uri": "https://localhost:8080/"} id="b2ujQ2S13xgk" executionInfo={"status": "ok", "timestamp": 1724773638745, "user_tz": -180, "elapsed": 12529, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="daa2412b-c781-4a7d-ac93-db64255e83f1"
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
xgb = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', error_score=0)

# Perform grid search on the training data
xgb.fit(X_train, y_train)

# Get the best parameters and the best accuracy score
best_params = grid_search.best_params_


print("Best Parameters: ", best_params)

# %% colab={"base_uri": "https://localhost:8080/", "height": 409} id="GH2ege0T3x2G" executionInfo={"status": "ok", "timestamp": 1724773638745, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8329f848-cf41-4b98-e2b9-eb2027efdf45"
best_xgb=xgb.best_estimator_
y_pred = best_xgb.predict(X_test)
# accuracy score

xgb_acc = accuracy_score(y_test, y_pred)
print(xgb_acc)

# confusion matrix
from sklearn import metrics
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1723132218245, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="leVFZsR8BSGo" outputId="3d67aa23-8556-4958-933e-e373db4cad41"
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
plt.title('XGboost  (ROC) Curve for prostate cancer')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="n622khM74K1X"
# # camparison

# %% colab={"base_uri": "https://localhost:8080/", "height": 174} executionInfo={"elapsed": 326, "status": "ok", "timestamp": 1724773665857, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ox1bM_bt3x4z" outputId="550773ce-1301-44ec-98e7-023102652dd0"
models = pd.DataFrame({
    'Model': ['SVC',  'Decision Tree Classifier', 'Random Forest Classifier',
              'XgBoost'],
    'Score': [ svc_acc, dtc_acc, ran_clf_acc, xgb_acc]
})

models.sort_values(by = 'Score', ascending = False)
