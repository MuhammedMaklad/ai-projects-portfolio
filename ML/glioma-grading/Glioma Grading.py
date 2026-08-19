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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23375, "status": "ok", "timestamp": 1724779412163, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="5CkQFIxSwzUE" outputId="825ad808-a715-423a-f8a2-ad9c9b9c8965"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2346, "status": "ok", "timestamp": 1724779414506, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1z2vkklcwyII" outputId="b061a591-a592-4f7e-bbbd-17b59e053c69"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/Glioma Grading

# %% id="xxOxn9LyB7PA"
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# %% id="9726d39f"
import numpy as np # Array Processing
import pandas as pd # Data Processing
import os # Input of Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns

# %% [markdown] id="59b256e6"
# Lets import our main data into the notebook

# %% id="00a9da3c"
data = pd.read_csv("dataset.csv")

# %% [markdown] id="f6ed00c4"
# It is a good habit to take a look at the data first. It gives us a lot of knowledge

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 16, "status": "ok", "timestamp": 1724779420498, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="6a2a5fba" outputId="fb950240-8b4c-41b6-8811-f80b14a621d0"
data.head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1724779420498, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="7c469559" outputId="9cce4947-e862-4c51-8174-35e88d32385a"
data.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1724779420498, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="2DOyMGa4n21j" outputId="3b69c99c-2e3b-4c1a-f8b1-04ae1487f7cf"
data.Grade.value_counts(normalize=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="4ZJ43gprjy2W" executionInfo={"status": "ok", "timestamp": 1724779420498, "user_tz": -180, "elapsed": 12, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8e3b2256-4463-49f3-e513-aea6eefab4af"
data.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="paDkvA1bhO9j" executionInfo={"status": "ok", "timestamp": 1723235666284, "user_tz": -180, "elapsed": 7355, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ec500198-e6d2-4c7f-d9ad-7946fe9828bc"
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame with features
correlation_matrix = data.corr()

# Create a heatmap
plt.figure(figsize=(15, 13))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="2GwRkWx_0UcU" executionInfo={"status": "ok", "timestamp": 1724779424259, "user_tz": -180, "elapsed": 6, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="fc8387a3-fb3a-4549-ca3f-9be4d8014e97"
# Display summary statistics of the numerical features
print("\nSummary Statistics:")
print(data.describe())

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} id="dzN6awiO0yPj" executionInfo={"status": "ok", "timestamp": 1724779448544, "user_tz": -180, "elapsed": 832, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="1605c9cb-da9a-41f4-d48b-a5e31fe7775a"
# Visualize the distribution of the target variable 'Grade'
plt.figure(figsize=(5, 5))
sns.countplot(x='Grade', data=data)
plt.title('Distribution of Grades')
plt.show()

# %% id="W_adRT8c6_tz"
# creating features and label

X = data.drop('Grade', axis = 1)
y = data['Grade']

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1724779459666, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1zKsKri87-h7" outputId="7ed1f1c5-0354-44d9-c9d3-5ec2e15bf184"
print(X.shape)
print(y.shape)

# %% id="xqaYc2CvzXC6"
# scaling data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

# %% id="68a5d16e"
# splitting data into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# %% id="l9s4Ws2bB0oz"
from imblearn.over_sampling import RandomOverSampler
# Apply Random Oversampling to the training data
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="d0BMeIldg3cb" executionInfo={"status": "ok", "timestamp": 1700769814700, "user_tz": -120, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="34d537fa-9443-4d79-f0e4-d0dc69e39eba"
X_resampled.shape

# %% [markdown] id="9DGj0i-F0QEQ"
# ## Logistic regression

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 1354, "status": "ok", "timestamp": 1724779523211, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="85UWmsySz7TY" outputId="f1a61c8f-08d0-4b07-8c19-bf6886948b24"
class_names = ["LGG", "GBM"]
# Define the logistic regression model
log_reg = LogisticRegression(solver='liblinear')

# Define hyperparameter grid for grid search
param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}

# Perform grid search
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)

# Get the best model from grid search
best_log_reg = grid_search.best_estimator_

# Fit the best model to the data
best_log_reg.fit(X_train, y_train)

# Make predictions
y_pred = best_log_reg.predict(X_test)

# Accuracy score
log_reg_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", log_reg_acc)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=class_names,digits=4)
print("Classification Report:")
print(class_report)

from sklearn import metrics

cm=confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels=class_names)

cm_display.plot()
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression  (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="5o4e-YRB0dro"
# # K Neighbors Classifier (KNN)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 537, "status": "ok", "timestamp": 1724779538440, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="DHIOBpKP0kL2" outputId="119703fa-7feb-4201-f7fb-c181f8cef08a"
# Define the K-Nearest Neighbors model
knn = KNeighborsClassifier()

# Define hyperparameter grid for grid search
param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}

# Perform grid search
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_knn = grid_search.best_estimator_

# Fit the best model to the data
best_knn.fit(X_resampled, y_resampled)

# Make predictions
y_pred = best_knn.predict(X_test)

# Accuracy score
knn_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", knn_acc)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=class_names,digits=4)
print("Classification Report:")
print(class_report)

from sklearn import metrics

cm=confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels=class_names)

cm_display.plot()
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K-Nearest Neighbors (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="RY83wA-PlXmf"
#

# %% [markdown] id="_B1RZuvS1GYN"
# # Support Vector Classifier (SVC)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 7167, "status": "ok", "timestamp": 1724779546451, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="FB0vu81T1Het" outputId="284435e0-c46f-424a-bdee-596a22637553"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

# Define the Support Vector Classifier model
svc = SVC(probability=True)  # Set probability to True for ROC curve

# Define hyperparameter grid for grid search
param_grid = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf'],
              'gamma': ['scale', 'auto']}

# Perform grid search
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_svc = grid_search.best_estimator_

# Fit the best model to the data
best_svc.fit(X_resampled, y_resampled)

# Make predictions
y_pred = best_svc.predict(X_test)

# Accuracy score
svc_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", svc_acc)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Support Vector Classifier (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# %% [markdown] id="q52WAfoy1b2G"
# # SGD Classifier

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 1019, "status": "ok", "timestamp": 1724779547460, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="qvwHBAoM1cv2" outputId="c17eac15-2a92-459b-ff56-dab0d642bc6e"
# Define the Stochastic Gradient Descent model
sgd = SGDClassifier()

# Define hyperparameter grid for grid search
param_grid = {
    'penalty': ['l1', 'l2'],
    'alpha': [0.0001,0.01, 0.1],
}

# Perform grid search
grid_search = GridSearchCV(sgd, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)

# Get the best model from grid search
best_sgd = grid_search.best_estimator_

# Fit the best model to the data
best_sgd.fit(X_train, y_train)

# Make predictions
y_pred = best_sgd.predict(X_test)

# Accuracy score
sgd_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", sgd_acc)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stochastic Gradient Descent (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="N5exg45K1xUY"
# # Decision Tree Classifier

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 3159, "status": "ok", "timestamp": 1724779550613, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="cjGNFkEk1z2E" outputId="b07f7b77-6fa4-4c25-ce60-6f398b4371ff"
# Define the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

# Define hyperparameter grid for grid search
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(2, 32, 5),
    'min_samples_leaf': range(1, 10, 4),
    'min_samples_split': range(2, 10, 4),
}
# Perform grid search
grid_search_dt = GridSearchCV(dtc, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search_dt.fit(X_train, y_train)

# Get the best model from grid search
best_dtc = grid_search_dt.best_estimator_

# Print the best parameters
print("Best Parameters:", grid_search_dt.best_params_)

# Fit the best model to the data
best_dtc.fit(X_resampled, y_resampled)

# Make predictions
y_pred = best_dtc.predict(X_test)

# Accuracy score
dtc_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", dtc_acc)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree Classifier (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="Yy47W_qR2OTc"
# # Random Forest Classifier

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 47133, "status": "ok", "timestamp": 1724779597741, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="HqwgSp7q10Fe" outputId="48fb1494-20b3-4754-d5f4-9bc5a2c994ff"
from sklearn.ensemble import RandomForestClassifier

# Define the Random Forest Classifier
rfc = RandomForestClassifier()

# Define hyperparameter grid for grid search
parameters = {
    'n_estimators': [50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform grid search
grid_search_rf = GridSearchCV(rfc, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search_rf.fit(X_resampled, y_resampled)

# Get the best model from grid search
best_rfc = grid_search_rf.best_estimator_

# Print the best parameters
print("Best Parameters:", grid_search_rf.best_params_)

# Fit the best model to the data
best_rfc.fit(X_train, y_train)

# Make predictions
y_pred = best_rfc.predict(X_test)

# Accuracy score
rfc_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", rfc_acc)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Classifier (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="jW9QpWiz2rIe"
# # Ada Boost Classifier

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 5152, "status": "ok", "timestamp": 1724779602883, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="_UCHdns42y7a" outputId="13a2120c-4bd5-4bb4-86f3-16250f2f7933"
from sklearn.ensemble import AdaBoostClassifier

# Define the AdaBoost Classifier
adaboost = AdaBoostClassifier()

# Define hyperparameter grid for grid search
parameters = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 5],
}

# Perform grid search
grid_search_adaboost = GridSearchCV(adaboost, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search_adaboost.fit(X_resampled, y_resampled)

# Get the best model from grid search
best_adaboost = grid_search_adaboost.best_estimator_

# Print the best parameters
print("Best Parameters:", grid_search_adaboost.best_params_)

# Fit the best model to the data
best_adaboost.fit(X_train, y_train)

# Make predictions
y_pred = best_adaboost.predict(X_test)

# Accuracy score
adaboost_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", adaboost_acc)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AdaBoost Classifier (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="btGWoWMf29LK"
# # Gradient Boosting Classifier

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="cuTLZVjk29b5" executionInfo={"status": "ok", "timestamp": 1724779635608, "user_tz": -180, "elapsed": 32743, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="953495c9-6b0f-4175-a63a-7315e4922c6b"
from sklearn.ensemble import GradientBoostingClassifier

# Define the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier()

# Define hyperparameter grid for grid search
parameters = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform grid search
grid_search_gb = GridSearchCV(gb_classifier, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search_gb.fit(X_resampled, y_resampled)

# Get the best model from grid search
best_gb = grid_search_gb.best_estimator_

# Print the best parameters
print("Best Parameters:", grid_search_gb.best_params_)

# Fit the best model to the data
best_gb.fit(X_train, y_train)

# Make predictions
y_pred = best_gb.predict(X_test)

# Accuracy score
gb_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", gb_acc)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Gradient Boosting Classifier (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="j4f-sCDz3xOk"
# # Extreme Gradient Boosting

# %% id="71i7SUDQm9Y1"

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 7149, "status": "ok", "timestamp": 1724779642741, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="b2ujQ2S13xgk" outputId="58d14079-3b57-474c-c88a-9dae0a7b6082"
# Define the XGBoost Classifier
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()

# Define hyperparameter grid for grid search
parameters = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100],
    'max_depth': [3, 4, 5],
    'gamma': [0, 1, 2]
}

# Perform grid search
grid_search_xgb = GridSearchCV(xgb_classifier, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search_xgb.fit(X_resampled, y_resampled)

# Get the best model from grid search
best_xgb = grid_search_xgb.best_estimator_

# Print the best parameters
print("Best Parameters:", grid_search_xgb.best_params_)

# Fit the best model to the data
best_xgb.fit(X_train, y_train)

# Make predictions
y_pred = best_xgb.predict(X_test)

# Accuracy score
xgb_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", xgb_acc)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost Classifier (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="8rzZzbv7tMCH" executionInfo={"status": "ok", "timestamp": 1724779669010, "user_tz": -180, "elapsed": 26276, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="12123d28-9a45-4a4a-e063-d17fcbdb95a4"
from sklearn.ensemble import ExtraTreesClassifier


# Define the Extra Trees Classifier
et_classifier = ExtraTreesClassifier()

# Define hyperparameter grid for grid search
parameters = {
    'n_estimators': [50, 100],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Perform grid search
grid_search_et = GridSearchCV(et_classifier, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search_et.fit(X_resampled, y_resampled)

# Get the best model from grid search
best_et = grid_search_et.best_estimator_

# Print the best parameters
print("Best Parameters:", grid_search_et.best_params_)

# Fit the best model to the data
best_et.fit(X_train, y_train)

# Make predictions
y_pred = best_et.predict(X_test)

# Accuracy score
et_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", et_acc)

# Classification report
class_report = classification_report(y_test, y_pred, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Extra Trees Classifier (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="U2dYRe_zzmAz" executionInfo={"status": "ok", "timestamp": 1724779789820, "user_tz": -180, "elapsed": 120830, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d3f658d9-5e6c-4f54-d8fd-81173705f407"
from sklearn.neural_network import MLPClassifier
# Define the MLP Classifier
mlp_classifier = MLPClassifier()

# Define hyperparameter grid for grid search
mlp_parameters = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['sgd', 'adam'],
    'max_iter': [500, 1000],
}

# Perform grid search
grid_search_mlp = GridSearchCV(mlp_classifier, mlp_parameters, cv=5, n_jobs=-1, verbose=1)
grid_search_mlp.fit(X_train, y_train)

# Get the best model from grid search
best_mlp = grid_search_mlp.best_estimator_

# Print the best parameters
print("Best Parameters:", grid_search_mlp.best_params_)

# Fit the best model to the data
best_mlp.fit(X_train, y_train)

# Make predictions
y_pred = best_mlp.predict(X_test)

# Accuracy score
mlp_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", mlp_acc)

# Classification report
class_report = classification_report(y_test, y_pred, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()

# Calculate ROC curve
y_scores = best_mlp.predict_proba(X_test)[:, 1]  # For binary classification
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MLP Classifier (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %% [markdown] id="n622khM74K1X"
# # camparison

# %% colab={"base_uri": "https://localhost:8080/", "height": 394} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1724779789820, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ox1bM_bt3x4z" outputId="06ac919b-3417-4ed6-aa5e-9bf630a3e10f"
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'SVC', 'SGD Classifier', 'Decision Tree Classifier', 'Random Forest Classifier','Ada Boost Classifier',
             'Gradient Boosting Classifier', 'XgBoost', 'Extra trees', 'MLP'],
    'Score': [log_reg_acc, knn_acc, svc_acc, sgd_acc, dtc_acc, rfc_acc,adaboost_acc, gb_acc, xgb_acc, et_acc,mlp_acc]
})

models.sort_values(by = 'Score', ascending = False)
