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

# %% [markdown] id="730cf795"
# <h1> Student Success Prediction using Machine learning</h>

# %% [markdown] id="MDkhIfi8kLj2"
# Feature selection is an important step in building predictive models as it helps to enhance model performance by selecting the most relevant features and eliminating the redundant ones. There are several techniques for feature selection, including Recursive Feature Elimination (RFE), Forward Feature Selection, and Backward Feature Selection. Here's a detailed comparison of these methods:
#
# ### 1. Recursive Feature Elimination (RFE)
# **Definition:**  
# RFE is a recursive process that aims to select features by recursively considering smaller and smaller sets of features. It builds a model on the initial set of features and ranks them by importance. The least important features are removed, and the process is recursively repeated on the pruned set.
#
# **Process:**
# 1. Train a model on the entire dataset.
# 2. Rank the features based on their importance to the model.
# 3. Eliminate the least important feature(s).
# 4. Repeat the process with the remaining features until the desired number of features is reached.
#
# **Pros:**
# - Tends to give better results as it considers all combinations of features.
# - Can handle multicollinearity well.
#
# **Cons:**
# - Computationally expensive, especially with a large number of features.
#
# ### 2. Forward Feature Selection
# **Definition:**  
# Forward Feature Selection is a stepwise selection technique that starts with an empty model and adds features one by one. At each step, the feature that improves the model the most (according to a chosen criterion) is added.
#
# **Process:**
# 1. Start with an empty model.
# 2. Add the feature that improves the model the most.
# 3. Repeat step 2 until adding another feature does not significantly improve the model performance.
#
# **Pros:**
# - Less computationally expensive compared to RFE.
# - Easy to implement and understand.
#
# **Cons:**
# - Can miss interactions between features as it evaluates features individually.
# - May not handle multicollinearity as effectively as RFE.
#
# ### 3. Backward Feature Selection
# **Definition:**  
# Backward Feature Selection is the reverse process of Forward Feature Selection. It starts with all the features and removes them one by one. At each step, the feature that, when removed, improves the model the most (or decreases performance the least) is removed.
#
# **Process:**
# 1. Start with a model that includes all the features.
# 2. Remove the feature that improves the model the most when taken out.
# 3. Repeat step 2 until removing any more features does not significantly improve the model performance.
#
# **Pros:**
# - Can capture interactions between features better than forward selection.
# - Useful when starting with a smaller set of features.
#
# **Cons:**
# - Computationally expensive, especially with a large number of features.
# - Can overfit if not carefully managed.
#
# ### Summary of Differences
# - **RFE vs. Forward/Backward Selection:** RFE is recursive and tends to be more thorough but computationally expensive. Forward and backward selection are stepwise methods and generally faster but might miss feature interactions.
# - **Forward vs. Backward Selection:** Forward selection starts with no features and adds them, while backward selection starts with all features and removes them. Forward selection is faster with fewer initial features, whereas backward selection can better capture feature interactions with a smaller set of final features.
#
# Each method has its advantages and disadvantages, and the choice between them often depends on the specific dataset and problem at hand.

# %% papermill={"duration": 1.53618, "end_time": "2022-01-19T01:39:23.515347", "exception": false, "start_time": "2022-01-19T01:39:21.979167", "status": "completed"} id="72e78d72"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score,learning_curve, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import auc,confusion_matrix, roc_curve, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# %% colab={"base_uri": "https://localhost:8080/"} id="qAfB_SHC4vgb" executionInfo={"status": "ok", "timestamp": 1722668467132, "user_tz": -180, "elapsed": 25924, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="88b127d3-fe78-4fa3-afe9-cf2cdd5144c8"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="5CI5hTLU4wZr" executionInfo={"status": "ok", "timestamp": 1722668467949, "user_tz": -180, "elapsed": 820, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="edfc823b-d628-4810-efc1-d870e5695c05"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/Student Success

# %% [markdown] id="5a1c257f"
# ### Read the dataset

# %% papermill={"duration": 0.070758, "end_time": "2022-01-19T01:39:23.736923", "exception": false, "start_time": "2022-01-19T01:39:23.666165", "status": "completed"} id="b25c241b"
data = pd.read_csv("dataset.csv", sep=";")

# %% papermill={"duration": 0.074315, "end_time": "2022-01-19T01:39:23.861711", "exception": false, "start_time": "2022-01-19T01:39:23.787396", "status": "completed"} id="6df63fbd" executionInfo={"status": "ok", "timestamp": 1722668468395, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/", "height": 307} outputId="2f206e89-bbcc-4e39-d131-971b1f76cb4d"
data.head()

# %% papermill={"duration": 0.059785, "end_time": "2022-01-19T01:39:24.073298", "exception": false, "start_time": "2022-01-19T01:39:24.013513", "status": "completed"} id="83503634" executionInfo={"status": "ok", "timestamp": 1722668468395, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/"} outputId="f62c8d34-3c85-44c6-e035-56ed3615de5a"
data.shape

# %% papermill={"duration": 0.078689, "end_time": "2022-01-19T01:39:24.203703", "exception": false, "start_time": "2022-01-19T01:39:24.125014", "status": "completed"} id="06963df7" executionInfo={"status": "ok", "timestamp": 1722668468395, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/"} outputId="2740fc10-baab-4c24-929b-fcbe1a453531"
data.info()

# %% papermill={"duration": 0.094948, "end_time": "2022-01-19T01:39:24.349908", "exception": false, "start_time": "2022-01-19T01:39:24.254960", "status": "completed"} id="100e5863" executionInfo={"status": "ok", "timestamp": 1722668468395, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/", "height": 401} outputId="ef4d91d2-2457-49ff-9bd9-d14f40b90c72"
data.describe()

# %% [markdown] id="07d6aabf"
# ### check if the dataset has null values

# %% papermill={"duration": 0.06262, "end_time": "2022-01-19T01:39:24.464107", "exception": false, "start_time": "2022-01-19T01:39:24.401487", "status": "completed"} id="619b37db" executionInfo={"status": "ok", "timestamp": 1722668468395, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="ba322e7d-9afb-4e8e-bbb7-c9c553a77210"
data.isnull().sum()

# %% [markdown] id="196311a6"
# ### Count the number of every class in the output

# %% papermill={"duration": 0.227446, "end_time": "2022-01-19T01:39:26.564642", "exception": false, "start_time": "2022-01-19T01:39:26.337196", "status": "completed"} id="4fd223d5" executionInfo={"status": "ok", "timestamp": 1722668468396, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/", "height": 752} outputId="e5601ea1-b4e8-4f09-aa20-1c0ad46258d5"
sns.countplot(data.Target, palette="bwr")
plt.show()
data.Target.value_counts(normalize=True)

# %% [markdown] id="dfd2ccb0"
# ### Draw the correlation between the attributes

# %% [markdown] id="1dec189e"
# ## Convert Dropout =0, Graduate=1, Enrolled=3

# %% id="0bebdbcc"
data = pd.read_csv("dataset.csv", sep=";")
data["Target"]=data.Target.map(dict( Dropout =0, Graduate=1, Enrolled=2))

# %% [markdown] id="683e404c"
# ### select only Dropout and Graduate because the number of Enrolled students is very low

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="GLog3sugkNTU" executionInfo={"status": "ok", "timestamp": 1722668468940, "user_tz": -180, "elapsed": 26, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="62b4e43f-89a3-4fdb-aa4d-c6da5e0f964f"
data['Target'] != 2

# %% id="62cf8a44"
data = data[data['Target'] != 2]

# %% colab={"base_uri": "https://localhost:8080/"} id="iqQJ_PlWlX2y" executionInfo={"status": "ok", "timestamp": 1722668468940, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c6f5c581-58d3-4ec3-81ef-1e13268c4f40"
data.shape

# %% [markdown] id="1d82725c"
# ### Read the input and the output

# %% papermill={"duration": 0.073658, "end_time": "2022-01-19T01:39:34.690351", "exception": false, "start_time": "2022-01-19T01:39:34.616693", "status": "completed"} id="011f71a6"
X = data.drop("Target",axis=1)
y = data["Target"]

# %% papermill={"duration": 0.081611, "end_time": "2022-01-19T01:39:34.840205", "exception": false, "start_time": "2022-01-19T01:39:34.758594", "status": "completed"} id="2640555e" executionInfo={"status": "ok", "timestamp": 1722668468940, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/", "height": 307} outputId="ee4ee3bf-4aea-43ca-9055-66c2808ad645"
X.head()

# %% id="855e83f2" executionInfo={"status": "ok", "timestamp": 1722668468940, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/", "height": 241} outputId="58d12b7c-9d37-46b0-9d46-ec10da9c7b19"
y.head()

# %% id="115b2729"
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X=scaler.fit_transform(X)

# %% id="S50-yfU-hHqM"
#Convert X to dataframe

import pandas as pd
X = pd.DataFrame(X)


# %% colab={"base_uri": "https://localhost:8080/"} id="3aaabc22" executionInfo={"status": "ok", "timestamp": 1722668468940, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="67bded71-7d47-49e6-ee4e-3f234e67a4ee"
from sklearn.feature_selection import SelectPercentile, chi2
selectPercentile= SelectPercentile(chi2, percentile=75)
X_new =selectPercentile.fit_transform(X, y)
print(X.columns[selectPercentile.get_support()].to_list())
X_new.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="H_LzUvDWnFRD" executionInfo={"status": "ok", "timestamp": 1722668468940, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="224c3f9d-8dbd-44e9-f920-a6d94e53e8b7"
# prompt: need to know the importances values for the best features using SelectPercentile(chi2, percentile=80)

# Get the chi2 scores for each feature
chi2_scores = chi2(X, y)[0]

# Get the indices of the selected features
selected_feature_indices = selectPercentile.get_support(indices=True)

# Print the importance values (chi2 scores) for the selected features
for index in selected_feature_indices:
  print(f"Feature {X.columns[index]}: Importance = {chi2_scores[index]}")


# %% [markdown] id="7451e7ec"
# ## Data splitting

# %% id="d2e48059"
X_train, X_test, y_train, y_test = train_test_split( X_new, y, test_size=0.2, random_state=10)

# %% colab={"base_uri": "https://localhost:8080/"} id="V2IXMbexoRMy" executionInfo={"status": "ok", "timestamp": 1722668468940, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ea7f4854-a66f-45b4-cc4d-f7accfb0e9d3"
# prompt: print shapes for every varaible of train and test

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)


# %% [markdown] papermill={"duration": 0.068839, "end_time": "2022-01-19T01:39:35.132728", "exception": false, "start_time": "2022-01-19T01:39:35.063889", "status": "completed"} id="83ff4275"
# ## Build models

# %% [markdown] id="8kQX1-GiAVYu"
# ## Random forest with gridsearch
#

# %% colab={"base_uri": "https://localhost:8080/"} id="hsMcfou7qiT1" executionInfo={"status": "ok", "timestamp": 1722534163807, "user_tz": -180, "elapsed": 202400, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="5a700cdf-9dd6-492c-d17b-61f9fc83839f"
# Random forest with gridsearch

import numpy as np
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=1) # set verbose to a non-zero value
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Calculate the total number of fittings
num_combinations = np.prod([len(v) for v in param_grid.values()])
num_folds = grid_search.cv
total_fits = num_combinations * num_folds
print("Total number of fittings:", total_fits)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)


# %% colab={"base_uri": "https://localhost:8080/", "height": 573} id="f9NHU2fUu20I" executionInfo={"status": "ok", "timestamp": 1722534163808, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="fffd598f-b8e4-4e6d-c218-1ff451cb080f"
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt
print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="LQwTTLYXtdDr" executionInfo={"status": "ok", "timestamp": 1722534164688, "user_tz": -180, "elapsed": 892, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ec65a235-f76a-47cd-fd96-c583608b29ec"
#  Draw the roc curev and auc using the results of random forest

import matplotlib.pyplot as plt
# ROC Curve and AUC
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] papermill={"duration": 0.065624, "end_time": "2022-01-19T01:39:39.162419", "exception": false, "start_time": "2022-01-19T01:39:39.096795", "status": "completed"} id="031687d1"
# ## SVM

# %% papermill={"duration": 1.395977, "end_time": "2022-01-19T01:39:40.624683", "exception": false, "start_time": "2022-01-19T01:39:39.228706", "status": "completed"} id="aabcae4d" outputId="1a6f8103-daef-4b46-e609-3f6cdf0f5e88" colab={"base_uri": "https://localhost:8080/", "height": 508} executionInfo={"status": "ok", "timestamp": 1722535013952, "user_tz": -180, "elapsed": 172570, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
model = SVC(probability=True) # Set probability to True to enable predict_proba
params = {'kernel': ['linear', 'rbf'], 'C': np.arange(0.5,1,0.1), 'gamma': np.arange(0.01,0.05,0.01)}
cv_svm = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1,cv=5, verbose=2)
cv_svm.fit(X_train, y_train)
model= cv_svm.best_estimator_
y_pred=model.predict(X_test)
print(cv_svm.best_params_, cv_svm.best_score_)

#  Draw the roc curev and auc using the results of random forest
# ROC Curve and AUC
y_pred_proba = cv_svm.predict_proba(X_test)[:, 1] # Now predict_proba should work
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# %% id="3ty7E3c8slaH"
# Print classification report for SVC

print(classification_report(y_test, y_pred, digits=4))


# %% [markdown] id="ccb106db"
# ## MLP implementation

# %% id="fd470897"
from sklearn.neural_network import MLPClassifier
mlp_gs = MLPClassifier(max_iter=300)
parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels
y_pred=clf.predict(X_test)

# %% id="06b09c77" outputId="89bf2b79-dc59-4ca6-d098-b21a3b0e995d" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722535817754, "user_tz": -180, "elapsed": 756, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#  Draw the roc curev and auc using the results of random forest
# ROC Curve and AUC
y_pred_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] id="t9wLyvb4AGGp"
# ## GradientBoostingClassifier

# %% id="a0441d03" outputId="2c83f608-9bc7-4948-da30-9bf918f45bdc" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722535827057, "user_tz": -180, "elapsed": 9314, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Updated parameters with valid options
parameters = {
    "loss":["log_loss"],  # Use 'log_loss' or 'exponential'
    "learning_rate": [0.01, 0.075, 0.15],
    "min_samples_split": np.linspace(0.3, 0.5, 2),
    "min_samples_leaf": np.linspace(0.3, 0.5, 2),
    "max_depth":[8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse", "squared_error"],  # Use valid options
    "subsample":[ 0.75, 0.95],
    "n_estimators":[10]
}

#passing the scoring function in the GridSearchCV
clf_GB = GridSearchCV(GradientBoostingClassifier(), parameters,cv=5, n_jobs=-1, verbose=1)

clf_GB.fit(X_train, y_train)
y_pred=clf_GB.predict(X_test)

# %% id="1cbdc1db" outputId="df872828-d1d9-482b-c4d7-41fe5634a680" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722535827888, "user_tz": -180, "elapsed": 846, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#  Draw the roc curev and auc using the results of random forest
# ROC Curve and AUC
y_pred_proba = clf_GB.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] id="VLiC2xrSqiW8"
# ## Voting Classier

# %% id="a60b691c" outputId="0ffa42b3-8987-4417-ba08-9bd7d78d5a4d" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722535855613, "user_tz": -180, "elapsed": 27731, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV


eclf = VotingClassifier(estimators=[
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression()),
    ], voting='soft')

#Use the key for the classifier followed by __ and the attribute
params = {'lr__C': [1.0, 100.0],
      'svm__C': [2,3,4],}

grid_VC = GridSearchCV(estimator=eclf, param_grid=params, cv=5,n_jobs=-1, verbose=3)

grid_VC.fit(X_train,y_train)
print (grid_VC.best_params_)

y_pred=grid_VC.predict(X_test)




# %% id="0331d0db" outputId="50986fc9-714b-4b15-cea2-4acf48a3bbed" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722535856416, "user_tz": -180, "elapsed": 819, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#  Draw the roc curev and auc using the results of random forest
# ROC Curve and AUC
y_pred_proba = grid_VC.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] id="lKWUX8Lt_-dm"
# ## Linear_SGD classifier

# %% id="d22e18c6" outputId="2542631a-c160-4289-e7d3-a99b44999e7d" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722536600604, "user_tz": -180, "elapsed": 455, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Implementing Linear_SGD classifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(max_iter=1000, loss='log_loss') # Change loss to 'log' for probability estimates
Cs = [0.0001,0.001, 0.01, 0.1, 1, 10]
tuned_parameters = [{'alpha': Cs}]
grid_SGD = GridSearchCV(clf, tuned_parameters, scoring = 'accuracy', cv=5,n_jobs=-1, verbose=3)
grid_SGD.fit(X_train, y_train)
y_pred=grid_SGD.predict(X_test)

# %% id="e1943129" outputId="edb2bb0a-23bc-4ca1-85a4-f4f91187c23c" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722536607292, "user_tz": -180, "elapsed": 1562, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#  Draw the roc curev and auc using the results of random forest
# ROC Curve and AUC
y_pred_proba = grid_SGD.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] id="hZrQiAUuq5ns"
# ## Bagging Classifier

# %% id="d1d3c7e3" outputId="a3b8636d-7dc8-4be7-9f9e-505b1b17aa9d" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722536829494, "user_tz": -180, "elapsed": 214471, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.ensemble import BaggingClassifier

bc_params = {"base_estimator__max_depth": [10,20],
          "base_estimator__max_features": ["sqrt"], # Changed 'auto' to 'sqrt'
          "base_estimator__min_samples_leaf": [3, 7, 10],
          "base_estimator__min_samples_split": [5, 7],
          'bootstrap_features': [False, True],
          'max_features': [0.5, 0.7, 1.0],
          'max_samples': [0.5, 0.7, 1.0],
          'n_estimators': [2, 5, 10, 20],
}


bc_gs = GridSearchCV(BaggingClassifier(DecisionTreeClassifier()), bc_params, cv=5, verbose=1)
bc_gs.fit(X_train, y_train)
y_pred=bc_gs.predict(X_test)

# %% id="e70bf6c1" outputId="9c75215d-fd97-4144-f60d-c99601681740" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722536834125, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
y_pred_proba = bc_gs.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] id="R6bLj3zwrCnj"
# ## Stacking Classifier
#
# **Base Models (Level-0 Models):**
#
# Train multiple base models on the training data.
# Each base model makes predictions on the training data and often on a hold-out validation set to avoid overfitting.
# The predictions from these base models are collected and used as new features for the next level.
#
# **Meta-Model (Level-1 Model):**
#
# Train a meta-model on the predictions made by the base models.
# The meta-model learns how to combine the base model predictions to make the final prediction.
#
# **Making Predictions:**
#
# To make a prediction on new data, first, the base models make their predictions.
# These predictions are then fed into the meta-model, which outputs the final prediction.

# %% colab={"base_uri": "https://localhost:8080/"} id="XzOKNhAb0AXu" executionInfo={"status": "ok", "timestamp": 1722536855349, "user_tz": -180, "elapsed": 21234, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="4558d6b5-b853-4373-abf9-335c2eb4fac0"
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

# define the base models
level0 = list()
level0.append(('LR', LogisticRegression(C= 0.1, penalty='l2')))
level0.append(('RF',RandomForestClassifier(n_estimators= 300)))
level0.append(('DT', DecisionTreeClassifier(max_depth=6, min_samples_leaf=1, min_samples_split=2, splitter= 'random')))
level0.append(('SGD',SGDClassifier(alpha= 0.01)))
level0.append(('XGB',XGBClassifier(subsample= 0.8, min_child_weight= 5, max_depth= 5, gamma= 1, colsample_bytree= 0.8)))
level0.append(('MLP',MLPClassifier(activation= 'relu', alpha= 0.0001, hidden_layer_sizes= (20,), learning_rate= 'adaptive', solver= 'adam')))
level0.append(('Adaboost',AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6, min_samples_leaf=1, min_samples_split=2, splitter= 'random'))))
level0.append(('bagging',BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=6, min_samples_leaf=1, min_samples_split=2, splitter= 'random'))))

# define meta learner model
level1= SVC(C= 0.7999999999999999, gamma= 0.04, kernel= 'rbf')
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 573} id="6i98WSHnwd7N" executionInfo={"status": "ok", "timestamp": 1722536957969, "user_tz": -180, "elapsed": 1200, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0d888ede-bb09-4237-a3b6-146918b55572"
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


