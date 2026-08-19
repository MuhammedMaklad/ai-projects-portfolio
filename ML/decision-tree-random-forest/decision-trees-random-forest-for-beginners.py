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

# %% [markdown] id="150ff3dc"
#
# # 🌳 Decision Tree & Random Forest
#
# # 1. Decision Tree
#
# ![Capture.PNG](attachment:Capture.PNG)
#
# Decision Trees are an important type of algorithm for predictive modeling machine learning.
#
# The classical decision tree algorithms have been around for decades and modern variations like random forest are among the most powerful techniques available.
#
# Classification and Regression Trees or `CART` for short is a term introduced by `Leo Breiman` to refer to Decision Tree algorithms that can be used for classification or regression predictive modeling problems.
#
# Classically, this algorithm is referred to as “`decision trees`”, but on some platforms like R they are referred to by the more modern term CART.
#
# The `CART` algorithm provides a foundation for important algorithms like `bagged decision trees`, `random forest` and `boosted decision trees`.
#
# ### CART Model Representation
# The representation for the CART model is a binary tree.
#
# This is your binary tree from algorithms and data structures, nothing too fancy. Each root node represents a single input variable (x) and a split point on that variable (assuming the variable is numeric).
#
# The leaf nodes of the tree contain an output variable (y) which is used to make a prediction.
#
# Given a new input, the tree is traversed by evaluating the specific input started at the root node of the tree.
#
# #### Some **advantages** of decision trees are:
# * Simple to understand and to interpret. Trees can be visualised.
# * Requires little data preparation.
# * Able to handle both numerical and categorical data.
# * Possible to validate a model using statistical tests.
# * Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.
#
# #### The **disadvantages** of decision trees include:
# * Overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
# * Decision trees can be unstable. Mitigant: Use decision trees within an ensemble.
# * Cannot guarantee to return the globally optimal decision tree. Mitigant: Training multiple trees in an ensemble learner
# * Decision tree learners create biased trees if some classes dominate. Recommendation: Balance the dataset prior to fitting
#
# # 2. Random Forest
# Random Forest is one of the most popular and most powerful machine learning algorithms. It is a type of ensemble machine learning algorithm called Bootstrap Aggregation or bagging.
# ![inbox_3363440_e322b7c76f2ca838ba3753e3c76c5efc_inbox_2301650_875af39bcc296f0a783519a400412dee_RF.jpg](attachment:inbox_3363440_e322b7c76f2ca838ba3753e3c76c5efc_inbox_2301650_875af39bcc296f0a783519a400412dee_RF.jpg)
# To improve performance of Decision trees, we can use many trees with a random sample of features chosen as the split.

# %% [markdown] id="AS78rLbY34zM"
# https://www.kaggle.com/code/faressayah/decision-trees-random-forest-for-beginners

# %% [markdown] id="e2875c48"
# # 3. Decision Tree & Random Forest Implementation in python
#
# We will use Decision Tree & Random Forest in Predicting the attrition of your valuable employees.

# %% executionInfo={"elapsed": 2030, "status": "ok", "timestamp": 1725174638938, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="882f9e29"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1725174638939, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="VxaA18cYxLsX"

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23675, "status": "ok", "timestamp": 1725174662611, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="lxZD_hmDxL9s" outputId="7e070f5f-768f-454a-a5fd-5d7f035dea2f"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1847, "status": "ok", "timestamp": 1725174664453, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="wnln3efNxQGh" outputId="ffaf56bc-fa24-4c73-e3d5-d241dcd5e7ff"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Decision Tree and Random forest

# %% colab={"base_uri": "https://localhost:8080/", "height": 325} executionInfo={"elapsed": 1516, "status": "ok", "timestamp": 1725174665965, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1cd477c7" outputId="a42cd433-9e77-4bd6-ebbc-0c01e6eaf32d"
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()

# %% [markdown] id="b5489440"
# # Exploratory Data Analysis

# %% colab={"base_uri": "https://localhost:8080/", "height": 496} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1725174665965, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="3298555e" outputId="87ede46a-03c6-4623-c323-552205a2cbe5"
sns.countplot(x='Attrition', data=df)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="FqAq8tsjg9rz" executionInfo={"status": "ok", "timestamp": 1725174665965, "user_tz": -180, "elapsed": 12, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="58f6cd9c-9397-460d-ac62-b04d9e5006a3"
df.isna().sum()

# %% colab={"base_uri": "https://localhost:8080/"} id="vkao11mrhBvh" executionInfo={"status": "ok", "timestamp": 1725174665965, "user_tz": -180, "elapsed": 12, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c3f85b25-8d0b-4d3b-b0ec-ada605f78c39"
df.info()

# %% executionInfo={"elapsed": 475, "status": "ok", "timestamp": 1725174666430, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="2d9f91ea"
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)

categorical_col = []
for column in df.columns:
    if df[column].dtype == object and len(df[column].unique()) <= 50:
        categorical_col.append(column)

# %% colab={"base_uri": "https://localhost:8080/", "height": 458} executionInfo={"elapsed": 28, "status": "ok", "timestamp": 1725174666431, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="eXWLnzO1xurl" outputId="2beeebc8-2891-4f96-da6d-21da1219f572"
df['Attrition']

# %% executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1725174666431, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="LUnBXHJCwj5C"
df['Attrition'] = df.Attrition.astype("category").cat.codes

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1725174666431, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="7mSWmjUYxy6Q" outputId="6a501519-800c-4e5b-cabe-7c5709951349"
print(df['Attrition'])

# %% [markdown] id="d407259a"
# # Data Processing

# %% executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1725174666431, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="9bc951c2"
categorical_col.remove('Attrition')

# %% executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1725174666431, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="f35f86b2"
# Transform categorical data into dummies
# categorical_col.remove("Attrition")
# data = pd.get_dummies(df, columns=categorical_col)
# data.info()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
for column in categorical_col:
    df[column] = label.fit_transform(df[column])

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} id="PAGkk_WaiAnK" executionInfo={"status": "ok", "timestamp": 1725174666431, "user_tz": -180, "elapsed": 25, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9443a49c-eee5-4900-ad97-058d52b1179a"
df.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 349} id="yvuUgVVuiLih" executionInfo={"status": "ok", "timestamp": 1725174666431, "user_tz": -180, "elapsed": 24, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c87a064c-3ce4-4ff4-e215-f023f217a28f"
df.describe()

# %% executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1725174666431, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="e0bbc2c6"
from sklearn.model_selection import train_test_split

X = df.drop('Attrition', axis=1)
y = df.Attrition

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% id="p8HhKxCviJvd" executionInfo={"status": "ok", "timestamp": 1725174666431, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# prompt: need to balance the X_train and y_train using oversampling

#from imblearn.over_sampling import RandomOverSampler

#ros = RandomOverSampler(random_state=42)
#X_train, y_train = ros.fit_resample(X_train, y_train)


# %% [markdown] id="d9f7a8d8"
# # Applying Tree & Random Forest algorithms

# %% executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1725174666431, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="5ae14673"
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# %% [markdown] id="d67152db"
# ## 1. Decision Tree Classifier
#
# **Decision Tree parameters:**
# - `criterion`: The function to measure the quality of a split. Supported criteria are "`gini`" for the Gini impurity and "`entropy`" for the information gain.
# ***
# - `splitter`: The strategy used to choose the split at each node. Supported strategies are "`best`" to choose the best split and "`random`" to choose the best random split.
# ***
# - `max_depth`: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than `min_samples_split` samples.
# ***
# - `min_samples_split`: The minimum number of samples required to split an internal node.
# ***
# - `min_samples_leaf`: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
# ***
# - `min_weight_fraction_leaf`: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
# ***
# - `max_features`: The number of features to consider when looking for the best split.
# ***
# - `max_leaf_nodes`: Grow a tree with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
# ***
# - `min_impurity_decrease`: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
# ***
# - `min_impurity_split`: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 450, "status": "ok", "timestamp": 1725174666858, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1791bad6" outputId="1b2f0458-3724-401c-86b5-9465e59c38fa"
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

# %% [markdown] id="f9d1a96a"
# ## 2. Decision Tree Classifier Hyperparameter tuning

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11170, "status": "ok", "timestamp": 1725174678022, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="f3c8b525" outputId="c11e8f2d-7f83-4b53-8f62-0007c08c9b23"
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tree_clf = DecisionTreeClassifier(random_state=42)
tree_cv = GridSearchCV(
    tree_clf,
    param_grid,
    scoring="accuracy",
    n_jobs=-1, # use all cores of the CPU
    verbose=1,  # print progress
    cv=5 #kfold count
)

tree_cv.fit(X_train, y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")

best_DT=tree_cv.best_estimator_

print_score(best_DT, X_train, y_train, X_test, y_test, train=True)
print_score(best_DT, X_train, y_train, X_test, y_test, train=False)

# %% [markdown] id="703f4aaa"
# ### Visualization of a tree

# %% executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1725174678022, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="01d0dd60"
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pydot

features = list(df.columns)
features.remove("Attrition")

# %% colab={"base_uri": "https://localhost:8080/", "height": 351} executionInfo={"elapsed": 15, "status": "error", "timestamp": 1725174678023, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="cbd9103a" outputId="0c917188-6654-4399-fcf2-796fc77a0d86"
dot_data = StringIO()
export_graphviz(tree_cv, out_file=dot_data, feature_names=features, filled=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())

# %% [markdown] id="3be807b2"
# # 3. Random Forest
#
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
#
# - **Random forest algorithm parameters:**
# - `n_estimators`: The number of trees in the forest.
# ***
# - `criterion`: The function to measure the quality of a split. Supported criteria are "`gini`" for the Gini impurity and "`entropy`" for the information gain.
# ***
# - `max_depth`: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than `min_samples_split` samples.
# ***
# - `min_samples_split`: The minimum number of samples required to split an internal node.
# ***
# - `min_samples_leaf`: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
# ***
# - `min_weight_fraction_leaf`: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
# ***
# - `max_features`: The number of features to consider when looking for the best split.
# ***
# - `max_leaf_nodes`: Grow a tree with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
# ***
# - `min_impurity_decrease`: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
# ***
# - `min_impurity_split`: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
# ***
# - `bootstrap`: Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
# ***
# - `oob_score`: Whether to use out-of-bag samples to estimate the generalization accuracy.

# %% executionInfo={"elapsed": 870, "status": "ok", "timestamp": 1725174683678, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="be8c7dd6" colab={"base_uri": "https://localhost:8080/"} outputId="d6277792-5a90-416c-8774-88faf017f7e1"
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

# %% [markdown] id="a63bb0e1"
# ## 4. Random Forest hyperparameter tuning

# %% [markdown] id="0c74ab64"
# ### a) Randomized Search Cross Validation

# %% id="a3473343" executionInfo={"status": "ok", "timestamp": 1725176042222, "user_tz": -180, "elapsed": 918695, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/"} outputId="677be2ef-456c-47ae-e80f-c8482cb3720d"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [100, 500, 1000, 1500]
max_features = ['auto', 'sqrt']
max_depth = [3,5,8,15,20]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False] # 1000 record  = > 5 parts , every part 200 with replacement

random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

rf_clf = RandomForestClassifier(random_state=42)

rf_cv = RandomizedSearchCV(
    estimator=rf_clf,
    scoring='accuracy',
    param_distributions=random_grid,
    n_iter=200,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_cv.fit(X_train, y_train)
rf_best_params = rf_cv.best_params_
print(f"Best paramters: {rf_best_params})")

rf_clf = RandomForestClassifier(**rf_best_params)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

# %% [markdown] id="31ecf3cd"
# `Random search` allowed us to narrow down the range for each hyperparameter. Now that we know where to concentrate our search, we can explicitly specify every combination of settings to try. We do this with `GridSearchCV`, a method that, instead of sampling randomly from a distribution, evaluates all combinations we define.

# %% [markdown] id="9adb7e01"
# ### b) Grid Search Cross Validation

# %% id="7f77bd10" executionInfo={"status": "ok", "timestamp": 1725178999184, "user_tz": -180, "elapsed": 2956970, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} colab={"base_uri": "https://localhost:8080/"} outputId="57877fb4-ad34-48f0-ef47-5b9e6fff8e53"
n_estimators = [100, 500, 1000, 1500]
max_features = ['auto', 'sqrt']
max_depth = [2, 3, 5]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4, 10]
bootstrap = [True, False]

params_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

rf_clf = RandomForestClassifier(random_state=42)

rf_cv = GridSearchCV(
    rf_clf,
    params_grid,
    scoring="accuracy",
    cv=5,
    verbose=1,
    n_jobs=-1
)


rf_cv.fit(X_train, y_train)
best_params = rf_cv.best_params_
print(f"Best parameters: {best_params}")

rf_clf = RandomForestClassifier(**best_params)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

# %% [markdown] id="8bb36713"
# # Summary
# In this notebook we learned the following lessons:
# - Decsion tree and random forest algorithms and the parameters of each algorithm.
# - How to tune hyperparameters for both Decision tree and Random Forest.
# - Balance your dataset before training to prevent the tree from being biased toward the classes that are dominant.
#   - By sampling an equal number of samples from each class  
#   - By normalizing the sum of the sample weights (sample_weight) for each class to the same value.
#
#   
# ## References:
# - [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
# - [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
# - [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
# - [Bagging and Random Forest Ensemble Algorithms for Machine Learning](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)
