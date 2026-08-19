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

# %% [markdown]
# # Mushroom Classification
#
# Attribute Information: (classes: edible=e, poisonous=p)
#
# cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
#
# cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
#
# cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
#
# bruises: bruises=t,no=f
#
# odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
#
# gill-attachment: attached=a,descending=d,free=f,notched=n
#
# gill-spacing: close=c,crowded=w,distant=d

# %%
import pandas as pd

# %%
# read dataset
df = pd.read_csv("mushrooms.csv")

# %% [markdown]
# # Explore Dataset

# %%
# show first 5 rows
df.head()

# %%
# show last 5 rows
df.tail()

# %%
df.info()

# %% [markdown]
# - all features are categorical

# %% [markdown]
# # Pre Processing for datasets

# %%
# check if dataset contains missing values
for col in df.columns:
    print(f"feature {col} contains {df[col].isnull().sum()} null values")
    print("-" * 20)

# %% [markdown]
# - Features not contain any missing values

# %%
# Some information about dataset
df.describe()

# %%
# Values count for each variable in each feature
for col in df.columns:
    print(f"feature {col}")
    print(df[col].value_counts())
    print("--" * 15)

# %%
# Split data into features and target
X = df.drop("class", axis=1)
y = df["class"]

# %%
# Get Feature Column names
feature_names = X.columns
print(feature_names)

# %%
X.shape

# %%
# Encode Categorical features using Hot One Encoding
X = pd.get_dummies(X, columns=feature_names)

# %%
X.shape

# %%
X.head(10)

# %%
X.info()

# %%
y.unique()

# %%
# ecode our target
y = y.map({
    'e':1,
    'p':0
})

# %%
y.head()

# %%
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train.shape

# %%
X_test.shape

# %%
y_train.shape

# %%
y_test.shape

# %%
X_train.head()

# %% [markdown]
# # Building Model

# %%
# # build our model using discussion tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# %%
# Initialize model
decision_tree_model = DecisionTreeClassifier(random_state=42)

# %%
# Define Hyperparameters for Grid Search
param_grid = {
    'criterion': ['gini', 'entropy'],  # Splitting criteria
    'max_depth': [3, 5, 10, None],     # Depth of the tree
    'min_samples_split': [2, 5, 10],   # Minimum samples required to split
    'min_samples_leaf': [1, 2, 4],     # Minimum samples required at a leaf node
    'max_features': [None, 'sqrt', 'log2']  # Number of features to consider for best split
}

# %%
# perform grid search
# Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=decision_tree_model, param_grid=param_grid, 
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters
print("Best Parameters:", grid_search.best_params_)


# %%
# train model in best params
best_dt = grid_search.best_estimator_
best_dt.fit(X_train, y_train)

# %%
# Evaluate model
# Predictions
y_pred = best_dt.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# %%
y_train_pred = best_dt.predict(X_train)
train_accuracy =  accuracy_score(y_train,y_train_pred)
print("Training accuracy: ", train_accuracy)

# %%
# Visualize the Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(best_dt, feature_names=X.columns, class_names=['negative', 'positive'], filled=True)
plt.show()

# %%
# Building model using KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# %%
# Initialize the KNN Classifier
knn = KNeighborsClassifier()

# Define the hyperparameter grid
param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}


# %%
# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)


# %%
# Retrieve the best model
best_knn = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_knn.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Test Set Accuracy: {accuracy:.4f}')


# %%
# Build Model using NB
from sklearn.naive_bayes import GaussianNB
import numpy as np

# %%
# Initialize the Gaussian Naive Bayes Classifier
gnb = GaussianNB()

# Define the hyperparameter grid
param_grid = {
    'var_smoothing': np.logspace(-9, 0, 10)
}

# %%
# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# %%
# Retrieve the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Test Set Accuracy: {accuracy:.4f}')


# %%
# Build Model using Logistic Regression
from sklearn.linear_model import LogisticRegression


# %%
# Initialize Logistic Regression model
logreg = LogisticRegression(max_iter=1000)

# Define parameter grid for GridSearchCV
param_grid = {
    'C': np.logspace(-4, 4, 20),   # Regularization strength
    'penalty': ['l1', 'l2'],       # Regularization type
    'solver': ['liblinear']        # Solver for L1/L2 penalties
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Fit GridSearch to training data
grid_search.fit(X_train, y_train)

# %%
# Retrieve the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Test Set Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
