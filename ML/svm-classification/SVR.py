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

# %% id="8Ka7eJslaFPu" executionInfo={"status": "ok", "timestamp": 1725610933174, "user_tz": -180, "elapsed": 429, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# Step 1: Generate synthetic data for regression
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# %% colab={"base_uri": "https://localhost:8080/"} id="yMMk3t6taXtu" executionInfo={"status": "ok", "timestamp": 1725610933598, "user_tz": -180, "elapsed": 6, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="5569d715-7780-4180-98db-43585de1b005"
X[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="pwqR1vLzaah0" executionInfo={"status": "ok", "timestamp": 1725610933598, "user_tz": -180, "elapsed": 5, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="86b4f009-43b7-42a6-cd59-8edb28efac1a"
y[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="ighECtlnaWY4" executionInfo={"status": "ok", "timestamp": 1725610936493, "user_tz": -180, "elapsed": 2900, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="272cd1c3-6ada-4ff1-b54d-6e160f85f9b4"
# Step 2: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the SVR model
svr = SVR()

# Step 4: Set up the parameter grid to perform grid search
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]  # Only relevant for 'poly' kernel
}

# Step 5: Use GridSearchCV to search for the best hyperparameters
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

# Step 6: Train the model with the grid search
grid_search.fit(X_train, y_train)

# Step 7: Check the best parameters from the grid search
print("Best parameters from Grid Search:", grid_search.best_params_)

# Step 8: Predict using the best model
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 508} id="9a9y7piGbSph" executionInfo={"status": "ok", "timestamp": 1725610936953, "user_tz": -180, "elapsed": 472, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="17be7d34-bf27-4287-afae-7b73892ed0d0"
# Step 9: Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Step 10: Plot the results
plt.scatter(X_test, y_test, color='blue', label='True values')
plt.scatter(X_test, y_pred, color='red', label='Predicted values')
plt.title('SVR Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
