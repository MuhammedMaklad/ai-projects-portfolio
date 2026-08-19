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

# %%
import pandas as pd

# %%
data = pd.read_csv("tvmarketing.csv")

# %%
data.head()

# %%
data.tail()

# %%
data.info()

# %%
data.describe()

# %%
for col in data.columns:
    print(f"{col} contains na values " + f"{data[col].isna().sum()}")
    print(f"{col} contains null values " + f"{data[col].isnull().sum()}")
    print(f"{col} contains unique values " + f"{data[col].unique().sum()}")
    print(f"{col} contains duplicates values " + f"{data[col].duplicated().sum()}")
    print("-" * 40)

# %%
for col in data.columns:
    print(data[col].value_counts())

# %%
data.plot(
    kind="scatter",
    x="TV",
    y="Sales",
    title="TV vs Sales",
    xlabel="TV advertising budget",
    ylabel="Sales" 
)

# %%
# Splitting data
X = data['TV']
X.head()

# %%
y = data['Sales']
y.head()

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=42)

# %%
import numpy as np

# %%
X_train.shape

# %%
X_train = X_train.values.reshape(-1,1)

# %%
X_test = X_test.values.reshape(-1, 1)

# %%
# Scaling data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# build Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# %%
# Print the intercept and coefficients
print(model.intercept_)
print(model.coef_)

# %%
# predict phase
y_pred = model.predict(X_test)

# %%
y_pred[:5]

# %%
y_test[:5]

# %%
# calculate R^2 and mean squared error
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2, mse

# %%
y_test.shape

# %%
# Actual vs Predicted
import matplotlib.pyplot as plt
c = [i for i in range(1,61,1)]         # generating index
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                       # Y-label

# %%
# Error terms
c = [i for i in range(1,61,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
