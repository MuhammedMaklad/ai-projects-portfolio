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

# %% id="KjOO7iK1VLXG" executionInfo={"status": "ok", "timestamp": 1723966998382, "user_tz": -180, "elapsed": 300, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn. import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data

X = 2 * np.random.rand(10000, 1)    #  100 value , 0 -1
y = 1 +  X + np.random.randn(10000, 1)   #  y = a+bX

# %% colab={"base_uri": "https://localhost:8080/"} id="qNVZptBvZ_Qm" executionInfo={"status": "ok", "timestamp": 1723966998682, "user_tz": -180, "elapsed": 10, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0f2c4d64-a564-4e5a-fa56-04f8c2a47931"
X[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="Mlzl4riMavdj" executionInfo={"status": "ok", "timestamp": 1723966998682, "user_tz": -180, "elapsed": 9, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="77ee02a7-93a3-4a0d-f8d6-c6a36a1d0d20"
y[:10]

# %% id="MrqefKvFX_YU" executionInfo={"status": "ok", "timestamp": 1723966998683, "user_tz": -180, "elapsed": 7, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Scale the values of X using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# %% colab={"base_uri": "https://localhost:8080/"} id="AcnOWOSkYS2W" executionInfo={"status": "ok", "timestamp": 1723966998683, "user_tz": -180, "elapsed": 7, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="13d540bc-df09-4a51-a1d4-0f593a961604"
X_scaled[:5]

# %% id="Yx8UZJ1DZ3uM" executionInfo={"status": "ok", "timestamp": 1723966998683, "user_tz": -180, "elapsed": 6, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=44)

# %% colab={"base_uri": "https://localhost:8080/"} id="SybcxvWNLGVO" executionInfo={"status": "ok", "timestamp": 1723966999009, "user_tz": -180, "elapsed": 332, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e3000b5f-c844-4ef6-d14e-d71a3e9f38c0"
X_train.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="cgx-d2QGLLbJ" executionInfo={"status": "ok", "timestamp": 1723966999009, "user_tz": -180, "elapsed": 9, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f83b306b-b151-4e22-ba4b-5bc924979338"
y_train.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="b9GurI27r6Ed" executionInfo={"status": "ok", "timestamp": 1723966999009, "user_tz": -180, "elapsed": 7, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="917e7ae7-6e95-4694-f595-f0d096103cf3"
X_test.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="8VFwj4DcY8aw" executionInfo={"status": "ok", "timestamp": 1723966999009, "user_tz": -180, "elapsed": 6, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ad76dc1e-ad8a-4ca8-9322-cbba68fc7396"
y_test.shape

# %% id="kH1Wi9Eca0sG" executionInfo={"status": "ok", "timestamp": 1723966999009, "user_tz": -180, "elapsed": 5, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="RTEb_JowLnsm" executionInfo={"status": "ok", "timestamp": 1723966999009, "user_tz": -180, "elapsed": 5, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="199a31dd-7832-4f28-c112-dd3f0bf12af3"
y_pred[:10]

# %% id="J2CFtTIVLrPh" executionInfo={"status": "ok", "timestamp": 1723966999009, "user_tz": -180, "elapsed": 5, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ffe4ee7f-5c9d-4569-c002-d4e7a7fe75d4" colab={"base_uri": "https://localhost:8080/"}
y_test[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="otUw_-V_aRqd" executionInfo={"status": "ok", "timestamp": 1723966999009, "user_tz": -180, "elapsed": 4, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="7bebffc8-6108-4316-ef2c-8bb55f15fb63"
# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="yutsdaOZa5Bx" executionInfo={"status": "ok", "timestamp": 1723967001399, "user_tz": -180, "elapsed": 1287, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="4303bc4a-5928-4282-f3ee-1730b074c985"
# Plot the training data, testing data, and the regression line
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='red', label='Testing data')
plt.plot(X_test, y_pred, color='yellow', linewidth=2, label='Regression line (predicted)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Example')
plt.show()

