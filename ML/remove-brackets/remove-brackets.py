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

# %% colab={"base_uri": "https://localhost:8080/"} id="rhzhTb5AZErA" executionInfo={"status": "ok", "timestamp": 1724771750979, "user_tz": -180, "elapsed": 2894, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="cecf4255-80e7-4dcc-9204-d2c5b424cd71"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="qn-S_l3eZJPM" executionInfo={"status": "ok", "timestamp": 1724771750980, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="1676f883-b893-4a1b-f18e-d09ff9a09bdb"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/remove brackets

# %% id="a5a422d0" outputId="b150bbb9-5af1-4e5a-a10c-310328c9d929" colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"status": "ok", "timestamp": 1724771878977, "user_tz": -180, "elapsed": 2052, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
import pandas as pd

data = pd.read_csv('dataset.csv', names=['x1', 'x2', 'x3', 'y'])
data.head()

# %% id="680a26a7" outputId="7ab7ae8e-047f-4793-8a5e-bbfb74845e6c" colab={"base_uri": "https://localhost:8080/", "height": 209} executionInfo={"status": "ok", "timestamp": 1724771886911, "user_tz": -180, "elapsed": 373, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.dtypes

# %% id="8aea0bf3" outputId="84fa66c4-b48b-4602-84e2-5cbc9fe47060" colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"status": "ok", "timestamp": 1724771928615, "user_tz": -180, "elapsed": 346, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data_numeric = data.copy()
data_numeric['x1'] = data['x1'].str.replace('[', '', regex=False).astype('int64')
data_numeric['x3'] = data['x3'].str.replace(']', '', regex=False).astype('int64')
data_numeric.head()

# %% id="6a47a06c" outputId="77e9e9f5-24fd-4aff-921b-6fd2774bd2a1" colab={"base_uri": "https://localhost:8080/", "height": 209} executionInfo={"status": "ok", "timestamp": 1724771941716, "user_tz": -180, "elapsed": 320, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data_numeric.dtypes

# %% id="3f6da764"
X=data_numeric.iloc[:,:-1]
y=data_numeric.iloc[:,-1]

# %% colab={"base_uri": "https://localhost:8080/"} id="XgzLI1aQa-Oe" executionInfo={"status": "ok", "timestamp": 1724772318494, "user_tz": -180, "elapsed": 284999, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="412b6897-644f-46d7-bf07-c1c36e623d3a"
# Scaling , splitting modeling for all ML algorithms

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modeling
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {accuracy:.2f}")

