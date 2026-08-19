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

# %% colab={"base_uri": "https://localhost:8080/", "height": 90} id="A0N3vo0BbZM_" outputId="3192296a-ead7-4256-f64b-8655068f47f2"
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))


# %% colab={"base_uri": "https://localhost:8080/"} id="g083NfvcmpCc" outputId="c44e8be3-8230-4b63-a47f-4510df6fd893"
# !ls /content

# %% id="mxU8DjOOmuNC"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# %% colab={"base_uri": "https://localhost:8080/", "height": 273} id="2XNBO23Wm4Nj" outputId="1799cfd4-1062-40b0-eef3-27a1eafee4ef"
df = pd.read_csv('breast-cancer.csv')
df.head()

# %% [markdown] id="dgQ5CfrenEPT"
# - Will Drop id Column no need it

# %% id="A4GMUyn4nC_q"
df.drop('id', axis=1, inplace=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="YI549yA9nMoW" outputId="a06dc0ef-75f1-45d7-ecf9-3b204a5d394e"
df.info()

# %% [markdown] id="KMD2eK08nSiz"
# - all features are numerical no missing values

# %% id="zbYyrbmDngzL"
# encode target to numerical feature
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="IKVCrODOnXeT" outputId="6b05cb03-a9b9-4c36-8ac9-10f98f420f41"
plt.figure(figsize=(20,18))
sns.heatmap(df.corr(), annot=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="b1wVFz7Tn6Ce" outputId="63218732-5d2a-4c60-afa8-e44e7af9d9bc"
# get all features the corr between it and target > .2
df_corrr = df.corr()['diagnosis']
df_corrr = df_corrr[df_corrr > .2]
df_corrr

# %% colab={"base_uri": "https://localhost:8080/"} id="jwr9OH1ZodRr" outputId="07a91d36-2ffe-49cd-abd3-aa092d4c2a3b"
# get target feature that have strong influence in our target
target_feature = [feature for feature, value in df_corrr.items()]
target_feature

# %% colab={"base_uri": "https://localhost:8080/"} id="cuSSDVQEo6P7" outputId="6e09bb57-99f6-4537-bc93-d9d50774ff23"
target_feature.remove('diagnosis')
target_feature

# %% id="2Y8_b3e_pKjI"
X = df[target_feature]
y = df['diagnosis']

# %% [markdown] id="p7JpcqSXpjEL"
# - X the features that will work with it
# - y the our target

# %% colab={"base_uri": "https://localhost:8080/", "height": 273} id="926w2D9DpRmG" outputId="5fe93026-a271-41ea-9f91-d29882a3c527"
X.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} id="_wttD753pcF_" outputId="44b897b7-b46b-4d6f-9f10-939cb64a338f"
y.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 833} id="6sP94irppfEs" outputId="443707e2-5c93-4a02-fff9-f2f6c070b49a"
X.describe().T

# %% [markdown] id="Pgza2Cdgp0bT"
# - Will check for outliers

# %% id="2nP-CZ3zqaCe"
import math

# %% colab={"base_uri": "https://localhost:8080/", "height": 993} id="R9va4_-Kp3oW" outputId="f5722e6e-875e-4d27-c503-c8ef707c19a8"
plt.figure(figsize=(20,18))
n_features = len(X.columns)
n_cols = math.ceil(math.sqrt(n_features))  # Columns ≈ sqrt(n_features)
n_rows = math.ceil(n_features / n_cols)    # Rows adjust to fit
for idx, feature in enumerate(X.columns, 1):
  plt.subplot(n_rows, n_cols, idx)
  sns.boxplot(df[feature])
  plt.title(feature)
plt.tight_layout()
plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} id="LGg01RthsHOJ" outputId="13b68b68-838f-4f01-bcb3-d3ac1285d126"
def get_outlier_features(df, features):
    outlier_features = []
    for feature in features:
        # Calculate Q1, Q3, and IQR
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Check for outliers
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        if not outliers.empty:
            outlier_features.append(feature)
    return outlier_features

outlier_cols = get_outlier_features(df, target_feature)
print("Features with outliers:", outlier_cols)


# %% id="zOooKaDlsdvd"
def impute_outliers_iqr(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
    df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])
    return df


# %% id="xoiMkZHispum"
clean_X = impute_outliers_iqr(X, target_feature)

# %% id="SJFQNp8Ns-Tl"
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
clean_X = scaler.fit_transform(clean_X)
X = scaler.transform(X)

# %% id="wKSA6bvYtxkW"
clean_X = pd.DataFrame(clean_X, columns=target_feature)

# %% colab={"base_uri": "https://localhost:8080/", "height": 273} id="E9UkXOLlu6Re" outputId="16385efa-8c13-4de5-afc8-9639dc0c2908"
clean_X.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 273} id="2KC5C8LSvZDf" outputId="85a9fbdc-50d4-44f9-856f-b7603df3760a"
X = pd.DataFrame(X, columns=target_feature)
X.head()

# %% [markdown] id="0cROd9RXtLJO"
# ### Will process data using clean data and data that contain outliers

# %% id="D6j-T5K3tKMz"
from sklearn.model_selection import train_test_split
def split_data(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test


# %% [markdown] id="BV2PHXZ0tinW"
# ### Run SVM with feature that contain outliers

# %% id="27MAaA21tdGk"
X_train, X_test, y_train, y_test = split_data(X, y)

# %% colab={"base_uri": "https://localhost:8080/"} id="ibgDof5UwFNM" outputId="e96fcba3-2413-4d12-dfba-1b559d23d1f0"
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
    'kernel': ['rbf', 'linear', 'poly']  # Kernel type
}

# Create an SVC classifier
svc = SVC()

# Create GridSearchCV object
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate the best model on the test data
best_svc = grid_search.best_estimator_
test_accuracy = best_svc.score(X_test, y_test)
print("Test accuracy:", test_accuracy)


# %% colab={"base_uri": "https://localhost:8080/"} id="9tSj6NLOwW1w" outputId="5a345dd6-3a33-4d21-c182-74acafaa5731"
from sklearn.metrics import classification_report, confusion_matrix
y_pred = best_svc.predict(X_test)
print(classification_report(y_test, y_pred))

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="jGwki5AjwjRV" outputId="b8527f5c-d686-4c41-ef14-9025b9df0d67"
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()

# %% [markdown] id="coOXj-LKwoF8"
# ### work on data that not contain outliers

# %% id="nb75kGuPwnzC"
X_train, X_test, y_train, y_test = split_data(clean_X, y)

# %% colab={"base_uri": "https://localhost:8080/"} id="2pBEKL01wyvN" outputId="78417b53-305b-4a79-d075-ce665966d536"
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
    'kernel': ['rbf', 'linear', 'poly']  # Kernel type
}

# Create an SVC classifier
svc = SVC()

# Create GridSearchCV object
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate the best model on the test data
best_svc = grid_search.best_estimator_
test_accuracy = best_svc.score(X_test, y_test)
print("Test accuracy:", test_accuracy)


# %% colab={"base_uri": "https://localhost:8080/"} id="PBcGtobLw1G8" outputId="70ca8f8f-bbe8-4207-a3c3-ae1e80c9ce1f"
y_pred = best_svc.predict(X_test)
print(classification_report(y_test, y_pred))

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="gNVx5_csw4jk" outputId="79ab745a-91c4-4108-a83d-6514cacc393e"
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()
