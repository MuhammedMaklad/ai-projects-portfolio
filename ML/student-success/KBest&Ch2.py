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

# %% id="xbkN1d7kXo-h"
# prompt: code to upload file from local mahein

from google.colab import files
uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

# %% id="hGhRTImhlbra"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
import math

# %% id="Y7r2lHglvOuu"
df = pd.read_csv('dataset.csv',sep=';')

# %% colab={"base_uri": "https://localhost:8080/", "height": 464} id="M2MceUSyvh-J" outputId="67aa16e4-1654-4810-873c-21a8568c6f5b"
df.head(10)

# %% colab={"base_uri": "https://localhost:8080/"} id="JRwoYuZuvng9" outputId="9a72f2d2-9cf5-49da-96fd-3ef794af4e3f"
df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="RlsxTgDOvtqq" outputId="55fd720f-db07-4f89-a77b-4ba1fa9ab857"
df.describe(include='all').T

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="pyorDh3zvz6J" outputId="148b003d-26fa-4aa0-f854-2a1058965c4a"
df['Target'].value_counts().plot(kind='barh')
plt.show()

# %% id="PGXVkUj1wdPP"
df['Target'] = df['Target'].map({'Graduate':1,'Dropout':0,'Enrolled':2})

# %% id="SiOWO_UlwokR"
df = df[df['Target']!= 2]

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="ubY-6IiXwtIH" outputId="9ce075a6-e9c9-4ecf-879b-a84cfab5c575"
df['Target'].value_counts().plot(kind='barh')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="s3zBCWTzwxE6" outputId="159d4a96-e357-4f23-fdac-48bd1fb9ec7c"
nfeature = len(df.columns)
cols = int(math.ceil(math.sqrt(nfeature)))
rows = int(math.ceil(nfeature / cols))

width_per_plot = 5
height_per_plot = 4
plt.figure(figsize=(cols * width_per_plot, rows * height_per_plot))

for i, col in enumerate(df.columns):
    plt.subplot(rows, cols, i + 1)
    sns.boxplot(df[col])
    plt.title(f"Feature: {col}")

plt.tight_layout()
plt.show()

# %% id="Egd_4Ri0yy4a"
X = df.drop('Target',axis=1)
y = df['Target']

# %% id="kVlUWORfzbt2"
features_name = X.columns

# %% id="BUF1i2QTy12A"
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# %% id="Vata3puMzf1E"
X = pd.DataFrame(X,columns=features_name)

# %% colab={"base_uri": "https://localhost:8080/"} id="_tpbl0UIAhN3" outputId="da02c9ce-30b7-4dee-97ce-fca6d4c3fb0c"
print(f'Number of features {len(X.columns)}')

# %% colab={"base_uri": "https://localhost:8080/"} id="sGVdmoTiziG0" outputId="1e0a2fbb-c5c2-4eb0-ff1e-15cb085fac57"
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(score_func=chi2,k=20)
X_new = selector.fit_transform(X,y)
print(X_new.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="kHlxMoV2BL0s" outputId="77192ede-f4e5-4b7b-e4e8-bcc016a486d0"
# Get selected feature indices
selected_indices = selector.get_support(indices=False)
print("Selected feature indices:", selected_indices)

# %% colab={"base_uri": "https://localhost:8080/"} id="hFVq3dvHEeTp" outputId="59174050-3fdb-4870-8cff-ad080509a832"
print(X.columns[selected_indices])

# %% id="xd0A_PTPFl9b"

# %% id="wJVBcliyD4fK"
data = np.column_stack((X_new,y))
data = pd.DataFrame(data,columns=X.columns[selected_indices].tolist()+['Target'])

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="jk3_DxyVFuLH" outputId="add61849-1d88-4bcd-91a4-b95aa2715585"
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

# %% id="ee6Mm9DGGAi4"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# %% colab={"base_uri": "https://localhost:8080/", "height": 166} id="fEdTZxmfGQys" outputId="5cdf794f-89ef-4ace-e3ed-b238f2569736"
# code to classification using Dicision tree using GridSearch

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Random Forest
param_grid = {
    'max_depth': np.arange(1, 15, 3),
    'min_samples_split': np.arange(2, 10, 2),
    'min_samples_leaf': np.arange(1, 10, 2),
    'criterion': ['gini', 'entropy'], # Added criterion
    'splitter': ["best", "random"],
}

# Initialize the Random Forest classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="4_Rmej_CHAO8" outputId="c1b286c3-629b-4632-ace9-4677dd447732"
# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# %% colab={"base_uri": "https://localhost:8080/"} id="El4MZEdDHCHw" outputId="f7b0c9e7-8b46-4773-f184-2e353e8812fe"
# Evaluate the best model on the test set
best_rf_model = grid_search.best_estimator_
test_accuracy = best_rf_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# %% colab={"base_uri": "https://localhost:8080/"} id="oqnFa05xTSwi" outputId="63c836e9-7d1c-466f-e74d-dba2988b383b"
from sklearn.metrics import classification_report
y_pred = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="bjqAkRwVTWXj" outputId="73843156-f0c7-48e5-d2f1-d99e43d49ffb"
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 176} id="PxGd5FaLTx0S" outputId="8881390c-fdb0-4dca-a4f0-50ae154f02ca"
rf_params = {
    key: value for key, value in best_params.items()
}

# %% id="-zMOqpOwT6-V"
# prompt: code to use the best params getting from DT to build Random forst classifier

from sklearn.ensemble import RandomForestClassifier

# Use the best parameters from the Decision Tree GridSearchCV to initialize the Random Forest
rf_classifier = RandomForestClassifier(random_state=42, **best_params)

# Train the Random Forest classifier
rf_classifier.fit(X_train, y_train)

# Evaluate the Random Forest classifier
rf_test_accuracy = rf_classifier.score(X_test, y_test)
print("Random Forest Test Accuracy:", rf_test_accuracy)

y_rf_pred = rf_classifier.predict(X_test)
print(classification_report(y_test, y_rf_pred))

cm_rf = confusion_matrix(y_test, y_rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d')
plt.show()


# %% id="86hPZwXYTbWJ"
from sklearn.ensemble import RandomForestClassifier
rf_params = {
    'n_estimators': [100, 200, 300],
}
rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
