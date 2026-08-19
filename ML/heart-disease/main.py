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
# # Heart Disease Classification Dataset
#

# %%
import pandas as pd

# %%
# read database using pandas
df = pd.read_csv('Heart Attack.csv')

# %%
# show first 5 rows of database
df.head()

# %%
# show last 5 rows of database
df.tail()   

# %%
# info about database
df.info()

# %% [markdown]
# - there are 8 features & one target class
# - all features numerical

# %%
# get description about database
df.describe()

# %%
# get database columns names
df.columns

# %%
# Select Variable vector & target
X = df.drop('class',axis=1)
y = df['class']

# %%
# explore features only
X.head(7)

# %%
# checking for missing values in features
for col in X.columns:
    print(f"{col}: {X[col].isnull().sum()}")
    print(f"{col}: {X[col].isna().sum()}")
    print("-" *20)

# %%
# checking for outlier in features
X.describe()

# %%
for col in X.columns:
    Q1 = X[col].quantile(.25)
    Q3 = X[col].quantile(.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    
    outliers = X[(X[col]  < lower_bound) | (X[col] > upper_bound)]
    
    print(f"Feature: {col}")
    print(f"Number of Outliers: {outliers.shape[0]}")
    print("=" * 20)

# %%
# visualizations outliers
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))
for i, col in enumerate(X.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(X[col])
    plt.title(f'Boxplot for {col}')
    plt.tight_layout()
plt.show()

# %%
X.columns

# %%
features_contain_outliers = ['age','impluse', 'pressurehight','pressurelow','glucose','kcm','troponin'] 

# %%
import numpy as np


# %%
# treat outliers using imputing Outliers
def replace_outliers(date_frame):
    df_clean = date_frame.copy()
    for col in features_contain_outliers:
        Q1 = df_clean[col].quantile(.25)
        Q3 = df_clean[col].quantile(.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    return df_clean



# %%
X = replace_outliers(X)

# %%
plt.figure(figsize=(20, 15))
for i, col in enumerate(X.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(X[col])
    plt.title(f'Boxplot for {col}')
    plt.tight_layout()
plt.show()

# %% [markdown]
# - no outlier in out featurs

# %%
# perform scaling for features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# %%
X = pd.DataFrame(X, columns=['age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose',
       'kcm', 'troponin'])

# %%
X.head()

# %%
y.head()

# %%
y.unique()

# %%
# convert target class variable into numerical
y = y.map({'positive': 1, 'negative': 0})

# %%
# check for missing value in target
y.isnull().sum()

# %%
y.head()

# %%
# spliting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train.shape

# %%
y_train.shape

# %%
# checking if data is balanced or not
y.value_counts()

# %% [markdown]
# - data is extremally balanced

# %%
# build our model using discussion tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# %%
# Initialize the model
dt = DecisionTreeClassifier(random_state=42)


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
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, 
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
# Visualize the Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(best_dt, feature_names=X.columns, class_names=['negative', 'positive'], filled=True)
plt.show()

