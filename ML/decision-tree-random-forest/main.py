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
from traceback import print_stack

import pandas as pd

# %%
# read dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# %%
# explore dataset
df.head(10)

# %%
import seaborn as sns

# %%
sns.countplot(x=df['Attrition'],data=df)

# %% [markdown]
# - dataset is imbalanced

# %% [markdown]
# ## Pre - Processing

# %%
# Checking for null or missing values in dataset
print(df.isna().sum())

# %% [markdown]
# - there no missing values in dataset

# %%
# info about dataset
df.info()

# %% [markdown]
# - dataset contains numerical & categorical values

# %%
## Visualization of features Dependence on Target
import matplotlib.pyplot as plt

# %%
#Visualize numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
for feature in numerical_features:
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Attrition',y = feature, data=df)
    plt.title(f'Box Plot of {feature} byAttrition')
    plt.show()

# %% [markdown]
# - we can remove both features `EmployeeNumber`, `StanderdHours` , `EmployeeCount` Target Variable don't depend on

# %%
# visualize categorical features
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    if (feature != 'Attrition'):
        plt.figure(figsize=(10,6))
        sns.countplot(x=feature, hue='Attrition',data=df)
        plt.title(f'Count plot of {feature} by Attrition')
        plt.xticks(rotation=50)
        plt.show()

# %% [markdown]
# - we can remove `Over18` feature, Target don't depend on

# %%
# Dropping feature that Target don't depend on
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)

# %%
df.head()

# %%
# get unique values in each categorical variable
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    print(f"{feature} contains {df[feature].nunique()} values")
    print("-" * 20)

# %%
df['Attrition'].head()

# %%
# encode target features 
df['Attrition'] = df['Attrition'].astype('category').cat.codes

# %%
df['Attrition'].head(10)

# %%
categorical_features = [feature for feature in categorical_features if feature != 'Attrition']

# %%
categorical_features

# %%
# encode categorical features
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for feature in categorical_features:
    df[feature] = encoder.fit_transform(df[feature])
    

# %%
df[categorical_features].head()

# %%
numerical_features = df.select_dtypes(include=['int64']).columns

# %%
numerical_features

# %%
# checking for outlier in numerical features
for feature in numerical_features:
    Q1 = df[feature].quantile(.25)
    Q3 = df[feature].quantile(.75)
    
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    
    outliers = df[(df[feature]  < lower_bound) | (df[feature] > upper_bound)]

    print(f'Feature : {feature}')
    print(f'Number of outliers : {outliers.shape[0]}')
    print("-" * 20)

# %%
# visualizations outliers
import math
# Calculate the number of rows and columns needed for subplots
num_features = len(numerical_features)
rows = math.ceil(num_features / 3)  # Number of rows needed (3 columns per row)

plt.figure(figsize=(20, rows * 5))
for i, col in enumerate(numerical_features):
    plt.subplot(rows, 3, i+1)
    sns.boxplot(df[col])
    plt.title(f'Boxplot for {col}')
    plt.tight_layout()
plt.show()

# %%
# Treat outlier using imputing outliers
for feature in numerical_features:
    Q1 = df[feature].quantile(.25)
    Q3 = df[feature].quantile(.75)
    
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    
    df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)


# %%
# Calculate the number of rows and columns needed for subplots
num_features = len(numerical_features)
rows = math.ceil(num_features / 3)  # Number of rows needed (3 columns per row)

plt.figure(figsize=(20, rows * 5))
for i, col in enumerate(numerical_features):
    plt.subplot(rows, 3, i+1)
    sns.boxplot(df[col])
    plt.title(f'Boxplot for {col}')
    plt.tight_layout()
plt.show()

# %% [markdown]
# - there are not outlires in features

# %%
# Split data to features & target
X = df.drop('Attrition',axis=1)
y = df['Attrition']

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.3)

# %%
X_train.shape

# %%
X_test.shape

# %%
y_train.shape

# %%
y_test.shape

# %%
# handle imbalanced data
from imblearn.combine import SMOTEENN
smoteenn = SMOTEENN()

X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)

# %%
X_train_resampled.shape

# %%
y_train_resampled.shape

# %%
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


# %%
def get_score(model,x_data, y_data, data='training'):
    y_pred = model.predict(x_data)
    print(f'Accuracy for {data} data {accuracy_score(y_data, y_pred)}')
    print('-' * 20)
    print(f'confusion matrix \n{confusion_matrix(y_data, y_pred)}')
    print('-' * 20)
    clf_report = pd.DataFrame(classification_report(y_data, y_pred, output_dict=True))
    print(f'classification report \n{clf_report}')
    print('-' * 20)
    print()


# %%
# building model using Decision Tree 
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X=X_train_resampled, y=y_train_resampled)

# %%
get_score(dtc, X_train_resampled, y_train_resampled)
get_score(dtc, X_test, y_test,data='testing');

# %% [markdown]
# - there are overfitting

# %%
# build Decision Tree model using hypertuning parameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
tree_cv = GridSearchCV(
    dtc,
    param_grid,
    scoring="accuracy",
    verbose=1,  # print progress
    cv=5 #kfold count
)

tree_cv.fit(X_train_resampled, y_train_resampled)
print(f'Best params\n {tree_cv.best_params_}')
best_model = tree_cv.best_estimator_
get_score(best_model, X_train_resampled, y_train_resampled)
get_score(best_model, X_test, y_test)

# %%
# fitting model in X_train, y_train
tree_cv.fit(X_train, y_train)
print(f'Best params\n {tree_cv.best_params_}')
best_model = tree_cv.best_estimator_
get_score(best_model, X_train, y_train)
get_score(best_model, X_test, y_test)

# %%
# building model using RandomForest
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)

get_score(rf_clf, X_train, y_train)
get_score(rf_clf,X_test, y_test, data='testing')

# %% [markdown]
# - there are overfitting

# %%
# Random Forest hyperparameter tunning
# using Randomize search
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

get_score(rf_clf, X_train, y_train)
get_score(rf_clf, X_test, y_test, data='testing')

# %% [markdown]
# - still there are overfitting

# %%
# apply gridsearch and cross validation
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

get_score(rf_clf, X_train, y_train)
get_score(rf_clf, X_test, y_test, data='testing')
