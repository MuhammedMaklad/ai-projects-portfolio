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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv("adult.csv")

# %%
df.shape

# %%
df.head()

# %%
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names
df.head()

# %%
df.info()

# %%
df.head(15)

# %%
columns_name = df.columns.tolist()
columns_name

# %%
for column in df.columns.tolist():
    print(column)
    print(f"Null in dataset {df[column].isnull().sum()}")
    print(f"Na in dataset {df[column].isna().sum()}")
    print(f"unique in dataset {df[column].nunique()}")
    print(f"dublicates in dataset {df[column].duplicated().sum()}")
    print("-" * 40)

# %% [markdown]
# # Exploring categorical variables

# %%
categorical = [cat for cat in df.columns if df[cat].dtype == "O"]
categorical

# %%
# view categorical variables
df[categorical]

# %% [markdown]
# # Summary of Categorical Variables
# * there are 9 categorical variables
# * income is the target Variable
#

# %% [markdown]
#

# %% [markdown]
# # Explore problems within categorical variables
# - first, will explore the categorical variables
#
# # missing values

# %%
for cat in categorical:
    print(f"Missing values in {cat} column: {df[cat].isnull().sum()}")
    print("-" * 40)

# %%
df[categorical].isnull().sum()

# %% [markdown]
# # frequency counts of categorical variables
# - now, i will check the frequency counts of categorical variables

# %%
for cat in categorical:
    print(df[cat].value_counts())
    print("-" * 20)

# %% [markdown]
# now, you can see that several variables like `workclass`, `accupation`, `native_country` which contain `?`
#

# %%
df.workclass.unique()

# %%
df.workclass.value_counts()

# %% [markdown]
# we see that there are `1836` values encoded as `?`, we will replace it to `NaN`

# %%
# replace ? valuess in workclass to NaN
df['workclass'].replace(' ?', np.NaN, inplace=True)

# %%
df.workclass.value_counts()

# %%
# replace ? valuess in workclass to NaN
df['occupation'].replace(' ?', np.NaN, inplace=True)
df.occupation.value_counts()

# %%
# replace ? valuess in workclass to NaN
df['native_country'].replace(' ?', np.NaN, inplace=True)
df['native_country'].value_counts()

# %% [markdown]
# # checking missing value in categorical variables

# %%
df[categorical].isnull().sum()

# %% [markdown]
# # Number of labels: Cardinality
# - the number of labels within a categorical variable is known as `Cardinality`
# - high `cardinality` may pose some serious problems in machine learning models

# %% [markdown]
# # Lets Check for Cardinality

# %%
for cat in categorical:
    print(cat, ' contains ', len(df[cat].unique()), ' labels')
    print("-" * 40)

# %% [markdown]
# we can see that native_country column contains relatively large number of labels as compared to other columns

# %% [markdown]
# # Exploring the Numerical Variables

# %%
numerical = [numeric for numeric in df.columns if df[numeric].dtype != "O"]
numerical

# %%
df[numerical].head()

# %%
df[numerical].info()

# %% [markdown]
# # Summary of numerical variables
# - there are 6 numerical variables
# - all of the numerical variables are of discrete data type

# %%
df[numerical].isnull().sum()

# %% [markdown]
# # Declare feature vector and target variables

# %%
X = df.drop(['income'],axis=1)
y = df['income']

# %% [markdown]
# # Split data into separate training and test set

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# %%
# check for shape of x-train and x-test
X_train.shape, X_test.shape

# %% [markdown]
# # Feature Engineering

# %%
X_train.dtypes

# %% [markdown]
# # Engineering missing values in categorical variables

# %%
categorical = [col for col in X_train.columns if X_train[col].dtype == 'O' ]
categorical

# %% [markdown]
#

# %%
numerical = [col for col in X_train.columns if X_train[col].dtype != 'O']
numerical

# %%
# print percentages of missing values in the categorical variables in trainning set
X_train[categorical].isnull().mean()

# %%
# print categorical variables with missing data
for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))

# %%
# impute missing categorical variables with the most frequent value
for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)  
    df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)

# %%
X_train[categorical].isnull().sum()

# %%
X_test[categorical].isnull().sum()

# %% [markdown]
# # Encode categorical variables

# %%
# print categorical variables
categorical

# %%
# !pip install category_encoders


# %%
import category_encoders as ce

# %%
# encode categorical variables with one-hot encoding

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship',
                                 'race', 'sex', 'native_country'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# %%
X_train.head()

# %% [markdown]
# # Feature Scaling 

# %%
cols = X_train.columns

# %%
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

# %%
X_train.head()

# %%
# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB

# instantiate the model
gnb = GaussianNB()


# %%
# apply a Gaussian with Grid Search and print results with best estimator
from sklearn.model_selection import GridSearchCV

# define parameters range
param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}

grid = GridSearchCV(
    estimator=gnb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
)
grid.fit(X_train, y_train)
