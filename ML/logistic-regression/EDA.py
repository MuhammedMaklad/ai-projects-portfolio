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
# read dataset using pandas
# by pandas convert dataset to dataframe, that enable manipulation for dataset
df = pd.read_csv("weatherAUS.csv")

# %%
# Explor dataset
# show first 10 rows
df.head(10)

# %%
# explor the last 5 rows of dataset
df.tail()

# %%
# get shape of dataset
df.shape

# %% [markdown]
# - dataset contain 23 features
# - dataset contain 145460 records

# %%
# get info about dataset
df.info()

# %%
# get description about dataset
df.describe()

# %%
# get columns names of dataset
df.columns

# %%
# get na and null variables in dataset
for column in df.columns:
    print(f"{column} contains {df[column].isna().sum()} na values")
    print(f"{column} contains {df[column].isnull().sum()} null values")
    print("--" * 40)

# %%
# find categorical variables in data set
categorical = [var for var in df.columns if df[var].dtype == 'O']
categorical

# %%
df[categorical].head()

# %%
# check missing values in categorical variables
df[categorical].isnull().sum()

# %%
categorical_missing = [cat for cat in categorical if df[cat].isnull().sum()!=0]
categorical_missing

# %%
df[categorical_missing].isnull().sum()

# %%
# get frequencies of categorical variables
for cat in categorical:
    print(f"{df[cat].value_counts()}")
    print("--" * 20)

# %%
# view frequency distribution of categorical variables
n = len(df)
for var in categorical:
    print(f"{var} frequency distribution: {df[var].value_counts() / n}")
    print("--" * 40)

# %%
categorical

# %%
# check for cardinality in categorical variables
for var in categorical:
    print(f"{var} contains {df[var].nunique()} labels")

# %% [markdown]
# - cardinality for date is high that leads to problems in machine learning

# %% [markdown]
# # Features Engineering to Date Variable

# %%
# convert date from text(string) to datetime to can split year, month,and day
df['Date'] =  pd.to_datetime(df['Date'])

# %%
df['Date'].dtype

# %%
# extract year from dat
df['Year'] = df['Date'].dt.year
df['Year'].head()

# %%
# extract month from date
df['Month'] = df['Date'].dt.month
df['Month'].head()

# %%
# extract day from date
df['Day'] = df['Date'].dt.day
df['Day'].head()

# %%
# lets Drop Date columns not need any more
df.drop('Date', axis=1,inplace=True)
df.head()

# %%
df.shape

# %%
df.info()

# %%
# lets explore categorical variables
categorical = [var for var in df.columns if df[var].dtype == 'O']
categorical

# %%
print(f"we have {len(categorical)} variables in dataset")

# %%
df[categorical].isnull().sum()

# %% [markdown]
# - `WindGusterDir`, `WindDir9am`, `WindDir3pm`, `RainToday` has missing values

# %% [markdown]
# # Explore Location Variable

# %%
df['Location'].head()

# %%
# get number of labels in Location Variable
print(f"Location contain {df['Location'].nunique()} labels")
print(df['Location'].unique())


# %%
# Lets encode the location to be numerical values
# using hot encoding
# get k-1 dummy variables after One Hot Encoding

pd.get_dummies(df.Location,drop_first=True).head()

# %%
df['Location'].head()

# %%
categorical

# %%
df[categorical].isnull().sum()

# %% [markdown]
# # Explore `WindGusDir` Variable

# %%
# check null values
print(f"WindGustDir Variable contains {df['WindGustDir'].isnull().sum()}")

# %%
# Get number of labels in This Feature
print(f"WindGustDir Variable contains {df['WindGustDir'].nunique()} labels")

# %%
# get unique variable in this feature
df['WindGustDir'].unique()

# %%
# check frequency distribution for values in feature
df['WindGustDir'].value_counts()

# %%
df['WindGustDir'].value_counts()/ len(df['WindGustDir'])

# %%
# do hot one ecoding for feature
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()

# %% [markdown]
# Avoiding Multicollinearity: Dropping the first category prevents perfect multicollinearity, making the model more stable and interpretable.
# Reference Category: The dropped category serves as the reference level against which other categories are compared.

# %%
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)

# %%
categorical

# %% [markdown]
# # Explore `WindDir9am` Variable

# %%
# check null in feature
print(f"WindDir9am contains {df['WindDir9am'].isnull().sum()} NaN")

# %%
# get frequency for each value
df['WindDir9am'].value_counts()

# %%
# get number of labels in feature
print(f"WindDir9am contains {df['WindDir9am'].nunique()} labels")

# %%
# get labels in feature
df['WindDir9am'].unique()

# %%
# perform One Hot Encoding
# remove first column to make model more generlalized
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()

# %%
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)

# %%
categorical

# %% [markdown]
# # Explore the WindDir3pm Variable

# %%
# Check NaN values in feature
print(f"WindDir3pm contains {df['WindDir3pm'].isnull().sum()} NaN")

# %%
# Get Frequency for each value in feature
df['WindDir3pm'].value_counts()

# %%
# get number of labels in feature
print(f"WindDir3pm contains {df['WindDir3pm'].nunique()} labels")

# %%
# Get unique labels in feature
df['WindDir3pm'].unique()

# %%
# do One-Hot Encoding using dummy
pd.get_dummies(df['WindDir3pm'],drop_first=True, dummy_na=True).head()

# %%
pd.get_dummies(df['WindDir3pm'],drop_first=True, dummy_na=True).sum(axis=0)

# %%
categorical

# %% [markdown]
# # Explore `RainToday` Variable

# %%
# check NaN values in feature
print(f"RainToday contains {df['RainToday'].isnull().sum()} NaN")

# %%
# Get Frequency for each value in feature
df['RainToday'].value_counts()

# %%
df['RainToday'].unique()

# %%
# perform one-Hot Encoding using dummy
pd.get_dummies(df['RainToday'], drop_first=True, dummy_na=True).head()

# %%
# perform one-Hot Encoding using dummy
pd.get_dummies(df['RainToday'], drop_first=True, dummy_na=True).sum(axis=0)

# %% [markdown]
# # Explore `Numeric Variables` in Dataset

# %%
numerical = [var for var in df.columns if df[var].dtype != 'O']
numerical

# %%
# Check null values in feature
print(df[numerical].isnull().sum())

# %%
print("Features that have null values")
print([var for var in numerical if df[var].isnull().sum() != 0])

# %% [markdown]
# # Check for Outlier in Variables

# %%
# View Summary Statistics for featuers
print(round(df[numerical].describe(),2))

# %%
import matplotlib.pyplot as plt

# %%
# draw boxplots to visualize outliers
# boxplots show the distribution of a set of numerical data through their quartiles
plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')

# %% [markdown]
# # Check Distribution of Variables

# %%
# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')

# %%
# find outliers for Rainfall variable
cols = ['Rainfall', 'Evaporation', 'WindSpeed9am', 'WindSpeed3pm']
for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (3 * IQR)
    upper_bound = Q3 + (3 * IQR)
    print(f'{col} outliers are values < {lower_bound} or > {upper_bound}')
    print("--" * 40)

# %% [markdown]
# # Declare Feature Vector and Target Variable

# %%
X = df.drop(columns=['RainTomorrow'],axis=1)
y = df['RainTomorrow']

# %%
print(y.head())

# %%
X.head()

# %%
# explore target variable
y

# %%
y.value_counts(
    dropna=False
)

# %% [markdown]
# - there are missing values in the target variable

# %%
# fix issues with missing values using most frequent 
y.fillna(
    y.mode()[0],
    inplace=True
)

# %%
y.value_counts(dropna=False)

# %%
y.head(10)

# %%
# replace No with 0, Yes with 1
y.replace({"Yes":1, "No":0}, inplace=True)

# %%
y.head()

# %% [markdown]
# # Split data into Two Set one for Training and other to Testing

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)

# %%
X_train.head()

# %%
y_test

# %%
print(X_train.shape)
print(X_test.shape)

# %%
y_test.value_counts()

# %% [markdown]
# - data in y_test imbalanced this will cause some problems in model

# %% [markdown]
# # Feature Engineering 

# %%
# Check data types in training data
X_train.dtypes

# %%
# Show categorical features
categorical = [var for var in X_train.columns if X_train[var].dtype == 'O']
categorical

# %%
# Show numerical features
numerical = [var for var in X_train.columns if X_train[var].dtype != 'O']
numerical

# %%
# Checking for missing values X_train
X_train[numerical].isnull().sum()

# %%
# Checking for numerical missing values in X_test
X_test[numerical].isnull().sum()

# %%
# print percentage of missing values in the numerical variables in training set
for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))

# %%
# dealing with missing values using median because there are outliers in our data
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median = X_train[col].median()
        df1[col].fillna(col_median, inplace=True)

# %%
# checking for missing values in X_train
X_train[numerical].isnull().sum()

# %%
# checking for missing values in X_test
X_test[numerical].isnull().sum()

# %% [markdown]
# # Engineering for missing values in categorical variables

# %%
categorical

# %%
# Print percentages for missing values in categorical variables
X_train[categorical].isnull().mean()

# %%
# Print percentages for missing values in categorical variables that have missing values
for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))

# %%
# Now, Dealing with missing data in categorical variables
# we will treat it using most frequency method
for df1 in [X_train, X_test]:
    for col in categorical:
        col_mode = X_train[col].mode()[0]
        df1[col].fillna(col_mode, inplace=True)

# %%
# Now lets check for missing values in our dataset
X_train[categorical].isnull().sum()

# %%
X_test[categorical].isnull().sum()

# %%
print(X_train.isnull().sum())

# %%
print(X_test.isnull().sum())

# %% [markdown]
# # Dealing with Outliers in numerical variables

# %%
import numpy as np
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])  # df3["rainfalll"]= 3

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

# %% [markdown]
# # Encoding categorical variables

# %%
X_train[categorical].nunique()

# %%
for col in categorical:
    print(f"\nUnique values in {col}:")
    print(X_train[col].unique())

# %%
from category_encoders import BinaryEncoder
encoder = BinaryEncoder(cols=['RainToday'])


X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# %%
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location),
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)

# %%
X_train.head()

# %%
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location),
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)

# %%
X_test.head()

# %% [markdown]
# # Feature Scaling

# %%
print(X_train.describe())

# %%
cols = X_train.columns
cols

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %%
X_train = pd.DataFrame(X_train, columns=[cols])

# %%
X_test = pd.DataFrame(X_test, columns=[cols])

# %%
X_train.shape

# %%
X_test.shape

# %%
X_train.describe()

# %% [markdown]
# # Model Training

# %%
# training logistic model in training set
from sklearn.linear_model import LogisticRegression

# instantiate the model
logreg = LogisticRegression(solver='liblinear',random_state=0)

# fit the model
logreg.fit(X_train, y_train)

# %% [markdown]
# # Predict Result

# %%
y_pred = logreg.predict(X_test)

# %% [markdown]
# - predict_prop method
# - predict_proba method gives the probabilities for the target variable(0 and 1) in this case, in array form.
# - 0 is for probability of no rain and 1 is for probability of rain.
#

# %%
# probability of getting output as 0 - no rain
logreg.predict_proba(X_test)[:,0]

# %%
# probability of getting output as 1 - no rain
logreg.predict_proba(X_test)[:, 1]

# %% [markdown]
# # Check Accuracy

# %%
from sklearn.metrics import accuracy_score

print(f"model accuracy {accuracy_score(y_test, y_pred)}")

# %%
y_train_pred = logreg.predict(X_train)

# %%
print(f"Training set accuracy {accuracy_score(y_train,y_train_pred)}")
