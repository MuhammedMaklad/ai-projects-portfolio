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
# - Student Performance datasets `https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression?select=Student_Performance.csv`
# - Variables [Hours Studied, Previous Scores, Extracurricular Activities, Sleep Hours, Sample Question Papers Practiced]
# - Target Variable [Performance Index]

# %%
import pandas as pd
# from keras.src.legacy.backend import categorical_focal_crossentropy

# %%
# Read Dataset using pandas
df = pd.read_csv("Student_Performance.csv")

# %%
print(df.head())

# %%
df.tail()

# %% [markdown]
# # Dataset Datails & info

# %%
df.info()

# %% [markdown]
# # Explore Categorical Variables

# %%
# Get Categorical Variables
categorical = [var for var in df.columns if df[var].dtype == 'O']
categorical

# %%
# explore Categorical Variables
df[categorical].head(10)

# %%
# check that categorical variable contains non values
df[categorical].isnull().sum()

# %%
# get labels in Categorical Variables
df[categorical].nunique()

# %%
# get frequencies for each value in Categorical Variables
df[categorical].value_counts()

# %% [markdown]
# # Explore Numerica Variables

# %%
# Get numerical Variables in dataset
numerical = [var for var in df.columns if df[var].dtype != 'O']
numerical

# %%
# Explore Numerica Variables
df[numerical].head(10)

# %%
# Check if numerical Variables contain missing values
for var in numerical:
    print(f"{var} contains {df[var].isnull().sum()} null values")
    print(f"{var} contains {df[var].isna().sum()} na values")
    print("-" * 40)

# %%
# Get description about numerical Variables
# detect outlier using statistical
df[numerical].describe()

# %%
# checking outlier in numerical variables
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
for i in range(len(numerical)):
    plt.subplot((len(numerical) + 1) // 2, 2, i + 1)  # Adjust subplot configuration
    fig = sns.boxplot(df[numerical[i]])
    fig.set_title(f'{numerical[i]} Boxplot')
    fig.set_ylabel(f'{numerical[i]}')

# %% [markdown]
# - no outlier in numerical variables

# %% [markdown]
# # define vector Variable and target variable

# %%
df.columns

# %%
X = df.drop(columns=['Performance Index'], axis=1)
y = df['Performance Index']

# %%
print(X.head())

# %%
y.head()

# %%
categorical

# %%
df[categorical]

# %%
df[categorical[0]].nunique()

# %%
df[categorical[0]].unique()

# %%
# encode categorical variable 'Extracurricular Activities'
import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['Extracurricular Activities'])
X = encoder.fit_transform(X)

# %%
X.head(10)

# %%
# Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# %%
X

# %%
# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train.shape

# %%
# Build our model 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# %%
# Coefficient Calculations
print(f'intercept {model.intercept_}')
print(f"Coefficient {model.coef_}")

# %%
# predict model in test set
y_pred = model.predict(X_test)

# %%
# prompt: compute the MSE and R square
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r2)
