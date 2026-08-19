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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2668, "status": "ok", "timestamp": 1724749441024, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="17D1A4-9rzq1" outputId="f31fb351-bbae-4ec1-cab9-07e60ad52efe"
# from google.colab import drive
# drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1724749441024, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="MCnFy801r4rA" outputId="d11a860a-abc1-485b-f287-65ca24ce2457"
# # cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Logistic regression

# %% [markdown] id="Hc4rAFuK5CKT"
# <a class="anchor" id="0"></a>
# # **Logistic Regression Classifier Tutorial with Python**
#
#
#
# In this kernel, I implement Logistic Regression with Python and Scikit-Learn. I build a Logistic Regression classifier to predict whether or not it will rain tomorrow in Australia. I train a binary classification model using Logistic Regression.

# %% [markdown] id="ZPoZkL0H5HYR"
# https://www.kaggle.com/code/prashant111/logistic-regression-classifier-tutorial/input

# %% [markdown] id="pDocywYK5CKZ"
# <a class="anchor" id="0.1"></a>
# # **Table of Contents**
#
#
# 1.	[Introduction to Logistic Regression](#1)
# 2.	[Logistic Regression intuition](#2)
# 3.	[Assumptions of Logistic Regression](#3)
# 4.	[Types of Logistic Regression](#4)
# 5.	[Import libraries](#5)
# 6.	[Import dataset](#6)
# 7.	[Exploratory data analysis](#7)
# 8.	[Declare feature vector and target variable](#8)
# 9.	[Split data into separate training and test set](#9)
# 10.	[Feature engineering](#10)
# 11.	[Feature scaling](#11)
# 12.	[Model training](#12)
# 13.	[Predict results](#13)
# 14.	[Check accuracy score](#14)
# 15.	[Confusion matrix](#15)
# 16.	[Classification metrices](#16)
# 17.	[Adjusting the threshold level](#17)
# 18.	[ROC - AUC](#18)
# 19.	[k-Fold Cross Validation](#19)
# 20.	[Hyperparameter optimization using GridSearch CV](#20)
# 21.	[Results and conclusion](#21)
# 22. [References](#22)
#

# %% [markdown] id="yXdp2sFT5CKa"
# # **1. Introduction to Logistic Regression** <a class="anchor" id="1"></a>
#
#
# [Table of Contents](#0.1)
#
#
# When data scientists may come across a new classification problem, the first algorithm that may come across their mind is **Logistic Regression**. It is a supervised learning classification algorithm which is used to predict observations to a discrete set of classes. Practically, it is used to classify observations into different categories. Hence, its output is discrete in nature. **Logistic Regression** is also called **Logit Regression**. It is one of the most simple, straightforward and versatile classification algorithms which is used to solve classification problems.

# %% [markdown] id="QRraVa9S5CKa"
# # **2. Logistic Regression intuition** <a class="anchor" id="2"></a>
#
#
# [Table of Contents](#0.1)
#
#
# In statistics, the **Logistic Regression model** is a widely used statistical model which is primarily used for classification purposes. It means that given a set of observations, Logistic Regression algorithm helps us to classify these observations into two or more discrete classes. So, the target variable is discrete in nature.
#
#
# The Logistic Regression algorithm works as follows -

# %% [markdown] id="iTE6jI1K5CKb"
# ## **Implement linear equation**
#
#
# Logistic Regression algorithm works by implementing a linear equation with independent or explanatory variables to predict a response value. For example, we consider the example of number of hours studied and probability of passing the exam. Here, number of hours studied is the explanatory variable and it is denoted by x1. Probability of passing the exam is the response or target variable and it is denoted by z.
#
#
# If we have one explanatory variable (x1) and one response variable (z), then the linear equation would be given mathematically with the following equation-
#
#     z = β0 + β1x1    
#
# Here, the coefficients β0 and β1 are the parameters of the model.
#
#
# If there are multiple explanatory variables, then the above equation can be extended to
#
#     z = β0 + β1x1+ β2x2+……..+ βnxn
#     
# Here, the coefficients β0, β1, β2 and βn are the parameters of the model.
#
# So, the predicted response value is given by the above equations and is denoted by z.

# %% [markdown] id="sR4AW1kN5CKb"
# ## **Sigmoid Function**
#
# This predicted response value, denoted by z is then converted into a probability value that lie between 0 and 1. We use the sigmoid function in order to map predicted values to probability values. This sigmoid function then maps any real value into a probability value between 0 and 1.
#
# In machine learning, sigmoid function is used to map predictions to probabilities. The sigmoid function has an S shaped curve. It is also called sigmoid curve.
#
# A Sigmoid function is a special case of the Logistic function. It is given by the following mathematical formula.
#
# Graphically, we can represent sigmoid function with the following graph.

# %% [markdown] id="zuLdVSJu5CKb"
# ### Sigmoid Function
#
# ![Sigmoid Function](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

# %% [markdown] id="su82hwDF5CKc"
# ## **Decision boundary**
#
# The sigmoid function returns a probability value between 0 and 1. This probability value is then mapped to a discrete class which is either “0” or “1”. In order to map this probability value to a discrete class (pass/fail, yes/no, true/false), we select a threshold value. This threshold value is called Decision boundary. Above this threshold value, we will map the probability values into class 1 and below which we will map values into class 0.
#
# Mathematically, it can be expressed as follows:-
#
# p ≥ 0.5 => class = 1
#
# p < 0.5 => class = 0
#
# Generally, the decision boundary is set to 0.5. So, if the probability value is 0.8 (> 0.5), we will map this observation to class 1. Similarly, if the probability value is 0.2 (< 0.5), we will map this observation to class 0. This is represented in the graph below-

# %% [markdown] id="JxcLAenT5CKc"
# ![Decision boundary in sigmoid function](https://ml-cheatsheet.readthedocs.io/en/latest/_images/logistic_regression_sigmoid_w_threshold.png)

# %% [markdown] id="IdoFvJWy5CKc"
# ## **Making predictions**
#
# Now, we know about sigmoid function and decision boundary in logistic regression. We can use our knowledge of sigmoid function and decision boundary to write a prediction function. A prediction function in logistic regression returns the probability of the observation being positive, Yes or True. We call this as class 1 and it is denoted by P(class = 1). If the probability inches closer to one, then we will be more confident about our model that the observation is in class 1, otherwise it is in class 0.
#

# %% [markdown] id="Il4gj33k5CKc"
# # **3. Assumptions of Logistic Regression** <a class="anchor" id="3"></a>
#
#
# [Table of Contents](#0.1)
#
#
# The Logistic Regression model requires several key assumptions. These are as follows:-
#
# 1. Logistic Regression model requires the dependent variable to be binary, multinomial or ordinal in nature.
#
# 2. It requires the observations to be independent of each other. So, the observations should not come from repeated measurements.
#
# 3. Logistic Regression algorithm requires little or no multicollinearity among the independent variables. It means that the independent variables should not be too highly correlated with each other.
#
# 4. Logistic Regression model assumes linearity of independent variables and log odds.
#
# 5. The success of Logistic Regression model depends on the sample sizes. Typically, it requires a large sample size to achieve the high accuracy.

# %% [markdown] id="GFyJYXpB5CKd"
# # **4. Types of Logistic Regression** <a class="anchor" id="4"></a>
#
#
# [Table of Contents](#0.1)
#
#
# Logistic Regression model can be classified into three groups based on the target variable categories. These three groups are described below:-
#
# ### 1. Binary Logistic Regression
#
# In Binary Logistic Regression, the target variable has two possible categories. The common examples of categories are yes or no, good or bad, true or false, spam or no spam and pass or fail.
#
#
# ### 2. Multinomial Logistic Regression
#
# In Multinomial Logistic Regression, the target variable has three or more categories which are not in any particular order. So, there are three or more nominal categories. The examples include the type of categories of fruits - apple, mango, orange and banana.
#
#
# ### 3. Ordinal Logistic Regression
#
# In Ordinal Logistic Regression, the target variable has three or more ordinal categories. So, there is intrinsic order involved with the categories. For example, the student performance can be categorized as poor, average, good and excellent.
#

# %% [markdown] id="qVuNimvN5CKd"
# # **5. Import libraries** <a class="anchor" id="5"></a>
#
#
# [Table of Contents](#0.1)

# %% id="irEtL5255CKd"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
# %matplotlib inline

# %% id="V2GDR9C45CKe"
import warnings

warnings.filterwarnings('ignore')

# %% [markdown] id="vA-RzrxY5CKe"
# # **6. Import dataset** <a class="anchor" id="6"></a>
#
#
# [Table of Contents](#0.1)

# %% id="m1WjfJS35CKf"
data = 'weatherAUS.csv'

df = pd.read_csv(data)

# %% [markdown] id="al4qVHjn5CKf"
# # **7. Exploratory data analysis** <a class="anchor" id="7"></a>
#
#
# [Table of Contents](#0.1)
#
#
# Now, I will explore the data to gain insights about the data.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1724749442912, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="-8R9tHLs5CKf" outputId="c47f10c0-c7a9-493a-c94c-22bb87bb8fda"
# view dimensions of dataset

df.shape

# %% [markdown] id="vdpsO5PS5CKf"
# We can see that there are 142193 instances and 24 variables in the data set.

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1724749442912, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="9E6EzwRP5CKf" outputId="492130de-2a7e-482f-f9d4-0a22691f3ef8"
# preview the dataset

df.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1724749442912, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="f3ZHAh0YqcAx" outputId="eb665577-a164-44c5-fc8b-a90900684bfb"
df.tail()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1724749442912, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="wXlLTKA75CKg" outputId="6aea7407-dc82-40d2-866b-cc7f1c04420e"
col_names = df.columns

col_names

# %% [markdown] id="2-UZCKyW5CKg"
# ### Drop  RISK_MM variable
#
# It is given in the dataset description, that we should drop the `RISK_MM` feature variable from the dataset description. So, we
# should drop it as follows-

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 486, "status": "ok", "timestamp": 1724749443377, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="QUO-nx3s5CKg" outputId="88615f6b-af58-443f-f3ec-32cec05bbd74"
# view summary of dataset

df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 805} executionInfo={"elapsed": 40, "status": "ok", "timestamp": 1724749443378, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="9J0TLIH0rNcY" outputId="86e7c129-c038-4b12-9b65-2c748164e46a"
df.isnull().sum()

# %% [markdown] id="e5S2iwmt5CKg"
# ### Types of variables
#
#
# In this section, I segregate the dataset into categorical and numerical variables. There are a mixture of categorical and numerical variables in the dataset. Categorical variables have data type object. Numerical variables have data type float64.
#
#
# First of all, I will find categorical variables.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 39, "status": "ok", "timestamp": 1724749443378, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="bMiQDi_H5CKg" outputId="ec7fd070-c477-4b84-ef89-3b05fc3377af"
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']  #Object

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 320, "status": "ok", "timestamp": 1724749443661, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="GcgoypDd5CKh" outputId="b7f43610-aa22-42f2-8575-036e25aab4c8"
# view the categorical variables

df[categorical].head()

# %% [markdown] id="cl6PCOPd5CKh"
# ### Summary of categorical variables
#
#
# - There is a date variable. It is denoted by `Date` column.
#
#
# - There are 6 categorical variables. These are given by `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` and  `RainTomorrow`.
#
#
# - There are two binary categorical variables - `RainToday` and  `RainTomorrow`.
#
#
# - `RainTomorrow` is the target variable.

# %% [markdown] id="SLBBljPF5CKh"
# ## Explore problems within categorical variables
#
#
# First, I will explore the categorical variables.
#
#
# ### Missing values in categorical variables

# %% colab={"base_uri": "https://localhost:8080/", "height": 303} executionInfo={"elapsed": 638, "status": "ok", "timestamp": 1724749443980, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="n68Xt9hC5CKi" outputId="5014bb2e-ed27-495c-95b7-851734ae33a8"
# check missing values in categorical variables

df[categorical].isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 350, "status": "ok", "timestamp": 1724749444308, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="_3MZI4jW5CKi" outputId="8bc2626e-4511-465f-86ca-0347a921a8eb"
# print categorical variables containing missing values

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())

# %% [markdown] id="xhhjNKKg5CKi"
# We can see that there are only 4 categorical variables in the dataset which contains missing values. These are `WindGustDir`, `WindDir9am`, `WindDir3pm` and `RainToday`.

# %% [markdown] id="cqMtFfoy5CKj"
# ### Frequency counts of categorical variables
#
#
# Now, I will check the frequency counts of categorical variables.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 333, "status": "ok", "timestamp": 1724749444634, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ZOkXBNHY5CKj" outputId="28b4cf8e-511e-4ccf-e4c8-0584b7754efe"
# view frequency of categorical variables

for var in categorical:

    print(df[var].value_counts())

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 365, "status": "ok", "timestamp": 1724749444990, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="l5TjHO4g5CKj" outputId="0b79dce3-6a62-4511-9cff-08618e3acbf4"
# view frequency distribution of categorical variables

for var in categorical:

    print(df[var].value_counts()/float(len(df)))

# %% [markdown] id="d-7WLFr85CKj"
# ### Number of labels: cardinality
#
#
# The number of labels within a categorical variable is known as **cardinality**. A high number of labels within a variable is known as **high cardinality**. High cardinality may pose some serious problems in the machine learning model. So, I will check for high cardinality.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 426, "status": "ok", "timestamp": 1724749445408, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="xARO_6HL5CKk" outputId="a3bf16ae-97b5-4fad-e45b-605016f4d3f0"
# check for cardinality in categorical variables

for var in categorical:

    print(var, ' contains ', len(df[var].unique()), ' labels')

# %% [markdown] id="qdMOFGEu5CKq"
# We can see that there is a `Date` variable which needs to be preprocessed. I will do preprocessing in the following section.
#
#
# All the other variables contain relatively smaller number of variables.

# %% [markdown] id="2MWFaFUa5CKq"
# ### Feature Engineering of Date Variable

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1724749445408, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="io-Bb4_45CKq" outputId="739868e8-08ee-43fd-f89e-c3ebbcbbec47"
df['Date'].dtypes

# %% [markdown] id="YzlwFdyP5CKr"
# We can see that the data type of `Date` variable is object. I will parse the date currently coded as object into datetime format.

# %% id="ekY3KC7L5CKr"
# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1724749447119, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="weiCAFMTuzbn" outputId="614f5b1b-19ce-42d9-cf15-c1f4e42d2ea6"
df['Date'].dtypes

# %% colab={"base_uri": "https://localhost:8080/", "height": 458} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1724749447120, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="my7QTFOB5CKr" outputId="d362f401-59eb-4687-f741-67364589bf2a"
# extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head(100)

# %% colab={"base_uri": "https://localhost:8080/", "height": 711} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1724749447120, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="5V6VdEpF5CKr" outputId="f7cb9022-e7d6-4565-bbff-2293e91c57c2"
# extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head(20)

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1724749447120, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="QBGYg2un5CKs" outputId="cb00b6e3-8bd4-47cf-a060-72b3ad6d5bf1"
# extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 438, "status": "ok", "timestamp": 1724749447533, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="041c1h5W5CKs" outputId="dbafe041-d815-42ec-8be8-170c474032df"
# again view the summary of dataset

df.info()

# %% [markdown] id="n2ZlMO-D5CKs"
# We can see that there are three additional columns created from `Date` variable. Now, I will drop the original `Date` variable from the dataset.

# %% id="foyrgPHN5CKs"
# drop the original Date variable

df.drop('Date', axis=1, inplace = True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 1075, "status": "ok", "timestamp": 1724749448597, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="mTY9zwed5CKs" outputId="bb2cf66b-f491-4f00-de08-4e15520cd010"
# preview the dataset again

df.head()

# %% [markdown] id="YgAhcAdW5CKs"
# Now, we can see that the `Date` variable has been removed from the dataset.
#

# %% [markdown] id="G2FD01il5CKt"
# ### Explore Categorical Variables
#
#
# Now, I will explore the categorical variables one by one.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1724749448598, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="gnOtt8eJ5CKt" outputId="cb876c2d-260e-4f58-d4de-15b13d124a56"
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)

# %% [markdown] id="NnodQJcI5CKt"
# We can see that there are 6 categorical variables in the dataset. The `Date` variable has been removed. First, I will check missing values in categorical variables.

# %% colab={"base_uri": "https://localhost:8080/", "height": 272} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1724749448598, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="14hPxDyv5CKt" outputId="5c02bbcb-3889-4c5f-f624-655d92626184"
# check for missing values in categorical variables

df[categorical].isnull().sum()

# %% [markdown] id="VD-d4RwW5CKt"
# We can see that `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` variables contain missing values. I will explore these variables one by one.

# %% [markdown] id="GVqP4Wlo5CKt"
# ### Explore `Location` variable

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1724749448598, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="grIrEWCa5CKt" outputId="1fb8c691-ef71-4098-970d-09ba1282e56a"
# print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1724749448598, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="lPSchOxg5CKu" outputId="7542ffe4-01c0-4e94-d354-98299d13901a"
# check labels in location variable

df.Location.unique()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1724749448598, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="GJed3JtV5CKu" outputId="ed9d9c74-ba1c-4f06-80bd-cdcdf52a6d05"
# check frequency distribution of values in Location variable

df.Location.value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 883, "status": "ok", "timestamp": 1724749449461, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="vqTdyBpK5CKu" outputId="f82cfd44-f1c5-42fa-80b9-db37f2939e3c"
# let's do One Hot Encoding of Location variable
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first=True).head()

# %% [markdown] id="UmNXx6ai5CKu"
# ### Explore `WindGustDir` variable

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 34, "status": "ok", "timestamp": 1724749449461, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="H0NYwcjQ5CKu" outputId="b7713457-0a84-4dfd-e235-b8e844e7a1fb"
# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1724749449461, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Zo0XAY9Z5CKu" outputId="a0c2f662-7416-4b08-db95-e0a608910cb2"
# check labels in WindGustDir variable

df['WindGustDir'].unique()

# %% colab={"base_uri": "https://localhost:8080/", "height": 617} executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1724749449462, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Qxa6bci_5CKu" outputId="535d03b0-d9a2-4b03-bfeb-a5358ebb2e9d"
# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1724749449462, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ZgfOlw2x5CKv" outputId="273375b8-87cd-472c-ae6d-4aa57bd67b40"
# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()

# %% [markdown] id="Jo8Y95eo64w6"
# Avoiding Multicollinearity: Dropping the first category prevents perfect multicollinearity, making the model more stable and interpretable.
# Reference Category: The dropped category serves as the reference level against which other categories are compared.

# %% colab={"base_uri": "https://localhost:8080/", "height": 585} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1724749449462, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="_3GtELSz5CKv" outputId="7e129b00-2d96-4521-8955-1884217b16e1"
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)

# %% [markdown] id="J41k2rbR5CKv"
# We can see that there are 10326 missing values in WindGustDir variable.

# %% [markdown] id="2MLAQmd35CKv"
# ### Explore `WindDir9am` variable

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1724749449462, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="d3WGIQq95CKv" outputId="bce1f1f9-5725-43e6-c511-4f6a859fa9de"
# print number of labels in WindDir9am variable

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1724749449462, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="84Z45dUU5CKv" outputId="fc0172fb-a7ce-4a8c-fcfb-6ef8ca4bebd2"
# check labels in WindDir9am variable

df['WindDir9am'].unique()

# %% colab={"base_uri": "https://localhost:8080/", "height": 617} executionInfo={"elapsed": 725, "status": "ok", "timestamp": 1724749450162, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="L7ObSvpD5CKw" outputId="4938642c-3898-4935-dbd7-a927cc8399fa"
# check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 42, "status": "ok", "timestamp": 1724749450162, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="aTAOd-gE5CKw" outputId="ff71eed8-2004-4864-8363-e0189357563f"
# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 585} executionInfo={"elapsed": 41, "status": "ok", "timestamp": 1724749450162, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4NgY7e__5CKw" outputId="40d2a746-da8c-4d40-d912-191db9255ba6"
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)

# %% [markdown] id="xco9kXO55CKw"
# We can see that there are 10566 missing values in the `WindDir9am` variable.

# %% [markdown] id="BXNj9LXD5CKw"
# ### Explore `WindDir3pm` variable

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 41, "status": "ok", "timestamp": 1724749450162, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="xjcpSpcB5CKw" outputId="8e56243a-25bb-44e0-d9c8-bd5fd76fd486"
# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 38, "status": "ok", "timestamp": 1724749450162, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="s2Sj0Vxz5CKw" outputId="5f41fb19-d2b2-4876-9f51-9187dea89c77"
# check labels in WindDir3pm variable

df['WindDir3pm'].unique()

# %% colab={"base_uri": "https://localhost:8080/", "height": 617} executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1724749450162, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="knbrM2FG5CKw" outputId="59ffed15-5a46-43d9-924e-9b56a187d19f"
# check frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1724749450163, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="2x80_k7t5CKx" outputId="276cc837-f7be-4a1f-b66b-949cd1b25c55"
# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 585} executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1724749450163, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="_PGe_qtD5CKx" outputId="ebfab071-a08f-447b-cdf5-511264cc42b4"
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)

# %% [markdown] id="lqAqOcHR5CKx"
# There are 4228 missing values in the `WindDir3pm` variable.

# %% [markdown] id="J_QrA7tA5CKx"
# ### Explore `RainToday` variable

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 35, "status": "ok", "timestamp": 1724749450163, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="c41IoNKf5CKx" outputId="672ee82d-0f48-42bd-8f32-9dce5fac4788"
# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 34, "status": "ok", "timestamp": 1724749450163, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="FQpng2FC5CKx" outputId="98fdc78d-84c9-42f4-95a3-c4ce66108956"
# check labels in WindGustDir variable

df['RainToday'].unique()

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1724749450163, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="PU0K_c5M5CKx" outputId="8b3ad668-f3f7-4709-9d5e-bfd2daaed157"
# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1724749450163, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="XW8Hvh_x5CKx" outputId="b49e7cd5-3dff-4fb3-9d0c-00415e48357a"
# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 146} executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1724749450163, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="cZo2UmaO5CKy" outputId="65505549-3ead-4fd1-91cf-d9243c4f59dc"
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)

# %% [markdown] id="QqWjCprz5CKy"
# There are 3261 missing values in the `RainToday` variable.

# %% [markdown] id="WnxiNdqX5CKy"
# ### Explore Numerical Variables

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1724749450163, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="GCovHC265CKy" outputId="edee7605-f8ea-4126-a928-36c5de938d0d"
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype !='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

# %% colab={"base_uri": "https://localhost:8080/", "height": 226} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1724749450496, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="vy6OFtZL5CKy" outputId="8d05c833-ccbe-4c61-f76a-4494fbf4c589"
# view the numerical variables

df[numerical].head()

# %% [markdown] id="_6zvHlrF5CKz"
# ### Summary of numerical variables
#
#
# - There are 16 numerical variables.
#
#
# - These are given by `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am` and `Temp3pm`.
#
#
# - All of the numerical variables are of continuous type.

# %% [markdown] id="QNILsYFL5CKz"
# ## Explore problems within numerical variables
#
#
# Now, I will explore the numerical variables.
#
#
# ### Missing values in numerical variables

# %% colab={"base_uri": "https://localhost:8080/", "height": 679} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1724749450496, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="f2Ea9Luw5CKz" outputId="9e256282-f218-4e92-a7ff-30a1d72d219b"
# check missing values in numerical variables

df[numerical].isnull().sum()

# %% [markdown] id="s2LLC00H5CKz"
# We can see that all the 16 numerical variables contain missing values.

# %% [markdown] id="Osapi1815CKz"
# ### Outliers in numerical variables

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 291, "status": "ok", "timestamp": 1724749450763, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="-uqTsld35CKz" outputId="921a3ac0-15b0-4ab7-cdba-a08ae553b261"
# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)

# %% [markdown] id="S1xxTeIo5CKz"
# On closer inspection, we can see that the `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns may contain outliers.
#
#
# I will draw boxplots to visualise outliers in the above variables.

# %% colab={"base_uri": "https://localhost:8080/", "height": 690} executionInfo={"elapsed": 3069, "status": "ok", "timestamp": 1724749453824, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="gAQCg_p45CK0" outputId="2eb11c8d-2902-455a-aece-85c82c7e807b"
# draw boxplots to visualize outliers

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

# %% [markdown] id="cNjibgK25CK0"
# The above boxplots confirm that there are lot of outliers in these variables.

# %% [markdown] id="A52fk41t5CK0"
# ### Check the distribution of variables
#
#
# Now, I will plot the histograms to check distributions to find out if they are normal or skewed. If the variable follows normal distribution, then I will do `Extreme Value Analysis` otherwise if they are skewed, I will find IQR (Interquantile range).

# %% colab={"base_uri": "https://localhost:8080/", "height": 692} executionInfo={"elapsed": 1790, "status": "ok", "timestamp": 1724749455603, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="eRglmbvj5CK0" outputId="9d128344-3812-4e27-cf27-2ad783392363"
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

# %% [markdown] id="bdPj4hSz5CK0"
# We can see that all the four variables are skewed. So, I will use interquantile range to find outliers.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1724749455603, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="YBqk-TX95CK0" outputId="8ddcc11c-42f5-484a-abaf-f163aaf27abd"
# find outliers for Rainfall variable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# %% [markdown] id="EMFu_M4o5CK1"
# For `Rainfall`, the minimum and maximum values are 0.0 and 371.0. So, the outliers are values > 3.2.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1724749455603, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Itp-dmpC5CK1" outputId="084de4d7-b0f3-4ea3-828a-e1d7f92e5b8e"
# find outliers for Evaporation variable

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# %% [markdown] id="nLq23Mxq5CK1"
# For `Evaporation`, the minimum and maximum values are 0.0 and 145.0. So, the outliers are values > 21.8.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1724749455603, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="TSXesys75CK2" outputId="71f51c59-c7ca-44c4-8c0e-aa3eca497d01"
# find outliers for WindSpeed9am variable

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# %% [markdown] id="dAm8Rr6O5CK2"
# For `WindSpeed9am`, the minimum and maximum values are 0.0 and 130.0. So, the outliers are values > 55.0.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1724749455603, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ihIWEhxK5CK2" outputId="8f77e67d-ec68-4d0f-8869-f42fba752b77"
# find outliers for WindSpeed3pm variable

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# %% [markdown] id="uyK_egy25CK2"
# For `WindSpeed3pm`, the minimum and maximum values are 0.0 and 87.0. So, the outliers are values > 57.0.

# %% [markdown] id="mJo-h6Tm5CK2"
# # **8. Declare feature vector and target variable** <a class="anchor" id="8"></a>
#
#
# [Table of Contents](#0.1)

# %% id="gqaZbPge5CK3"
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']

# %% id="drNF_iIcuGuO"
# impute the y with the most frerquent and then convert to 0 for No and 1 for yes

# impute with most frequent
y = y.fillna(y.mode()[0])

# convert to numerical
y = y.replace({'No': 0, 'Yes': 1})


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 46, "status": "ok", "timestamp": 1724749456150, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Wa85bDGkuaj9" outputId="1791f07c-c85c-4a50-c8b7-ec3faca35969"
y.isnull().sum()
print(y[:20])

# %% [markdown] id="wDjllQVq5CK3"
# # **9. Split data into separate training and test set** <a class="anchor" id="9"></a>
#
#
# [Table of Contents](#0.1)

# %% id="b0XLchxm5CK3"
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# %% colab={"base_uri": "https://localhost:8080/", "height": 178} executionInfo={"elapsed": 44, "status": "ok", "timestamp": 1724749456150, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="pNGZJgYx87Pd" outputId="71abd8b4-44e0-4a37-a595-2cf404f614c6"
y_train.value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 43, "status": "ok", "timestamp": 1724749456150, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="yCsSFR_D5CK3" outputId="e4adc3b7-72bf-4613-8fcb-f87cabadace2"
# check the shape of X_train and X_test

X_train.shape, X_test.shape

# %% [markdown] id="HMMJAC215CK3"
# # **10. Feature Engineering** <a class="anchor" id="10"></a>
#
#
# [Table of Contents](#0.1)
#
#
# **Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. I will carry out feature engineering on different types of variables.
#
#
# First, I will display the categorical and numerical variables again separately.

# %% colab={"base_uri": "https://localhost:8080/", "height": 836} executionInfo={"elapsed": 40, "status": "ok", "timestamp": 1724749456150, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="y9H2k_5v5CK4" outputId="4564989a-685d-4c43-dcfd-06076c7704d9"
# check data types in X_train

X_train.dtypes

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 40, "status": "ok", "timestamp": 1724749456151, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="pAj0kW6n5CK4" outputId="8f2a9c30-bd5e-4aaa-ddd7-1878e4ab79b1"
# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 38, "status": "ok", "timestamp": 1724749456151, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ymAwScIG5CK4" outputId="8cd8f039-54fb-4d23-f141-8dab88cb5f36"
# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical

# %% [markdown] id="w7GQ9hS75CK5"
# ### Engineering missing values in numerical variables
#
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 679} executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1724749456151, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4bxSROGT5CK5" outputId="b5807c23-42ca-4335-824a-e85e1aa4bae4"
# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 679} executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1724749456151, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="mEVcT1yt5CK5" outputId="4a097716-62eb-4347-bcf4-4305ea9ee1ed"
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 35, "status": "ok", "timestamp": 1724749456151, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="7rxdTmsw5CK5" outputId="359c6e79-c55e-4e40-81f4-b89166761e6b"
# print percentage of missing values in the numerical variables in training set

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))

# %% [markdown] id="4POXMJdq5CK5"
# ### Assumption
#
#
# I assume that the data are missing completely at random (MCAR). There are two methods which can be used to impute missing values. One is mean or median imputation and other one is random sample imputation. When there are outliers in the dataset, we should use median imputation. So, I will use median imputation because median imputation is robust to outliers.
#
#
# I will impute missing values with the appropriate statistical measures of the data, in this case median. Imputation should be done over the training set, and then propagated to the test set. It means that the statistical measures to be used to fill missing values both in train and test set, should be extracted from the train set only. This is to avoid overfitting.

# %% id="awrKo7R55CK6"
'# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)


# %% colab={"base_uri": "https://localhost:8080/", "height": 679} executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1724749456151, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="bHRYgOOh5CK6" outputId="01779cc4-2aef-4f67-8488-de534fb335f6"
# check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 679} executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1724749456151, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="HSWK4I6F5CK6" outputId="9a5d1d52-5bcf-4591-ec02-b60f564d056f"
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()

# %% [markdown] id="c-P4nO275CK6"
# Now, we can see that there are no missing values in the numerical columns of training and test set.

# %% [markdown] id="vc7oMk5Q5CK7"
# ### Engineering missing values in categorical variables

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1724749456151, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="HF9SwXmc5CK7" outputId="40ec9893-1388-4a31-f41c-eb1d76c70396"
# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 357, "status": "ok", "timestamp": 1724749456476, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="efvmINDd5CK7" outputId="218f8fe4-ebab-4530-8ded-f17af49552ff"
# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))

# %% id="XbBXUrtj5CK7"
# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"elapsed": 35, "status": "ok", "timestamp": 1724749456476, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="tHBwx7dN5CK7" outputId="184c5450-12e6-421e-ab67-90fed0f766f1"
# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 836} executionInfo={"elapsed": 34, "status": "ok", "timestamp": 1724749456476, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="toQ281Y3rktX" outputId="b36e7fd1-18c0-49ac-d377-f3651a5307a5"
X_train.isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"elapsed": 34, "status": "ok", "timestamp": 1724749456476, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="CutEe_hl5CK8" outputId="4a3890bb-dd2e-4afe-d34a-2c282453746d"
# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()

# %% [markdown] id="DVwFLO705CK8"
# As a final check, I will check for missing values in X_train and X_test.

# %% colab={"base_uri": "https://localhost:8080/", "height": 836} executionInfo={"elapsed": 34, "status": "ok", "timestamp": 1724749456477, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="CwMs4fPk5CK8" outputId="41a77edd-2289-46e6-f6be-1ee970aa14c8"
# check missing values in X_train

X_train.isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 836} executionInfo={"elapsed": 446, "status": "ok", "timestamp": 1724749456889, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="lf7vP_3I5CK8" outputId="6fabd82a-336c-4ff3-8f9e-d7d40c326e48"
# check missing values in X_test

X_test.isnull().sum()


# %% [markdown] id="-v0Vdwrw5CK8"
# We can see that there are no missing values in X_train and X_test.

# %% [markdown] id="Lu2IjBGj5CK9"
# ### Engineering outliers in numerical variables
#
#
# We have seen that the `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns contain outliers. I will use top-coding approach to cap maximum values and remove outliers from the above variables.

# %% id="mYVhU3eK5CK9"
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])  # df3["rainfalll"]= 3

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 42, "status": "ok", "timestamp": 1724749456890, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="20KeSGgw5CK9" outputId="407a0da0-08d3-468f-affa-225f34864a9d"
X_train.Rainfall.max(), X_test.Rainfall.max()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 40, "status": "ok", "timestamp": 1724749456890, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="PVRUefpz5CK9" outputId="205d107f-1032-4f93-cf12-6affea248b5d"
X_train.Evaporation.max(), X_test.Evaporation.max()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 38, "status": "ok", "timestamp": 1724749456890, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="BHtua_965CLB" outputId="81f59922-58b5-4745-f286-b2d2c61c65a6"
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 35, "status": "ok", "timestamp": 1724749456890, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4oj6osWn5CLC" outputId="c02e9ecf-d12d-41f0-f35f-685d062a5fba"
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} executionInfo={"elapsed": 34, "status": "ok", "timestamp": 1724749456890, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="pPsUGQIF5CLC" outputId="e388d7fb-6fb4-44d8-c2da-eee43a0475cc"
X_train[numerical].describe()

# %% [markdown] id="8tnaxMK85CLC"
# We can now see that the outliers in `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns are capped.

# %% [markdown] id="AEqjd2Kx5CLC"
# ### Encode categorical variables

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1724749456890, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4ZR1UNYG5CLC" outputId="d617e21b-ddbe-468c-89eb-f2118265e7d6"
categorical

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1724749456890, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="0FvjgYsD5CLD" outputId="0de0979f-5296-4e12-c8a9-ac0e49e0dda9"
X_train[categorical].head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3288, "status": "ok", "timestamp": 1724749460149, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="q1m1DUXGU5vt" outputId="f624e237-a95f-48a9-b54e-8b9e83e2efff"
# !pip install category_encoders

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1724749460149, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="eiKgVRUucbZI" outputId="2aebe2d5-bf88-486e-f4d9-5f5d81804e64"
X_train[categorical].head()

# %% id="VsHSsxTF5CLD"
# encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 473, "status": "ok", "timestamp": 1724749460597, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="3YwYuDD15CLD" outputId="6ae3871b-f663-4b14-e89d-36c4973bd028"
X_train.head()

# %% [markdown] id="KpRP3stH5CLD"
# We can see that two additional variables `RainToday_0` and `RainToday_1` are created from `RainToday` variable.
#
# Now, I will create the `X_train` training set.

# %% id="VaqlWHHI5CLD"
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location),
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1724749460597, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="C1l8KSuF5CLE" outputId="8409a2dc-53df-4ebd-98ab-d6ee8e7d0d48"
X_train.head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1724749460597, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="UByrBcPZVn4f" outputId="87596538-dc94-47dc-9f3a-bad130d75224"
X_train.shape


# %% [markdown] id="e2DPbfFJ5CLE"
# Similarly, I will create the `X_test` testing set.

# %% id="sgRs3aBe5CLE"
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location),
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 28, "status": "ok", "timestamp": 1724749460597, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="7XPI0roP5CLE" outputId="89425b95-ec90-4423-ecce-2ed0fb1a0f91"
X_test.head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1724749460597, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="RiLejZ2QVyZB" outputId="ceafcafc-c540-4c01-92f3-854ef0b383ec"
X_test.shape

# %% [markdown] id="umS8ubIS5CLE"
# We now have training and testing set ready for model building. Before that, we should map all the feature variables onto the same scale. It is called `feature scaling`. I will do it as follows.

# %% [markdown] id="VpM3Trmj5CLE"
# # **11. Feature Scaling** <a class="anchor" id="11"></a>
#
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 349} executionInfo={"elapsed": 420, "status": "ok", "timestamp": 1724749460992, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="_C1ijSw55CLF" outputId="221cff67-4c30-42e0-d014-0d83b749ba51"
X_train.describe()

# %% id="twfDfYAC5CLF"
cols = X_train.columns

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1724749460992, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="0tn92yAzfpq7" outputId="ac52dc34-9316-42b8-c39f-b754556f1a5d"
X_train.head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 50068, "status": "ok", "timestamp": 1724749511487, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="osHtmxosEUwj" outputId="e8e16ddc-8c68-4af2-f4b2-4fbaffe4d5cc"
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter
print(Counter(y_train))

smote = SMOTE(random_state=42)
X_res_smote, y_res_smote = smote.fit_resample(X_train, y_train)
print(Counter(y_res_smote))
model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_res_smote, y_res_smote)
y_pred_smote = model_smote.predict(X_test)
print("SMOTE:")
print(classification_report(y_test, y_pred_smote, digits=4))

#smote_enn = SMOTEENN(random_state=42)
#X_train, y_train  = smote_enn.fit_resample(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="3kmziFEvFdLg" outputId="03e9f1cc-d426-4092-d87b-118698be6935"
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# Display resampled class distribution
print(f"Resampled class distribution: {Counter(y_resampled)}")

# Training a logistic regression model on the resampled dataset
model = LogisticRegression(random_state=42)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# Evaluating the model
print("SMOTEENN Results:")
print(classification_report(y_test, y_pred))

# %% id="59tev217F_1M"
ros = RandomOverSampler(random_state=42)
X_res_ros, y_res_ros = ros.fit_resample(X_train, y_train)
print(Counter(y_res_ros))
model_ros = LogisticRegression(random_state=42)
model_ros.fit(X_res_ros, y_res_ros)
y_pred_ros = model_ros.predict(X_test)
print("Random Oversampling:")
print(classification_report(y_test, y_pred_ros))

# %% id="pOAdz2rR5CLF"
from sklearn.preprocessing import MinMaxScaler  # 0 - 1

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# %% id="bjkoLge45CLF"
X_train = pd.DataFrame(X_train, columns=[cols])

# %% id="9EA5zeTt5CLG"
X_test = pd.DataFrame(X_test, columns=[cols])

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1724743637603, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="CSxw4dx1gQQd" outputId="6ec2871a-ea2a-47de-fa8b-3f6cec2e219c"
X_train.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 349} executionInfo={"elapsed": 1735, "status": "ok", "timestamp": 1724743639318, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="9suXDeiF5CLG" outputId="90c1a0d5-f4fe-47be-a641-c486a45ef710"
X_train.describe()

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1724743639318, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="UteloIQiXkS3" outputId="c1834b98-99b8-49c2-da44-5205af1459ca"
X_train[:5]

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1724743639318, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="cdpmbJ4eXnbU" outputId="cbddce9b-f1e3-4cbc-e49e-9cbf2161f6e8"
y_train[:5]

# %% [markdown] id="6dh7jtdI5CLG"
# We now have `X_train` dataset ready to be fed into the Logistic Regression classifier. I will do it as follows.

# %% [markdown] id="M4X3F5xE5CLG"
# # **12. Model training** <a class="anchor" id="12"></a>
#
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1724743639318, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="5tuKJZNmYLHY" outputId="549f025d-ae47-44c7-b0a7-430235e74afd"
X_train.shape

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1724743639319, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="bK4XCbnesTgR" outputId="a1375407-fc37-4b97-db37-a3f52198ea5c"
X_train.shape

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1724743639319, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="TLFYXxXLsXZK" outputId="8ebb01e8-e3e1-4258-fb06-5e1e2021335d"
y_train.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 458} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1724743639319, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="qR2nzPnlsdg3" outputId="5f88bdba-97d3-411e-bf2f-5f0d25b0304a"
X_test.isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1724743639319, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="EN-lPsHLsjwX" outputId="ae411d04-81b5-43e8-bb7b-99eb6d67a2c5"
y_test.isnull().sum()

# %% [markdown] id="dGQmSENj5CLH"
# # **13. Predict results** <a class="anchor" id="13"></a>
#
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 74} executionInfo={"elapsed": 7790, "status": "ok", "timestamp": 1724743647087, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="YN3OG07y5CLG" outputId="b344c6cd-ef8a-432b-99d3-5baa27101cf2"
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression   #only for classification


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1724743647860, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="B5no9jDc5CLH" outputId="3942175d-d14d-4f65-cab9-be8118de9bf2"
y_pred_test = logreg.predict(X_test)

y_pred_test

# %% [markdown] id="VkaXfeWf5CLH"
# ### predict_proba method
#
#
# **predict_proba** method gives the probabilities for the target variable(0 and 1) in this case, in array form.
#
# `0 is for probability of no rain` and `1 is for probability of rain.`

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 373, "status": "ok", "timestamp": 1724743656722, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="GtqjUis95CLI" outputId="b69d68c6-6478-4235-f026-020834b589d3"
# probability of getting output as 0 - no rain   #No =>0   , Yes=>1

logreg.predict_proba(X_test)[:,0]

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 320, "status": "ok", "timestamp": 1724743662104, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="U73nEJOc5CLI" outputId="b353ee7b-1e47-4d7c-cfa8-c047d9e93911"
# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1]

# %% [markdown] id="NWSkIKIf5CLI"
# # **14. Check accuracy score** <a class="anchor" id="14"></a>
#
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 286, "status": "ok", "timestamp": 1724743667830, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="M4rgQzKd5CLI" outputId="7ba26092-07ce-4ebd-fbe5-2dbda44004d8"
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

# %% [markdown] id="-5wsn3wR5CLI"
# Here, **y_test** are the true class labels and **y_pred_test** are the predicted class labels in the test-set.

# %% [markdown] id="BKjkKGyW5CLI"
# ### Compare the train-set and test-set accuracy
#
#
# Now, I will compare the train-set and test-set accuracy to check for overfitting.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 309, "status": "ok", "timestamp": 1724743811436, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Y6-kKnoh5CLJ" outputId="1aaa1f9d-dfd6-48a8-d59f-3291a35e5a21"
y_pred_train = logreg.predict(X_train)

y_pred_train

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 331, "status": "ok", "timestamp": 1724743819186, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="HqfYA0qi5CLJ" outputId="2f368278-48fc-4057-877e-b8425526cb30"
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# %% [markdown] id="fe4vJ7nb5CLJ"
# ### Check for overfitting and underfitting

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 319, "status": "ok", "timestamp": 1724743843194, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="eFsWDVUM5CLK" outputId="3d925362-68be-4def-d04a-a02cad7fe170"
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))

# %% [markdown] id="bYfBZuZL5CLK"
# The training-set accuracy score is 77.01 while the test-set accuracy to be 77.95. These two values are quite comparable. So, there is no question of overfitting.
#

# %% [markdown] id="3b94KwOD5CLK"
# In Logistic Regression, we use default value of C = 1. It provides good performance with approximately 77% accuracy on both the training and the test set. But the model performance on both the training and test set are very comparable. It is likely the case of underfitting.
#
# I will increase C and fit a more flexible model.

# %% colab={"base_uri": "https://localhost:8080/", "height": 74} executionInfo={"elapsed": 3826, "status": "ok", "timestamp": 1724743937419, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="pPbP3h0N5CLL" outputId="683fa131-4cae-41a3-cbb3-918c8c32fd0c"
# fit the Logsitic Regression model with C=100

# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 302, "status": "ok", "timestamp": 1724743940974, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="uD52V2we5CLL" outputId="4472075e-a1b9-46d7-b655-6a3862e78f66"
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))

# %% [markdown] id="OzIEw-vv5CLL"
# We can see that, C=100 results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.

# %% [markdown] id="VoGv_yF-5CLL"
# Now, I will investigate, what happens if we use more regularized model than the default value of C=1, by setting C=0.01.

# %% colab={"base_uri": "https://localhost:8080/", "height": 74} executionInfo={"elapsed": 2401, "status": "ok", "timestamp": 1724743982996, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="0fQvsyT55CLL" outputId="9a567345-9735-4647-94a4-e0c7dd517a95"
# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 326, "status": "ok", "timestamp": 1724743985690, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="pcVwOQME5CLL" outputId="219bdc73-3c61-4321-fce7-5da43a759677"
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))

# %% [markdown] id="h3xQWZzw5CLM"
# So, if we use more regularized model by setting C=0.01, then both the training and test set accuracy decrease relatiev to the default parameters.

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} executionInfo={"elapsed": 339, "status": "ok", "timestamp": 1724744025018, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="NPMpVyZ25CLM" outputId="04949caa-48c2-42ae-a1d3-460a6a9746bb"
# check class distribution in test set

y_test.value_counts()

# %% [markdown] id="Q7oJBYL25CLN"
# # **15. Confusion matrix** <a class="anchor" id="15"></a>
#
#
# [Table of Contents](#0.1)
#
#
# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.
#
#
# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-
#
#
# **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
#
#
# **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
#
#
# **False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**
#
#
#
# **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**
#
#
#
# These four outcomes are summarized in a confusion matrix given below.
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 296, "status": "ok", "timestamp": 1724744035524, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="_N7oCz9A5CLO" outputId="8896ff1f-3e7e-43f2-9524-55352b5bbb50"
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# %% [markdown] id="LfbgPdwH5CLO"
# The confusion matrix shows `20892 + 3285 = 24177 correct predictions` and `3087 + 1175 = 4262 incorrect predictions`.
#
#
# In this case, we have
#
#
# - `True Positives` (Actual Positive:1 and Predict Positive:1) - 20892
#
#
# - `True Negatives` (Actual Negative:0 and Predict Negative:0) - 3285
#
#
# - `False Positives` (Actual Negative:0 but Predict Positive:1) - 1175 `(Type I error)`
#
#
# - `False Negatives` (Actual Positive:1 but Predict Negative:0) - 3087 `(Type II error)`

# %% colab={"base_uri": "https://localhost:8080/", "height": 447} executionInfo={"elapsed": 1069, "status": "ok", "timestamp": 1724744051283, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="73kjSTgC5CLO" outputId="e407cfdb-7c2b-4b24-dce9-d7e1b57dace6"
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

# %% [markdown] id="2w1HBc595CLO"
# # **16. Classification metrices** <a class="anchor" id="16"></a>
#
#
# [Table of Contents](#0.1)

# %% [markdown] id="NvFsFxZ95CLO"
# ## Classification Report
#
#
# **Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later.
#
# We can print a classification report as follows:-

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 305, "status": "ok", "timestamp": 1724744092275, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="-35zDNsT5CLO" outputId="4b37c928-683c-4bb6-9e01-9d8fd8cf2bfa"
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test, digits=4))

# %% [markdown] id="wr5pIUxL5CLP"
# ## Classification accuracy

# %% id="e6S87saX5CLP"
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 309, "status": "ok", "timestamp": 1724744109684, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="3ceU7KA45CLP" outputId="410c532e-f3d9-491e-b8e5-360265f7b815"
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 329, "status": "ok", "timestamp": 1724744112050, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="hKPXLIR2nZRB" outputId="995e697f-6e11-407e-af13-7e788b311134"
print(1-((TP + TN) / float(TP + TN + FP + FN)))

# %% [markdown] id="MV673zKh5CLP"
# ## Classification error

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 996, "status": "ok", "timestamp": 1724744119992, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="5DhRrjP55CLP" outputId="f13503c3-8042-4dcb-8d6e-1e315a146326"
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# %% [markdown] id="Ei15920_5CLP"
# ## Precision
#
#
# **Precision** can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).
#
#
# So, **Precision** identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.
#
#
#
# Mathematically, precision can be defined as the ratio of `TP to (TP + FP).`
#
#
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 354, "status": "ok", "timestamp": 1724744124414, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="c-wu5Tbn5CLP" outputId="d1713181-e1ea-479c-94e5-575b2558121c"
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# %% [markdown] id="q6h6Now95CLQ"
# ## Recall
#
#
# Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes.
# It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). **Recall** is also called **Sensitivity**.
#
#
# **Recall** identifies the proportion of correctly predicted actual positives.
#
#
# Mathematically, recall can be given as the ratio of `TP to (TP + FN).`
#
#
#
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 294, "status": "ok", "timestamp": 1724744126743, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="JFzgeak35CLQ" outputId="de502aca-318d-4f5f-cad9-7d69202435fd"
recall = TP / float(TP + FN)    #recall  &  Sensitifity  & True positve rate

print('Recall or Sensitivity : {0:0.4f}'.format(recall))

# %% [markdown] id="KcXLefV35CLQ"
# ## True Positive Rate
#
#
# **True Positive Rate** is synonymous with **Recall**.
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 312, "status": "ok", "timestamp": 1724744130087, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ONw6CRxI5CLQ" outputId="9723f2a3-9834-498b-9afc-7661cffa28da"
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

# %% [markdown] id="6idjBGwz5CLQ"
# ## False Positive Rate

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 336, "status": "ok", "timestamp": 1724744133157, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="HV1NjT4m5CLR" outputId="1e10da42-5865-4777-c968-59d9a55ad4e7"
false_positive_rate = FP / float(FP + TN) # Specificity


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

# %% [markdown] id="fu5OXTNq5CLR"
# ## Specificity

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1724744135377, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="kffySjLh5CLR" outputId="943fc120-a579-42e7-c9bb-c66567aa85a4"
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))

# %% [markdown] id="1PDq3v735CLR"
# ## f1-score
#
#
# **f1-score** is the weighted harmonic mean of precision and recall. The best possible **f1-score** would be 1.0 and the worst
# would be 0.0.  **f1-score** is the harmonic mean of precision and recall. So, **f1-score** is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of `f1-score` should be used to
# compare classifier models, not global accuracy.
#
#

# %% [markdown] id="OhJnN_QF5CLR"
# ## Support
#
#
# **Support** is the actual number of occurrences of the class in our dataset.

# %% [markdown] id="_8Jpq3Cd5CLR"
# # **17. Adjusting the threshold level** <a class="anchor" id="17"></a>
#
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 312, "status": "ok", "timestamp": 1724744140806, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Fv7uTvIz5CLR" outputId="627bfb99-1d62-460f-a4c1-15b2c0b832a6"
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob

# %% [markdown] id="-pQMU9hC5CLS"
# # **18. ROC - AUC** <a class="anchor" id="18"></a>
#
#
# [Table of Contents](#0.1)
#
#
#
# ## ROC Curve
#
#
# Another tool to measure the classification model performance visually is **ROC Curve**. ROC Curve stands for **Receiver Operating Characteristic Curve**. An **ROC Curve** is a plot which shows the performance of a classification model at various
# classification threshold levels.
#
#
#
# The **ROC Curve** plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold levels.
#
#
#
# **True Positive Rate (TPR)** is also called **Recall**. It is defined as the ratio of `TP to (TP + FN).`
#
#
#
# **False Positive Rate (FPR)** is defined as the ratio of `FP to (FP + TN).`
#
#
#
#
# In the ROC Curve, we will focus on the TPR (True Positive Rate) and FPR (False Positive Rate) of a single point. This will give us the general performance of the ROC curve which consists of the TPR and FPR at various threshold levels. So, an ROC Curve plots TPR vs FPR at different classification threshold levels. If we lower the threshold levels, it may result in more items being classified as positve. It will increase both True Positives (TP) and False Positives (FP).
#
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"elapsed": 1365, "status": "ok", "timestamp": 1724744185416, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ZYF2CJuP5CLS" outputId="cf63a12d-b745-42ed-b5a5-3d7e97b9a328"
# plot ROC Curve

y_pred = logreg.predict(X_test)

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 343, "status": "ok", "timestamp": 1724744301123, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ai1uudXPxR_2" outputId="2ed11246-94a9-4bc3-dfcd-329c01fe3eee"
from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# %% [markdown] id="aOWNTp_n5CLT"
# ROC curve help us to choose a threshold level that balances sensitivity and specificity for a particular context.

# %% [markdown] id="kdy_7Dsg5CLT"
# ## ROC-AUC
#
#
# **ROC AUC** stands for **Receiver Operating Characteristic - Area Under Curve**. It is a technique to compare classifier performance. In this technique, we measure the `area under the curve (AUC)`. A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.
#
#
# So, **ROC AUC** is the percentage of the ROC plot that is underneath the curve.

# %% [markdown] id="XdEF-E-s5CLT"
# ### Comments
#
#
# - ROC AUC is a single number summary of classifier performance. The higher the value, the better the classifier.
#
# - ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18098, "status": "ok", "timestamp": 1724744797481, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="J82yv8mW5CLT" outputId="a16e679b-4416-4fbf-a7d9-16bb9b844538"
# calculate cross-validated Accuracy

from sklearn.model_selection import cross_val_score

Cross_validated_accuracy = cross_val_score(logreg, X_train, y_train, cv=5, scoring='accuracy').mean()

print('Cross validated accuracy : {:.4f}'.format(Cross_validated_accuracy))

# %% [markdown] id="WUtoVSH_5CLT"
# # **19. k-Fold Cross Validation** <a class="anchor" id="19"></a>
#
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12216, "status": "ok", "timestamp": 1724744831269, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="g7eO7CBT5CLT" outputId="a5cde597-b1fa-4edd-e5a6-d115d682e5ef"
# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))

# %% [markdown] id="95eCc1tJ5CLT"
# We can summarize the cross-validation accuracy by calculating its mean.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 308, "status": "ok", "timestamp": 1724744881298, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="SM169jD35CLT" outputId="7081d75c-65cd-4da2-cf5b-ed07f02527d5"
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))

# %% [markdown] id="PuNI8rvv5CLU"
# Our, original model score is found to be 0.8476. The average cross-validation score is 0.84.81. So, we can conclude that cross-validation does not result in performance improvement.

# %% [markdown] id="XalonQdA5CLU"
# # **20. Hyperparameter Optimization using GridSearch CV** <a class="anchor" id="20"></a>
#
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 315, "status": "ok", "timestamp": 1724746825265, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="P_w9mrE162Bi" outputId="895f3dcd-d71b-437e-877c-a4e02db907ee"
np.logspace(1,100,3)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1118497, "status": "ok", "timestamp": 1724748012397, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Xp7N4dKc5YNR" outputId="3b4fa849-ffd3-4298-9252-a8ef91bc8be8"
# logistic regreesssion with grid seach using all parameters

import numpy as np
from sklearn.model_selection import GridSearchCV

parameters = {'penalty': ['l1', 'l2'],
              'C': [1,10,50],     #[ 7 numbers ]
              'solver': ['newton-cg', 'lbfgs', 'liblinear']
              }

logreg = LogisticRegression()
clf = GridSearchCV(logreg,                    # model
                   param_grid = parameters,   # hyperparameters
                   scoring='accuracy',        # metric for scoring
                   cv=5, n_jobs=-1, verbose=1)       # number of folds

clf.fit(X_train,y_train)
print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :",clf.best_score_)


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1724748013240, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="3Qul04WN5CLU" outputId="7bc69a30-60dc-4ba9-bb78-59cc6d3e6050"
# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(clf.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (clf.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (clf.best_estimator_))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1724748013240, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="G8jqOYJt5CLU" outputId="1e2d000c-e26e-45c9-c13f-dfdfaa1748a2"
# Calculate GridSearch CV score on test set
best_LR=clf.best_estimator_
y_pred_test = best_LR.predict(X_test)
print('Accuracy score on test set: {0:0.4f}'.format(best_LR.score(X_test, y_test)))

# %% colab={"base_uri": "https://localhost:8080/", "height": 660} executionInfo={"elapsed": 866, "status": "ok", "timestamp": 1724748014100, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="3Ir44l5m8Qde" outputId="e8b56b4e-278f-4f9d-f3c9-287fc6e07886"
# Classification report , confusion matrix

import pandas as pd
import matplotlib.pyplot as plt
# Classification Report
print(classification_report(y_test, y_pred_test, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print('Confusion matrix\n\n', cm)

# Visualization of Confusion Matrix
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                         index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()


# %% [markdown] id="b8RbE-Xl5CLU"
# ### Comments
#
#
# - Our original model test accuracy is 0.8501 while GridSearch CV accuracy is 0.8507.
#
#
# - We can see that GridSearch CV improve the performance for this particular model.

# %% [markdown] id="RAfTtPPO5CLV"
# # **21. Results and conclusion** <a class="anchor" id="21"></a>
#
#
# [Table of Contents](#0.1)

# %% [markdown] id="Kc7y-FM95CLV"
# 1.	The logistic regression model accuracy score is 0.8501. So, the model does a very good job in predicting whether or not it will rain tomorrow in Australia.
#
# 2.	Small number of observations predict that there will be rain tomorrow. Majority of observations predict that there will be no rain tomorrow.
#
# 3.	The model shows no signs of overfitting.
#
# 4.	Increasing the value of C results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.
#
# 5.	Increasing the threshold level results in increased accuracy.
#
# 6.	ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.
#
# 7.	Our original model accuracy score is 0.8501 whereas accuracy score after RFECV is 0.8500. So, we can obtain approximately similar accuracy but with reduced set of features.
#
# 8.	In the original model, we have FP = 1175 whereas FP1 = 1174. So, we get approximately same number of false positives. Also, FN = 3087 whereas FN1 = 3091. So, we get slighly higher false negatives.
#
# 9.	Our, original model score is found to be 0.8476. The average cross-validation score is 0.8474. So, we can conclude that cross-validation does not result in performance improvement.
#
# 10.	Our original model test accuracy is 0.8501 while GridSearch CV accuracy is 0.8507. We can see that GridSearch CV improve the performance for this particular model.
#

# %% [markdown] id="FJAckTK-5CLV"
# # **22. References** <a class="anchor" id="22"></a>
#
#
# [Table of Contents](#0.1)
#
#
#
# The work done in this project is inspired from following books and websites:-
#
#
# 1. Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron
#
# 2. Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido
#
# 3. Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves
#
# 4. Udemy course – Feature Engineering for Machine Learning by Soledad Galli
#
# 5. Udemy course – Feature Selection for Machine Learning by Soledad Galli
#
# 6. https://en.wikipedia.org/wiki/Logistic_regression
#
# 7. https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
#
# 8. https://en.wikipedia.org/wiki/Sigmoid_function
#
# 9. https://www.statisticssolutions.com/assumptions-of-logistic-regression/
#
# 10. https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python
#
# 11. https://www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression
#
# 12. https://www.ritchieng.com/machine-learning-evaluate-classification-model/
#

# %% [markdown] id="LBxUvKQA5CLV"
# So, now we will come to the end of this kernel.
#
# I hope you find this kernel useful and enjoyable.
#
# Your comments and feedback are most welcome.
#
# Thank you
#

# %% [markdown] id="WM3-TIFg5CLW"
# [Go to Top](#0)
