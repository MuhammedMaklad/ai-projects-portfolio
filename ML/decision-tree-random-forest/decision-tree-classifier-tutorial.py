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

# %% [markdown] id="R-wpPBQs3l-k"
# <a class="anchor" id="0"></a>
# # **Decision Tree Classifier Tutorial with Python**
#
#
# Hello friends,
#
# In this kernel, I build a Decision Tree Classifier to predict the safety of the car. I build two models, one with criterion `gini index` and another one with criterion `entropy`. I implement Decision Tree Classification with Python and Scikit-Learn.

# %% [markdown] id="ws0SUmzp3l-q"
# <a class="anchor" id="0.1"></a>
# # **Table of Contents**
#
#
# 1.	[Introduction to Decision Tree algorithm](#1)
# 2.	[Classification and Regression Trees](#2)
# 3.	[Decision Tree algorithm terminology](#3)
# 4.	[Decision Tree algorithm intuition](#4)
# 5.	[Attribute selection measures](#5)
#     - 5.1 [Information gain](#5.1)
#     - 5.2 [Gini index](#5.2)
# 6.	[Overfitting in Decision-Tree algorithm](#6)
# 7.	[Import libraries](#7)
# 8.	[Import dataset](#8)
# 9.	[Exploratory data analysis](#9)
# 10.	[Declare feature vector and target variable](#10)
# 11.	[Split data into separate training and test set](#11)
# 12.	[Feature engineering](#12)
# 13.	[Decision Tree classifier with criterion gini-index](#13)
# 14.	[Decision Tree classifier with criterion entropy](#14)
# 15.	[Confusion matrix](#15)
# 16.	[Classification report](#16)
# 17.	[Results and conclusion](#17)
# 18. [References](#18)
#

# %% [markdown] id="yb7M23qj3l-r"
# # **1. Introduction to Decision Tree algorithm** <a class="anchor" id="1"></a>
#
# [Table of Contents](#0.1)
#
# A Decision Tree algorithm is one of the most popular machine learning algorithms. It uses a tree like structure and their possible combinations to solve a particular problem. It belongs to the class of supervised learning algorithms where it can be used for both classification and regression purposes.
#
#
# A decision tree is a structure that includes a root node, branches, and leaf nodes. Each internal node denotes a test on an attribute, each branch denotes the outcome of a test, and each leaf node holds a class label. The topmost node in the tree is the root node.
#
#
# We make some assumptions while implementing the Decision-Tree algorithm. These are listed below:-
#
# 1. At the beginning, the whole training set is considered as the root.
# 2. Feature values need to be categorical. If the values are continuous then they are discretized prior to building the model.
# 3. Records are distributed recursively on the basis of attribute values.
# 4. Order to placing attributes as root or internal node of the tree is done by using some statistical approach.
#
#
# I will describe Decision Tree terminology in later section.
#

# %% [markdown] id="B_PmY3kn3l-r"
# # **2. Classification and Regression Trees (CART)** <a class="anchor" id="2"></a>
#
# [Table of Contents](#0.1)
#
#
# Nowadays, Decision Tree algorithm is known by its modern name **CART** which stands for **Classification and Regression Trees**. Classification and Regression Trees or **CART** is a term introduced by Leo Breiman to refer to Decision Tree algorithms that can be used for classification and regression modeling problems.
#
#
# The CART algorithm provides a foundation for other important algorithms like bagged decision trees, random forest and boosted decision trees. In this kernel, I will solve a classification problem. So, I will refer the algorithm also as Decision Tree Classification problem.
#

# %% [markdown] id="nAXYzKWX3l-s"
# # **4. Decision Tree algorithm intuition** <a class="anchor" id="4"></a>
#
# [Table of Contents](#0.1)
#
# The Decision-Tree algorithm is one of the most frequently and widely used supervised machine learning algorithms that can be used for both classification and regression tasks. The intuition behind the Decision-Tree algorithm is very simple to understand.
#
#
# The Decision Tree algorithm intuition is as follows:-
#
#
# 1.	For each attribute in the dataset, the Decision-Tree algorithm forms a node. The most important attribute is placed at the root node.
#
# 2.	For evaluating the task in hand, we start at the root node and we work our way down the tree by following the corresponding node that meets our condition or decision.
#
# 3.	This process continues until a leaf node is reached. It contains the prediction or the outcome of the Decision Tree.
#

# %% [markdown] id="pHaoDeFb3l-s"
# # **5. Attribute selection measures** <a class="anchor" id="5"></a>
#
# [Table of Contents](#0.1)
#
#
# The primary challenge in the Decision Tree implementation is to identify the attributes which we consider as the root node and each level. This process is known as the **attributes selection**. There are different attributes selection measure to identify the attribute which can be considered as the root node at each level.
#
#
# There are 2 popular attribute selection measures. They are as follows:-
#
#
# - **Information gain**
#
# - **Gini index**
#
#
# While using **Information gain** as a criterion, we assume attributes to be categorical and for **Gini index** attributes are assumed to be continuous. These attribute selection measures are described below.
#

# %% [markdown] id="B137AZfE3l-t"
# ## **5.1 Information gain** <a class="anchor" id="5.1"></a>
#
# [Table of Contents](#0.1)
#
#
# By using information gain as a criterion, we try to estimate the information contained by each attribute. To understand the concept of Information Gain, we need to know another concept called **Entropy**.
#
# ## **Entropy**
#
# Entropy measures the impurity in the given dataset. In Physics and Mathematics, entropy is referred to as the randomness or uncertainty of a random variable X. In information theory, it refers to the impurity in a group of examples. **Information gain** is the decrease in entropy. Information gain computes the difference between entropy before split and average entropy after split of the dataset based on given attribute values.
#
# Entropy is represented by the following formula:-
#
#
#

# %% [markdown] id="FqJofwub3l-t"
# ![Entropy](http://www.learnbymarketing.com/wp-content/uploads/2016/02/entropy-formula.png)
#
#
#
# Here, **c** is the number of classes and **pi** is the probability associated with the ith class.

# %% [markdown] id="yF1jmNCE3l-t"
# The ID3 (Iterative Dichotomiser) Decision Tree algorithm uses entropy to calculate information gain. So, by calculating decrease in **entropy measure** of each attribute we can calculate their information gain. The attribute with the highest information gain is chosen as the splitting attribute at the node.

# %% [markdown] id="OAWJl8Ea3l-u"
# ## **5.2 Gini index** <a class="anchor" id="5.2"></a>
#
# [Table of Contents](#0.1)
#
#
# Another attribute selection measure that **CART (Categorical and Regression Trees)** uses is the **Gini index**. It uses the Gini method to create split points.
#
#
# Gini index can be represented with the following diagram:-

# %% [markdown] id="vt2eCOj93l-u"
# ## **Gini index**
#
# ![Gini index](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzYHkcmZKKp2sJN1HpHvw-NgqbD9EnapnbXozXRgajrSGvEnYy&s)
#
#
# Here, again **c** is the number of classes and **pi** is the probability associated with the ith class.

# %% [markdown] id="gxKIRfpD3l-u"
# Gini index says, if we randomly select two items from a population, they must be of the same class and probability for this is 1 if the population is pure.
#
# It works with the categorical target variable “Success” or “Failure”. It performs only binary splits. The higher the value of Gini, higher the homogeneity. CART (Classification and Regression Tree) uses the Gini method to create binary splits.
#
# Steps to Calculate Gini for a split
#
# 1.	Calculate Gini for sub-nodes, using formula sum of the square of probability for success and failure (p^2+q^2).
#
# 2.	Calculate Gini for split using weighted Gini score of each node of that split.
#
#
# In case of a discrete-valued attribute, the subset that gives the minimum gini index for that chosen is selected as a splitting attribute. In the case of continuous-valued attributes, the strategy is to select each pair of adjacent values as a possible split-point and point with smaller gini index chosen as the splitting point. The attribute with minimum Gini index is chosen as the splitting attribute.

# %% [markdown] id="dLREOjJh3l-v"
# # **6. Overfitting in Decision Tree algorithm** <a class="anchor" id="6"></a>
#
# [Table of Contents](#0.1)
#
#
# Overfitting is a practical problem while building a Decision-Tree model. The problem of overfitting is considered when the algorithm continues to go deeper and deeper to reduce the training-set error but results with an increased test-set error. So, accuracy of prediction for our model goes down. It generally happens when we build many branches due to outliers and irregularities in data.
#
# Two approaches which can be used to avoid overfitting are as follows:-
#
# - Pre-Pruning
#
# - Post-Pruning
#
#
# ## **Pre-Pruning**
#
# In pre-pruning, we stop the tree construction a bit early. We prefer not to split a node if its goodness measure is below a threshold value. But it is difficult to choose an appropriate stopping point.
#
#
# ## **Post-Pruning**
#
# In post-pruning, we go deeper and deeper in the tree to build a complete tree. If the tree shows the overfitting problem then pruning is done as a post-pruning step. We use the cross-validation data to check the effect of our pruning. Using cross-validation data, we test whether expanding a node will result in improve or not. If it shows an improvement, then we can continue by expanding that node. But if it shows a reduction in accuracy then it should not be expanded. So, the node should be converted to a leaf node.

# %% [markdown] id="oa0YmPqE3l-v"
# # **7. Import libraries** <a class="anchor" id="7"></a>
#
# [Table of Contents](#0.1)

# %% id="LPQY5txH3l-v" executionInfo={"status": "ok", "timestamp": 1725007358603, "user_tz": -180, "elapsed": 599, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
# %matplotlib inline

# %% id="rqgsrJ3W3l-w" executionInfo={"status": "ok", "timestamp": 1725007359017, "user_tz": -180, "elapsed": 3, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
import warnings

warnings.filterwarnings('ignore')

# %% colab={"base_uri": "https://localhost:8080/"} id="SPGywrDa8AyO" executionInfo={"status": "ok", "timestamp": 1725007361224, "user_tz": -180, "elapsed": 2209, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="23854f21-5eef-475c-aa2e-ed091532a941"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="CxDaTzxG8Fnv" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 39, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8d31004c-db46-4cb1-885a-f88f17293b12"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Decision Tree and Random forest

# %% colab={"base_uri": "https://localhost:8080/"} id="n8a7424aKK_s" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 38, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d12de277-5d56-40ab-ca15-fc4dea4b4add"
# ls

# %% [markdown] id="N_zzPCjS3l-w"
# # **8. Import dataset** <a class="anchor" id="8"></a>
#
# [Table of Contents](#0.1)

# %% id="cX16G23S3l-x" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 36, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data = 'car_evaluation.csv'

df = pd.read_csv(data, header=None)

# %% [markdown] id="TMEpWv1a3l-x"
# # **9. Exploratory data analysis** <a class="anchor" id="9"></a>
#
# [Table of Contents](#0.1)
#
#
# Now, I will explore the data to gain insights about the data.

# %% colab={"base_uri": "https://localhost:8080/"} id="Vg1tykP63l-x" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 35, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d704cad7-e395-412c-9d2b-1afe9df9fc08"
# view dimensions of dataset

df.shape

# %% [markdown] id="qcBxfutJ3l-x"
# We can see that there are 1728 instances and 7 variables in the data set.

# %% [markdown] id="McgrHVsR3l-x"
# ### View top 5 rows of dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="qM9HURY_3l-x" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 34, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a1317d47-3b76-4ecd-b3c8-fd26c1098c25"
# preview the dataset

df.head()

# %% [markdown] id="p5OE5kP_3l-x"
# ### Rename column names
#
# We can see that the dataset does not have proper column names. The columns are merely labelled as 0,1,2.... and so on. We should give proper names to the columns. I will do it as follows:-

# %% colab={"base_uri": "https://localhost:8080/"} id="HBsa9M623l-y" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 33, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d6bd4236-7eb1-447a-dc64-fd8adc662315"
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


df.columns = col_names

col_names

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="emrvDxDg3l-y" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 32, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c7dcef04-bc2f-4332-dfd7-2dd479378ab4"
# let's again preview the dataset

df.head()

# %% [markdown] id="Vfln-izr3l-y"
# We can see that the column names are renamed. Now, the columns have meaningful names.

# %% [markdown] id="kotNIRpi3l-y"
# ### View summary of dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="wKgk7zFA3l-y" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 31, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="4ea0331c-bba2-4168-ef3c-a2456a7b728b"
df.info()

# %% [markdown] id="yD3-XNG83l-y"
# ### Frequency distribution of values in variables
#
# Now, I will check the frequency counts of categorical variables.

# %% colab={"base_uri": "https://localhost:8080/"} id="XIPErtMw3l-y" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 30, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="3770ad6e-e020-462b-99aa-4a4811dbb164"
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


for col in col_names:

    print(df[col].value_counts())


# %% [markdown] id="nEpBwYmI3l-z"
# ### Summary of variables
#
#
# - There are 7 variables in the dataset. All the variables are of categorical data type.
#
#
# - These are given by `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety` and `class`.
#
#
# - `class` is the target variable.

# %% [markdown] id="neUDng2S3l-z"
# ### Explore `class` variable

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} id="GzLi0oJO3l-z" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 29, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="bf1c370f-182b-40c4-b58e-4925a88b61af"
df['class'].value_counts()

# %% [markdown] id="QQuSPru_3l-z"
# The `class` target variable is ordinal in nature.

# %% [markdown] id="pCpmSN7P3l-5"
# ### Missing values in variables

# %% colab={"base_uri": "https://localhost:8080/", "height": 303} id="cnzFr2_L3l-5" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 28, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="28460e5c-9dcf-4636-e345-529220f8106c"
# check missing values in variables

df.isnull().sum()

# %% [markdown] id="3yKhqkrS3l-5"
# We can see that there are no missing values in the dataset. I have checked the frequency distribution of values previously. It also confirms that there are no missing values in the dataset.

# %% [markdown] id="LvCIofMH3l-5"
# # **10. Declare feature vector and target variable** <a class="anchor" id="10"></a>
#
# [Table of Contents](#0.1)

# %% id="ySVa5EQI3l-5" executionInfo={"status": "ok", "timestamp": 1725007361225, "user_tz": -180, "elapsed": 27, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X = df.drop(['class'], axis=1)

y = df['class']

# %% [markdown] id="0vNvlFdE3l-6"
# # **11. Split data into separate training and test set** <a class="anchor" id="11"></a>
#
# [Table of Contents](#0.1)

# %% id="lEm3PoyH3l-6" executionInfo={"status": "ok", "timestamp": 1725007361636, "user_tz": -180, "elapsed": 438, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# %% colab={"base_uri": "https://localhost:8080/"} id="FhMHoUmy3l-6" executionInfo={"status": "ok", "timestamp": 1725007361637, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="2704e34a-2b3f-4451-b2d2-e502f7358c5e"
# check the shape of X_train and X_test

X_train.shape, X_test.shape

# %% [markdown] id="oPII_VBN3l-6"
# # **12. Feature Engineering** <a class="anchor" id="12"></a>
#
# [Table of Contents](#0.1)
#
#
# **Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. I will carry out feature engineering on different types of variables.
#
#
# First, I will check the data types of variables again.

# %% colab={"base_uri": "https://localhost:8080/", "height": 272} id="Sr6zVQiz3l-6" executionInfo={"status": "ok", "timestamp": 1725007361637, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0263dcb2-053a-43b0-8143-4d0031ea7e34"
# check data types in X_train

X_train.dtypes

# %% [markdown] id="g1kMStti3l-6"
# ### Encode categorical variables
#
#
# Now, I will encode the categorical variables.

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="J_Jl-qi63l-6" executionInfo={"status": "ok", "timestamp": 1725007361637, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="6bef8451-97be-48ee-d64a-3b34885d374f"
X_train.head()

# %% [markdown] id="3WK91Lha3l-7"
# We can see that all  the variables are ordinal categorical data type.

# %% colab={"base_uri": "https://localhost:8080/"} id="CqX2nRrV8l8N" executionInfo={"status": "ok", "timestamp": 1725007366086, "user_tz": -180, "elapsed": 4463, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="1cd673ba-34a3-49c3-90f1-0d22adbbcc5a"
# !pip install category_encoders

# %% id="DoBAKUWK3l-7" executionInfo={"status": "ok", "timestamp": 1725007366086, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# import category encoders

import category_encoders as ce

# %% id="JAdHgYyv3l-7" executionInfo={"status": "ok", "timestamp": 1725007366086, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# encode variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="fp0IhxwP3l-7" executionInfo={"status": "ok", "timestamp": 1725007366086, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="6acb4ecd-22ac-4a41-ef43-bc28ecd11ba4"
X_train.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="7sj8Lwsz3l-8" executionInfo={"status": "ok", "timestamp": 1725007366086, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9ee26f95-1dc4-4b3c-d622-a0bca827924e"
X_test.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="Slezw-20P9Ny" executionInfo={"status": "ok", "timestamp": 1725007369800, "user_tz": -180, "elapsed": 3728, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="65531faa-fc61-4ae9-cc5d-35b1bc6fcd17"
# prompt: implement SMOTEENN

# !pip install imblearn
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=0)
#X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)


# %% colab={"base_uri": "https://localhost:8080/"} id="UyK6U16uX0H8" executionInfo={"status": "ok", "timestamp": 1725007372893, "user_tz": -180, "elapsed": 3100, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e06d9bb1-a5ce-4cba-91b4-2e3bf410c1d9"
# prompt: need to perform SMOTE on X_train , y_train

# !pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

smote = SMOTE()
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# %% colab={"base_uri": "https://localhost:8080/"} id="Is998ZhnN7Me" executionInfo={"status": "ok", "timestamp": 1725007372894, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="bc9dc620-c5ac-4f27-e8a6-3a0c37e8b916"
# prompt: count every classs in y_train_resampled

from collections import Counter
Counter(y_train_resampled)


# %% [markdown] id="BMuePkiX3l-8"
# We now have training and test set ready for model building.

# %% [markdown] id="hzeQ7Key3l-8"
# # **13. Decision Tree Classifier with criterion gini index** <a class="anchor" id="13"></a>
#
# [Table of Contents](#0.1)

# %% id="xmGENE3P3l-8" executionInfo={"status": "ok", "timestamp": 1725007372894, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier  # for classification
from sklearn.tree import DecisionTreeRegressor # for regression


# %% colab={"base_uri": "https://localhost:8080/", "height": 74} id="7JVZtIFe3l-8" executionInfo={"status": "ok", "timestamp": 1725007372894, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="22c4244f-edc9-41bb-8874-afebd602f117"
# instantiate the DecisionTreeClassifier model with criterion gini index

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)


# fit the model
clf_gini.fit(X_train, y_train)


# %% [markdown] id="HCRi4xCj3l-8"
# ### Predict the Test set results with criterion gini index

# %% id="fZa_IHUL3l-9" executionInfo={"status": "ok", "timestamp": 1725007372894, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
y_pred_gini = clf_gini.predict(X_test)


# %% [markdown] id="kP7cXvcs3l-9"
# ### Check accuracy score with criterion gini index

# %% colab={"base_uri": "https://localhost:8080/"} id="Q8Ynsbyk3l-9" executionInfo={"status": "ok", "timestamp": 1725007372894, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="be7e4c23-2a61-48f0-bc0f-1ccfabc931c7"
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

# %% [markdown] id="C05mbGv73l-9"
# Here, **y_test** are the true class labels and **y_pred_gini** are the predicted class labels in the test-set.

# %% [markdown] id="E_WIDlEy3l-9"
# ### Check for overfitting and underfitting

# %% colab={"base_uri": "https://localhost:8080/"} id="QjR8Kzud3l--" executionInfo={"status": "ok", "timestamp": 1725007372894, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d2989bc3-0ac5-4029-e35c-e4f55120f93f"
# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))

# %% [markdown] id="HndbInt73l--"
# Here, the training-set accuracy score is 0.8025 while the test-set accuracy to be 0.8179. These two values are quite comparable. So, there is no sign of overfitting.
#

# %% [markdown] id="k9DToZhL3l--"
# ### Visualize decision-trees with graphviz

# %% colab={"base_uri": "https://localhost:8080/"} id="hTydNXH-85pq" executionInfo={"status": "ok", "timestamp": 1725007376155, "user_tz": -180, "elapsed": 3274, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c3999db4-8748-48ae-a6fc-5dbbf54a4eda"
# !pip install graphviz

# %% colab={"base_uri": "https://localhost:8080/", "height": 598} id="eWA_WRCL3l--" executionInfo={"status": "ok", "timestamp": 1725007376155, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9abe9ca7-f11f-4fc4-9f8e-4c07372f3758"
import graphviz
from sklearn import tree # Import the tree module

dot_data = tree.export_graphviz(clf_gini, out_file=None,
                              feature_names=X_train.columns,
                              class_names=y_train,
                              filled=True, rounded=True,
                              special_characters=True)

graph = graphviz.Source(dot_data)

graph

# %% [markdown] id="4e4DZA5q3l--"
# # **14. Decision Tree Classifier with criterion entropy** <a class="anchor" id="14"></a>
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 74} id="hP97l0al3l--" executionInfo={"status": "ok", "timestamp": 1725007376155, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="5cf3e359-3815-401d-ae17-7a202497f140"
# instantiate the DecisionTreeClassifier model with criterion entropy

clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)


# fit the model
clf_en.fit(X_train, y_train)

# %% [markdown] id="NCvhtXxR3l--"
# ### Predict the Test set results with criterion entropy

# %% id="sfAVi5Jj3l-_" executionInfo={"status": "ok", "timestamp": 1725007376155, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
y_pred_en = clf_en.predict(X_test)

# %% [markdown] id="WbYWm-z_3l-_"
# ### Check accuracy score with criterion entropy

# %% colab={"base_uri": "https://localhost:8080/"} id="pihHSIJ53l-_" executionInfo={"status": "ok", "timestamp": 1725007376156, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="af109014-6aac-4d52-c16c-fcbcfd052a3e"
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

# %% [markdown] id="r5OPCuEJ3l-_"
# ### Compare the train-set and test-set accuracy
#
#
# Now, I will compare the train-set and test-set accuracy to check for overfitting.

# %% colab={"base_uri": "https://localhost:8080/"} id="n50Nk7DL3l-_" executionInfo={"status": "ok", "timestamp": 1725007376156, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9522e350-e1d5-4fd7-fc25-5e44451237ad"
y_pred_train_en = clf_en.predict(X_train)

y_pred_train_en

# %% colab={"base_uri": "https://localhost:8080/"} id="6fxtBPBx3l-_" executionInfo={"status": "ok", "timestamp": 1725007376156, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="933ccc7b-aafb-4a5d-9fbe-807730293210"
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))

# %% [markdown] id="9gui2syo3l-_"
# ### Check for overfitting and underfitting

# %% colab={"base_uri": "https://localhost:8080/"} id="Z0RCJl4K3l-_" executionInfo={"status": "ok", "timestamp": 1725007376156, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="73c03910-28c6-4342-c493-870f6c66de05"
# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))

# %% [markdown] id="77uxG2v03l_A"
# We can see that the training-set score and test-set score is same as above. The training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting.
#

# %% [markdown] id="eHc6hyvq3l_A"
# ### Visualize decision-trees

# %% [markdown] id="9t51zOA63l_A"
# ### Visualize decision-trees with graphviz

# %% colab={"base_uri": "https://localhost:8080/", "height": 598} id="Jpeeoh753l_A" executionInfo={"status": "ok", "timestamp": 1725007376156, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="22f1340a-e83d-4833-b410-074cd8e7712e"
import graphviz
dot_data = tree.export_graphviz(clf_en, out_file=None,
                              feature_names=X_train.columns,
                              class_names=y_train,
                              filled=True, rounded=True,
                              special_characters=True)

graph = graphviz.Source(dot_data)

graph

# %% [markdown] id="L2qukHu83l_A"
# Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.
#
#
# But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making.
#
#
# We have another tool called `Confusion matrix` that comes to our rescue.

# %% colab={"base_uri": "https://localhost:8080/"} id="Mu_aDmZoSbNW" executionInfo={"status": "ok", "timestamp": 1725007385828, "user_tz": -180, "elapsed": 9686, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b2d2463c-7621-457e-8b24-ae4b68731372"
# prompt: need to implement Grid search using Decsison trees

from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

best_model = grid_search.best_estimator_


# %% colab={"base_uri": "https://localhost:8080/"} id="uWf9NR9xS1nJ" executionInfo={"status": "ok", "timestamp": 1725007385829, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="32d7bb84-8be8-49ea-880c-9132774ab14a"
# Print the results of training and testing  using best_model

# Predict on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

# You can also print the training accuracy if you like
train_accuracy = best_model.score(X_train, y_train)
print("Train Accuracy:", train_accuracy)


# %% [markdown] id="gZYxC6ve3l_A"
# # **15. Confusion matrix** <a class="anchor" id="15"></a>
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 410} id="8U12WSOtUmOM" executionInfo={"status": "ok", "timestamp": 1725007385829, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9493c272-f86d-4bcc-f717-b9c022b452e3"
# Draw the cm using heatmap with class names

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# %% [markdown] id="WOLRUGd13l_A"
# # **16. Classification Report** <a class="anchor" id="16"></a>
#
# [Table of Contents](#0.1)
#
#
# **Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later.
#
# We can print a classification report as follows:-

# %% colab={"base_uri": "https://localhost:8080/"} id="xsRfJbr33l_B" executionInfo={"status": "ok", "timestamp": 1725007385829, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="cecb4cf7-db08-4cdc-ec8a-5e25820992b4"
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, digits=4))

# %% colab={"base_uri": "https://localhost:8080/"} id="-XiObpEpUDTr" executionInfo={"status": "ok", "timestamp": 1725007389274, "user_tz": -180, "elapsed": 3459, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="831b554f-c210-41b4-bdbb-845686a2677e"
# implement logistic regression using grid search

from sklearn.linear_model import LogisticRegression

param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

logistic_reg = LogisticRegression()

grid_search_logistic = GridSearchCV(estimator=logistic_reg, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_logistic.fit(X_train, y_train)

print("Best hyperparameters for Logistic Regression:", grid_search_logistic.best_params_)
print("Best accuracy score for Logistic Regression:", grid_search_logistic.best_score_)

best_model_logistic = grid_search_logistic.best_estimator_

# Predict on the test set using the best logistic regression model
y_pred_logistic = best_model_logistic.predict(X_test)

# Evaluate the best logistic regression model
test_accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("Test Accuracy (Logistic Regression):", test_accuracy_logistic)

# You can also print the training accuracy if you like
train_accuracy_logistic = best_model_logistic.score(X_train, y_train)
print("Train Accuracy (Logistic Regression):", train_accuracy_logistic)


# %% colab={"base_uri": "https://localhost:8080/"} id="gMrFdBUgcW2H" executionInfo={"status": "ok", "timestamp": 1725007642172, "user_tz": -180, "elapsed": 252904, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="edc4b9c5-e765-4cef-fffe-65b79c72740b"
# Implement Random forest with grid search

from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [ 5,8, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier()

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

print("Best hyperparameters for Random Forest:", grid_search_rf.best_params_)
print("Best accuracy score for Random Forest:", grid_search_rf.best_score_)

best_model_rf = grid_search_rf.best_estimator_

# Predict on the test set using the best random forest model
y_pred_rf = best_model_rf.predict(X_test)

# Evaluate the best random forest model
test_accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Test Accuracy (Random Forest):", test_accuracy_rf)

# You can also print the training accuracy if you like
train_accuracy_rf = best_model_rf.score(X_train, y_train)
print("Train Accuracy (Random Forest):", train_accuracy_rf)


# %% colab={"base_uri": "https://localhost:8080/", "height": 610} id="p39WOMnceT-o" executionInfo={"status": "ok", "timestamp": 1725007836943, "user_tz": -180, "elapsed": 975, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="3c9dc3a5-bd62-483e-ef0b-48c7e27838e6"
# Confusion matrix with heat map and classification report

import matplotlib.pyplot as plt
import numpy as np
# Assuming 'y_test' and 'y_pred' are already defined from your previous code

# Confusion Matrix with Heatmap
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred_rf, digits=4))


# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="7VCXB4Ytevcn" executionInfo={"status": "ok", "timestamp": 1725007967639, "user_tz": -180, "elapsed": 3357, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f9417e06-5bf0-4520-eb01-b7797c0916bc"
# prompt: need to print roc auc curve

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

# Binarize the output
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Learn to predict each class against the other
classifier = OneVsRestClassifier(best_model_rf)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot all ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(np.unique(y_test)[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] id="a_W0KIIP3l_B"
# # **18. References** <a class="anchor" id="18"></a>
#
# [Table of Contents](#0.1)
#
#
# The work done in this project is inspired from following books and websites:-
#
# 1. Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron
#
# 2. Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido
#
# 3. https://en.wikipedia.org/wiki/Decision_tree
#
# 4. https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
#
# 5. https://en.wikipedia.org/wiki/Entropy_(information_theory)
#
# 6. https://www.datacamp.com/community/tutorials/decision-tree-classification-python
#
# 7. https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
#
# 8. https://acadgild.com/blog/decision-tree-python-code
#
