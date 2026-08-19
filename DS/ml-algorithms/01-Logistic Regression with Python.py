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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ___
#
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Logistic Regression with Python
#
# For this lecture we will be working with the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). This is a very famous data set and very often is a student's first step in machine learning! 
#
# We'll be trying to predict a classification- survival or deceased.
# Let's begin our understanding of implementing Logistic Regression in Python for classification.
#
# We'll use a "semi-cleaned" version of the titanic data set, if you use the data set hosted directly on Kaggle, you may need to do some additional cleaning not shown in this lecture notebook.
#
# ## Import Libraries
# Let's import some libraries to get started!

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# Suppress warnings temporarily
warnings.filterwarnings("ignore")
# %matplotlib inline

# %% [markdown]
# ## The Data
#
# Let's start by reading in the titanic_train.csv file into a pandas dataframe.

# %%
train = pd.read_csv('titanic_train.csv')

# %% jupyter={"outputs_hidden": false}
train.head()

# %% [markdown]
# **A.1 Numerical Attributes**
#
# From the codes above, we obtain the descriptive statistics for numerical attributes: **What do we see here?**
# 1. **Survived**: The sample mean of this training data is 0,38, which could means *only about that percentage of passengers survived from titanic accident*
#
# 2. **Pclass** (Passenger Class:Tthere are 3 class of passenger. At Q2(50%) and Q3(75%) we could see the value is 3, which could means *there are minimum 50% (or more) passengers which is 3rd class passengers*. It seems logical since lower class usually have cheaper ticket prize, and more quota for that class
#
# 3. **Age**: From train and test data, the count values seems different from the others. yes, **Age attribute contains missing values**. Another useful information, the mean/average age on training data is 29 years old, which is 1 years older than the median value of the mean (30 mean and 27 median on test dataset), so what does it mean?
#     
#     It means the distributions of age values have **right skew**, which we expect some outliers in the *higher age value* (on the right size ofthe axis. As we can see, on the training and test dataset max value is 80 and 76 respectively.
#     
# 4. **SibSp and Parch**: These attributes indicate number of SIblings or spouses, and Parent or Children number aboard. From the mean value, seems *majority of the passengers is alone (neither have SibSp or Parch)*. It is interesting that we see the maximum value have 8 SibSp and 9 ParCh, *maybe the oldest person brought his/her entire family on the ship*
#
# 5. **Fare**: There are huge difference between mean and median value of this attributes, which is logical. *Many passengers from 3rd class which always have lower Fare*, on the other hand, we have so high value on max of Fare here, which seems an outlier that affect the average of this attributes (**again, right skew**). **Fare attribute contain 1 missing value on test dataset**

# %% [markdown]
# **A.2 Categorical Attributes** <br>
# Now, we're dealing with categorical attributes. From the describe method above, we get additional information:
# 1. **Name**: All names are unique (nothing special,) *but they contain title* - maybe we can perform feature engineering later to produce a new attribute (Title) which could improve model performance
#
# 2.  **Sex**: Or *gender*. Consists of 2 categories - male and female. In both training and test datasets, male have higher frequency (approximately 60:40.)
#
# 3.  **Ticket**: There are many unique values for this attributes - maybe we'll just drop this attribute for now and include it for future research
#
# 4. **Cabin**: Many **missing values** here (*204 filled from 891 possible* on training dataset and *91 filled from 418 possible* on test dataset). *Maybe some passengers*, which we already know, 3rd class or low-Fare passengers, **do not have Cabin**.
#
# 5. **Embarked**: There are **2 missing values** on training dataset. From the train and test datasets, we know that most passengers embarked from S (*what is "S" anyway?*)

# %% [markdown]
# # Exploratory Data Analysis
#
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
#
# ## Missing Data
#
# We can use seaborn to create a simple heatmap to see where we are missing data!

# %% jupyter={"outputs_hidden": false}
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# %% [markdown]
# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
#
# Let's continue on by visualizing some more of the data! Check out the video for full explanations over these plots, this code is just to serve as reference.

# %% jupyter={"outputs_hidden": false}
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')

# %% jupyter={"outputs_hidden": false}
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

# %% jupyter={"outputs_hidden": false}
#sns.set_style('whitegrid')
#sns.countplot(x='Pclass',hue='Survived',data=train,palette='rainbow')
#sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

# %% jupyter={"outputs_hidden": false}
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)

# %% jupyter={"outputs_hidden": false}
train['Age'].hist(bins=30,color='darkred',alpha=0.7)

# %% jupyter={"outputs_hidden": false}
sns.countplot(x='SibSp',data=train)

# %% jupyter={"outputs_hidden": false}
train['Fare'].hist(color='green',bins=40,figsize=(8,4))

# %% [markdown]
# ____
# ### Cufflinks for plots
# ___
#  Let's take a quick moment to show an example of cufflinks!

# %% jupyter={"outputs_hidden": false}
import cufflinks as cf
cf.go_offline()

# %% jupyter={"outputs_hidden": false}
train['Fare'].iplot(kind='hist',bins=30,color='green')

# %% [markdown]
# ___
# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class. For example:
#

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# %% [markdown]
# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# %% jupyter={"outputs_hidden": false}
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# %% [markdown]
# Now apply that function!

# %% jupyter={"outputs_hidden": false}
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

# %% [markdown]
# Now let's check that heat map again!

# %% jupyter={"outputs_hidden": false}
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# %% [markdown]
# Great! Let's go ahead and drop the Cabin column and the row in Embarked that is NaN.

# %% jupyter={"outputs_hidden": false}
train.drop('Cabin',axis=1,inplace=True)

# %% jupyter={"outputs_hidden": false}
train.head()

# %% jupyter={"outputs_hidden": false}
train.dropna(inplace=True)

# %% [markdown]
# ## Converting Categorical Features 
#
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# %% jupyter={"outputs_hidden": false}
train.info()

# %% jupyter={"outputs_hidden": false}
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

# %% jupyter={"outputs_hidden": false}
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

# %%
train = pd.concat([train,sex,embark],axis=1)

# %% jupyter={"outputs_hidden": false}
train.head()

# %% [markdown]
# Great! Our data is ready for our model!
#
# # Building a Logistic Regression model
#
# Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
#
# ## Train Test Split

# %%
from sklearn.model_selection import train_test_split

# %% jupyter={"outputs_hidden": false}
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

# %% [markdown]
# ## Training and Predicting

# %%
from sklearn.linear_model import LogisticRegression

# %% jupyter={"outputs_hidden": false}
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# %%
predictions = logmodel.predict(X_test)

# %% [markdown]
# Let's move on to evaluate our model!

# %% [markdown]
# ## Evaluation

# %% [markdown]
# We can check precision,recall,f1-score using classification report!

# %%
from sklearn.metrics import classification_report

# %% jupyter={"outputs_hidden": false}
print(classification_report(y_test,predictions))

# %% [markdown]
# Not so bad! You might want to explore other feature engineering and the other titanic_text.csv file, some suggestions for feature engineering:
#
# * Try grabbing the Title (Dr.,Mr.,Mrs,etc..) from the name as a feature
# * Maybe the Cabin letter could be a feature
# * Is there any info you can get from the ticket?
#
# ## Great Job!

# %%
