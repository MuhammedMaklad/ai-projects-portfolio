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

# %% executionInfo={"elapsed": 376, "status": "ok", "timestamp": 1725352991911, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="3a9ad9e5"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1915, "status": "ok", "timestamp": 1725352994128, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="LuoJFCPvnGJA" outputId="8153cd36-7e24-443a-b259-1fa3076b5751"
# from google.colab import drive
# drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1725352994129, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="0sewMXvPnHB3" outputId="10059dc8-5f2b-472f-9231-ab2abd3c1eb9"
# # cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/KNN

# %% [markdown] id="8ce8b293"
# # Import Packages

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1725352994129, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="f81ef46f"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# %% colab={"base_uri": "https://localhost:8080/", "height": 678} executionInfo={"elapsed": 345, "status": "ok", "timestamp": 1725352994469, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="8138b57f" outputId="24a57885-d79a-4b89-ed54-ab8d9c454f46"
data = pd.read_csv("adult.csv")
data.head(10)

# %% [markdown] id="942dc904"
# Attribute Information:
#  1. age: continuous.
#  2. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#  3. fnlwgt: continuous.
#  4. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#  5. education-num: continuous.
#  6. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#  7. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving,   Priv-house-serv, Protective-serv, Armed-Forces.
#  8. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#  9. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#  10. sex: Female, Male.
#  11. capital-gain: continuous.
#  12. capital-loss: continuous.
#  13. hours-per-week: continuous.
#  14. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia,El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
#  class: >50K, <=50K
#
# ### Notes for Delve
# 1. One prototask (income) has been defined, using attributes 1-13 as inputs and income level as a binary target.
# 2. Missing values - These are confined to attributes 2 (workclass), 7 (occupation) and 14 (native-country). The prototask only uses cases with no missing values.
# 3. The income prototask comes with two priors, differing according to if attribute 4 (education) is considered to be nominal or ordinal.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1725352994469, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="23610878" outputId="de9fe028-51c8-40c8-8a49-4a1cd7b94505"
print(data.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 554} executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1725352994469, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="2bab3edb" outputId="9315baca-6961-44be-dc3d-127b9f56b0a0"
data.isin(['?']).sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 554} id="k_CsXeDQrBio" executionInfo={"status": "ok", "timestamp": 1725352994469, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="33a8444d-6a48-4b61-bbbe-9cba89ce2c39"
data.isna().sum()

# %% [markdown] id="1e51e171"
# #### Replacing ? with Nan

# %% executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1725352994469, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="0ce69a6e"
data['workclass']=data['workclass'].replace('?',np.nan)
data['occupation']=data['occupation'].replace('?',np.nan)
data['native-country']=data['native-country'].replace('?',np.nan)

# %% colab={"base_uri": "https://localhost:8080/", "height": 554} id="OVCTylcTrIQh" executionInfo={"status": "ok", "timestamp": 1725352994788, "user_tz": -180, "elapsed": 337, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="1d9291b8-cdbc-4e4e-baaf-71a70cd3c687"
data.isna().sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 229} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1725352994788, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="00d855a8" outputId="b5e31a64-909e-474e-9157-cd3f92a2eca8"
# Check For Missing Values
info = pd.DataFrame(data.isnull().sum(),columns=["IsNull"])
info.insert(1,"IsNa",data.isna().sum(),True)
info.insert(2,"Duplicate",data.duplicated().sum(),True)
info.insert(3,"Unique",data.nunique(),True)
info.T

# %% [markdown] id="33d848dd"
# results:
# 1. no null or Nan Value.but the dataset is not using the default nan string for missing values, instead "?" is used. after convert '?' to Nan three columns ['workclass','occupation','native-country] have nan
# 2. there are 52 duplicated row.
#

# %% executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1725352994788, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="b73d45dc"
df = data.copy()

# %% [markdown] id="38b51829"
# ## Remove nan value

# %% colab={"base_uri": "https://localhost:8080/", "height": 756} executionInfo={"elapsed": 361, "status": "ok", "timestamp": 1725352995139, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4924f9f9" outputId="ddeaf319-1db8-4faa-e0cf-e695fcb6d095"
df.dropna(how='any',inplace=True)
df

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1725352995139, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="d156c7fe" outputId="8aacae09-10c0-448b-e922-87a6f6568927"
df.shape

# %% [markdown] id="bebcbdc4"
# ## Remove Duplicated

# %% colab={"base_uri": "https://localhost:8080/", "height": 756} executionInfo={"elapsed": 405, "status": "ok", "timestamp": 1725352995535, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="24697255" outputId="229f67aa-e9fc-4663-ebe7-b86e334222b0"
df = df.drop_duplicates()
df

# %% [markdown] id="2bff3648"
# # remove educational-num, capital-gain, capital-loss

# %% [markdown] id="f5e031c8"
# #### Check 'education' & 'educational-num' columns

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1725352995535, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="77352d14" outputId="816a8fd2-264b-42eb-e753-8b748a3c21a2"
df['education'].unique()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1725352995535, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="29b14e96" outputId="524f1e85-c312-497e-8a64-0af7875e9bdc"
df['educational-num'].unique()

# %% [markdown] id="1e6c3ad9"
# * As we can see, 'education' & 'educational-num' both columns have similer values.
# * 'education' has string values
# * 'educational-num' has numerical values
# * all HS-grad from education represents 9 in education-num
# * all Masters from education represents 14 in education-num and likewise.
# * Both columns conveying same information
# * So, we can remove 'education-num' colum

# %% [markdown] id="c24994d5"
# #### Drop capital-gain, and capital-loss columns¶
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 456} executionInfo={"elapsed": 2294, "status": "ok", "timestamp": 1725352997820, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="358b63cd" outputId="cf424e0f-30fb-4cc3-9433-40769afe7296"
plt.figure(figsize=(20, 7))
plt.subplot(1, 2, 1)
sns.histplot(df['capital-gain'], kde = True,color='r')
plt.title('Histogram')
plt.subplot(1, 2, 2)
sns.histplot(df['capital-loss'], kde = True,color='r')
plt.title('Histogram')

# %% [markdown] id="8f4ff00d"
# [ 'capital-gain' ] & [ 'capital-loss' ] both columns have 75% data as 0.00
# So, we can drop [ 'capital-gain' ] & [ 'capital-loss' ] both columns

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1725352997820, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="43ff54d8"
df1 = df.drop(['educational-num', 'capital-gain', 'capital-loss'], axis=1)

# %% [markdown] id="f25dd828"
# ### Convert categorical cate to numerical using LableEncoder

# %% colab={"base_uri": "https://localhost:8080/"} id="yvav4HVftVKY" executionInfo={"status": "ok", "timestamp": 1725352997820, "user_tz": -180, "elapsed": 6, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="28d489b3-a86b-4225-9329-580d683a64ad"
df1.info()

# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1725352997820, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="71733a49"
from sklearn import preprocessing

# %% colab={"base_uri": "https://localhost:8080/", "height": 293} id="hOBDScGCuKCH" executionInfo={"status": "ok", "timestamp": 1725352998240, "user_tz": -180, "elapsed": 425, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="899edfb9-ac58-47c9-eac6-663ae02691ea"
df1.head()

# %% executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1725352998240, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="08f426b6"
label_encoder = preprocessing.LabelEncoder()

df1['gender'] = label_encoder.fit_transform(df1['gender'])
df1['workclass'] = label_encoder.fit_transform(df1['workclass'])
df1['education'] = label_encoder.fit_transform(df1['education'])
df1['marital-status'] = label_encoder.fit_transform(df1['marital-status'])
df1['occupation'] = label_encoder.fit_transform(df1['occupation'])
df1['relationship'] = label_encoder.fit_transform(df1['relationship'])
df1['race'] = label_encoder.fit_transform(df1['race'])
df1['native-country'] = label_encoder.fit_transform(df1['native-country'])
df1['income'] = label_encoder.fit_transform(df1['income'])

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} executionInfo={"elapsed": 399, "status": "ok", "timestamp": 1725352998627, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="aZeiwjrMj0er" outputId="819978b2-c804-4c6b-cc6f-03589132bec4"
df1.head(10)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1725352998627, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="QEKKhonVRQ7G" outputId="33117a01-d7a7-4060-cde9-54f331f08635"
df1.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1725352998627, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="482c9e2a" outputId="76038392-5b4c-4461-fa66-cb4c7f19553f"
# Check For Missing Values
info = pd.DataFrame(df1.isnull().sum(),columns=["IsNull"])
info.insert(1,"IsNa",df1.isna().sum(),True)
info.T


# %% colab={"base_uri": "https://localhost:8080/", "height": 935} executionInfo={"elapsed": 2878, "status": "ok", "timestamp": 1725353001492, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="885d423b" outputId="5e086304-d668-4ab7-c674-bf7f73be8557"
# Correlation matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df1.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


# %% id="fca3034f" executionInfo={"status": "ok", "timestamp": 1725353001492, "user_tz": -180, "elapsed": 2, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
def diagnostic_plots(df, variable,target):
    # The function takes a dataframe (df) and
    # the variable of interest as arguments.

    # Define figure size.
    plt.figure(figsize=(20, 7))

    # histogram
    plt.subplot(1, 4, 1)
    sns.histplot(df[variable], kde = True,color='r')
    plt.title('Histogram')


    # scatterplot
    plt.subplot(1, 4, 2)
    plt.scatter(df[variable],df[target],color = 'g')
    plt.title('Scatterplot')


    # boxplot
    plt.subplot(1, 4, 3)
    sns.boxplot(y=df[variable],color = 'b')
    plt.title('Boxplot')

    # barplot
    plt.subplot(1, 4, 4)
    sns.barplot(x = target, y = variable, data = df)
    plt.title('Barplot')


    plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 46306, "status": "ok", "timestamp": 1725353047796, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="94a8f3ff" outputId="1f70a327-58d8-4de0-f86b-51b485e4df3c"
for col in df1:
    diagnostic_plots(df1,col,'income')

# %% [markdown] id="f80a7299"
# # Model

# %% id="264c221f" executionInfo={"status": "ok", "timestamp": 1725353047797, "user_tz": -180, "elapsed": 8, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.model_selection import train_test_split
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# %% id="180ac2ef" executionInfo={"status": "ok", "timestamp": 1725353047797, "user_tz": -180, "elapsed": 8, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# split Data
X = df1.drop(columns={"income"},axis=1)
y =df1["income"].values.reshape(-1,1) #target

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1725353047797, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="c9b7a8f2" outputId="24c0bd64-bb50-40fb-fd03-76c3b3243391"
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=1)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# %% id="6GTsU1SQDW_l" executionInfo={"status": "ok", "timestamp": 1725353048168, "user_tz": -180, "elapsed": 3, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Robust scaling X_train and X_test

from sklearn.preprocessing import RobustScaler
robus_scaler = RobustScaler()
X_train = robus_scaler.fit_transform(X_train)
X_test = robus_scaler.transform(X_test)


# %% [markdown] id="3e494ad6"
# ### Scaling Data

# %% id="d647fe8c"
from sklearn.preprocessing import MinMaxScaler

st =MinMaxScaler()
X_train = st.fit_transform(X_train)
X_test = st.fit_transform(X_test)

# %% id="55cf0fcd" executionInfo={"status": "ok", "timestamp": 1725353141932, "user_tz": -180, "elapsed": 89070, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
#Find Best K
K = 40
error =[]
accuracy=[]
for i in range(1,K+1):
    knn= KNeighborsClassifier(n_neighbors= i) # n_neighbors = K
    knn.fit(X_train,y_train)
    y_pred =knn.predict(X_test)
    error.append(1-metrics.accuracy_score(y_test,y_pred))
    accuracy.append(metrics.accuracy_score(y_test,y_pred))


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1725353141932, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="PrFXVDhhyH3C" outputId="7518d9ec-2c7d-4e8a-9852-1422a3bb1eb0"
print(error)
print(accuracy)

# %% colab={"base_uri": "https://localhost:8080/", "height": 461} executionInfo={"elapsed": 855, "status": "ok", "timestamp": 1725353142782, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="b7773538" outputId="ac1249f9-5152-46d8-cb43-e68e173d6ec5"

plt.figure(figsize=(20, 7))
plt.subplot(1, 2, 1)
plt.plot(range(1,41),error,'r-',marker='o')
plt.xlabel('Values of K')
plt.ylabel('Error')
plt.grid()
plt.title('Error vs K')

plt.subplot(1, 2, 2)
plt.plot(range(1,41),accuracy,'r-',marker='o')
plt.xlabel('Values of K')
plt.ylabel('accuracy')
plt.grid()
plt.title('accuracy vs K')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1923, "status": "ok", "timestamp": 1725353144704, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="e87b7505" outputId="0569d103-e9a2-45e6-db8b-76c5bdae193c"
K=34
clf= KNeighborsClassifier(K)
clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
print("Accuracy :" ,metrics.accuracy_score(y_test,y_pred))

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} executionInfo={"elapsed": 643825, "status": "ok", "timestamp": 1725353788527, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="6cc06a03" outputId="8d1ad153-5b3b-479d-eca4-f05889ab1f0d"
#Find Best K
training_acc = []
test_acc = []

# try KNN for different K nearest neighbor from 1 to 30
neighbors_setting = range(1,35)
for n_neighbors in neighbors_setting:
    knn= KNeighborsClassifier(n_neighbors= n_neighbors, )
    knn.fit(X_train,y_train.ravel())
    training_acc.append(knn.score(X_train,y_train))
    test_acc.append(knn.score(X_test,y_test))

plt.plot(neighbors_setting,training_acc,label='Accuracy of the training set')
plt.plot(neighbors_setting,test_acc,label='Accuracy of the test set')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
plt.grid()
plt.legend()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1725353788527, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ZNoE84PoV2Uj" outputId="9915fbcf-c34d-443f-c846-1bfc288d09c5"
print(training_acc[-1], test_acc[-1])

# %% colab={"base_uri": "https://localhost:8080/", "height": 135} executionInfo={"elapsed": 746792, "status": "ok", "timestamp": 1725354535312, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1381e850" outputId="9872256f-2f94-46f5-968d-87eea8a59491"
#Find Best K

param_grid = {
    'n_neighbors': [ 7, 9, 11, 13, 15,25, 34],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_kn=GridSearchCV(estimator=knn,#model
                     param_grid=param_grid, #range of K
                    scoring='accuracy',# Strategy to evaluate the performance of the cross-validation model on the test set.
                    cv= 5, # Cross-validation generator
                    verbose= 1, #Time to calculate
                    n_jobs= -1 #Help to cpu
                    )
grid_kn.fit(X_train,y_train.ravel())

# %% colab={"base_uri": "https://localhost:8080/"} id="f90e9841" outputId="0cba65ea-7497-40a5-9337-b6b7f05c3726" executionInfo={"status": "ok", "timestamp": 1725354535313, "user_tz": -180, "elapsed": 61, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
grid_kn.best_params_

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3018, "status": "ok", "timestamp": 1725354538279, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="zqJr8F3QqebP" outputId="0392893a-2068-4c7e-c47a-9815fb0d333b"
best_knn = grid_kn.best_estimator_

# Make predictions on the test set
y_pred_best = best_knn.predict(X_test)

# Evaluate the model
accuracy_best = metrics.accuracy_score(y_test, y_pred_best)
print("Accuracy with best K (24):", accuracy_best)


# %% colab={"base_uri": "https://localhost:8080/", "height": 573} id="vpasZd6ZzDba" executionInfo={"status": "ok", "timestamp": 1725354538752, "user_tz": -180, "elapsed": 487, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a2a95913-31dd-4de7-edfc-bdec414df7e1"
# prompt: classification report , confusion matrix with heat map

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Classification report
print(classification_report(y_test, y_pred_best, digits=4))

# Confusion matrix with heat map
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

