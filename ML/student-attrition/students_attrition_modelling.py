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

# %% [markdown] id="ZZH8lrde6B66"
# #                    Early prediction of college attrition using Data Mining

# %% [markdown] id="bxNq-vCE6B7F"
# ## Import the libraries used

# %% id="pBKRsi2_6B7G" executionInfo={"status": "ok", "timestamp": 1726559297226, "user_tz": -180, "elapsed": 931, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% colab={"base_uri": "https://localhost:8080/"} id="cABeFPiXnpxs" executionInfo={"status": "ok", "timestamp": 1726559299668, "user_tz": -180, "elapsed": 1824, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a4ee505e-41e2-4f32-d7c4-87055e58331f"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="meedF3ISnrYM" executionInfo={"status": "ok", "timestamp": 1726559299668, "user_tz": -180, "elapsed": 6, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8e52e2d1-e83a-4826-8c57-488fc5735be8"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/student attrition

# %% [markdown] id="Hu0EpjVO6B7H"
# ## Read the dataset using pandas library

# %% id="6klHTwTN6B7I" executionInfo={"status": "ok", "timestamp": 1726559302971, "user_tz": -180, "elapsed": 3308, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data=pd.read_excel("Student.xlsx")

# %% id="5LbtArIs3JlY" outputId="df510a6f-7c37-43b5-9d1b-1cb3cffbc270" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1726559302978, "user_tz": -180, "elapsed": 26, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.shape

# %% [markdown] id="cY5QD8dK6B7I"
# ## Print the first five rows from the dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 273} id="gs8uevR76B7J" outputId="6046cb74-dc87-4667-cfa1-993675b39d5d" executionInfo={"status": "ok", "timestamp": 1726559302978, "user_tz": -180, "elapsed": 24, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.head()

# %% [markdown] id="zf9fDQXQ6B7L"
# ## Calculate the number of null values in every features in the dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="Bxt2upebFWvg" executionInfo={"status": "ok", "timestamp": 1726559302978, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8c36f881-250d-4234-c55b-4f5e3ed88a12"
data.info()

# %% id="q37onAUt6B7L" outputId="963832dc-d61c-44ce-e45f-a1ee8171009c" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1726559302978, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.isna().sum()

# %% [markdown] id="fIOLU0GN6B7M"
# ## Print the features names in the dataset

# %% id="ZtdzqkZg6B7M" outputId="95306134-d064-4fe1-ed27-a17f226d5bdb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1726559302978, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.columns

# %% [markdown] id="N6UJyCHj6B7N"
# ## Replace the year with month into only year because the month is repeated in all dataset

# %% id="mrqYoYQ76B7N" executionInfo={"status": "ok", "timestamp": 1726559302979, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data['FIRST_TERM']=data['FIRST_TERM'].replace({200508:2005, 200608:2006, 200708:2007, 200808:2008, 200908:2009, 201008:2010})

# %% id="kyK31B-y6B7O" executionInfo={"status": "ok", "timestamp": 1726559302979, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data['SECOND_TERM']=data['SECOND_TERM'].replace({200602:2006, 200702:2007, 200802:2008, 200902:2009, 201002:2010, 201102:2011})

# %% id="H0X_bwvR3Jla" outputId="0b0eb12b-4b99-4e0b-8321-9d321e33d8cc" colab={"base_uri": "https://localhost:8080/", "height": 443} executionInfo={"status": "ok", "timestamp": 1726559302979, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.iloc[:,15:25]

# %% [markdown] id="5nn0RYmu6B7O"
# ## Calculate the number of student that returned in the second year and the other that are not returned

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="EYoAY4-q6B7P" outputId="9cf2d8e9-283d-4180-b0a3-1ae00ebaf20f" executionInfo={"status": "ok", "timestamp": 1726559303404, "user_tz": -180, "elapsed": 442, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data['RETURNED_2ND_YR'].value_counts()

# %% [markdown] id="g_zVbe4h6B7P"
# ## Fill the empty values of FATHER_HI_EDU_CD and MOTHER_HI_EDU_CD with zeros because they are not availabe

# %% id="TuJbwTPX6B7Q" executionInfo={"status": "ok", "timestamp": 1726559303404, "user_tz": -180, "elapsed": 27, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
#Replacing all of them with a new value 0.0 as they dont provide any info
data['FATHER_HI_EDU_CD']=data['FATHER_HI_EDU_CD'].fillna(0.0)
data['MOTHER_HI_EDU_CD']=data['MOTHER_HI_EDU_CD'].fillna(0.0)

# %% [markdown] id="je0RZHFe6B7u"
# ## We can represnt the first term hours by dividing the two features (FIRST_TERM_EARNED_HRS , FIRST_TERM_ATTEMPT_HRS)

# %% id="zGZlhoTg6B7v" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 28, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data['FIRST_TERM_Hr']=data['FIRST_TERM_EARNED_HRS']/data['FIRST_TERM_ATTEMPT_HRS']
data['SECOND_TERM_Hr']=data['SECOND_TERM_EARNED_HRS']/data['SECOND_TERM_ATTEMPT_HRS']

# %% [markdown] id="hcvr7V-56B7v"
# # from the next features we can take only the characters from the feature value So we apply slicinf
# ##  Example  ECON 2105   ==> ECON
#

# %% id="MeuepftH6B7w" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 28, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data['CORE_COURSE_NAME_1_F']=data['CORE_COURSE_NAME_1_F'].str.slice(0,4)  #CHEM 1252  >  CHEM
data['CORE_COURSE_NAME_2_F']=data['CORE_COURSE_NAME_2_F'].str.slice(0,4)
data['CORE_COURSE_NAME_3_F']=data['CORE_COURSE_NAME_3_F'].str.slice(0,4)

data['CORE_COURSE_NAME_1_S']=data['CORE_COURSE_NAME_1_S'].str.slice(0,4)
data['CORE_COURSE_NAME_2_S']=data['CORE_COURSE_NAME_2_S'].str.slice(0,4)
data['CORE_COURSE_NAME_3_S']=data['CORE_COURSE_NAME_3_S'].str.slice(0,4)

# %% [markdown] id="BkLcC6Ot6B7w"
# ## We can drop te next features from the dataset as it's not valuable to the modeling
# ### X contains the training data

# %% id="-UN2WLAO6B7w" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 28, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X=data.drop(['STUDENT IDENTIFIER','CORE_COURSE_NAME_4_F','CORE_COURSE_GRADE_4_F',
             'CORE_COURSE_NAME_5_F','CORE_COURSE_GRADE_5_F','CORE_COURSE_NAME_6_F','CORE_COURSE_GRADE_6_F',
             'CORE_COURSE_NAME_4_S','CORE_COURSE_GRADE_4_S','CORE_COURSE_GRADE_5_S','CORE_COURSE_NAME_5_S',
             'CORE_COURSE_GRADE_6_S','CORE_COURSE_NAME_6_S','RETURNED_2ND_YR','FIRST_TERM_ATTEMPT_HRS',
             'FIRST_TERM_EARNED_HRS','SECOND_TERM_ATTEMPT_HRS','SECOND_TERM_EARNED_HRS','FATHER_HI_EDU_CD',
             'MOTHER_HI_EDU_CD','DEGREE_GROUP_CD','CORE_COURSE_NAME_3_F',
             'CORE_COURSE_GRADE_3_F','CORE_COURSE_NAME_3_S',
             'CORE_COURSE_GRADE_3_S','CORE_COURSE_NAME_4_F',
             'CORE_COURSE_GRADE_4_F','CORE_COURSE_NAME_4_S',
             'CORE_COURSE_GRADE_4_S','CORE_COURSE_NAME_1_F','CORE_COURSE_NAME_2_F','CORE_COURSE_NAME_2_S','CORE_COURSE_NAME_1_S','HIGH_SCHL_NAME'],axis=1)


# %% colab={"base_uri": "https://localhost:8080/"} id="U7y5Os_D6B7x" outputId="c0cc2367-dc6d-4a66-971e-a46c96a5bae1" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 28, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="e04woIOk6B7y" outputId="5de3b5e7-6e13-42be-91a0-4479b825c564" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 25, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X.columns

# %% colab={"base_uri": "https://localhost:8080/", "height": 192} id="WikRwsHC6B7y" outputId="603c6edd-0640-42a1-f85f-c22de1fd0ef3" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 24, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X[:3]


# %% [markdown] id="pfU6zo-C6B7z"
# ## y contain the actual out put of the training data

# %% id="9QnYTheq6B7z" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
y=data['RETURNED_2ND_YR']

# %% colab={"base_uri": "https://localhost:8080/", "height": 962} id="-0PY3XwMvNGw" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="cc9ed023-0e58-4aeb-e257-4067c8e059c3"
# prompt: check for null values in X

X.isnull().sum()


# %% [markdown] id="QgkHXeT26B7z"
# ## Print the first five rows from the training data

# %% id="xJg2k0yc6B70" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 22, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X['STDNT_TEST_ENTRANCE_COMB']=X['STDNT_TEST_ENTRANCE_COMB'].fillna(X['STDNT_TEST_ENTRANCE_COMB'].mean())

# %% id="PXcmbY9x6B71" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 22, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X['SECOND_TERM_Hr']=X['SECOND_TERM_Hr'].fillna(0)

# %% id="SfOdSQsi6B71" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 22, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X['STDNT_TEST_ENTRANCE1']=X['STDNT_TEST_ENTRANCE1'].fillna(X['STDNT_TEST_ENTRANCE1'].mean())
X['STDNT_TEST_ENTRANCE2']=X['STDNT_TEST_ENTRANCE2'].fillna(X['STDNT_TEST_ENTRANCE2'].mean())
X['DISTANCE_FROM_HOME']=X['DISTANCE_FROM_HOME'].fillna(X['DISTANCE_FROM_HOME'].mean())
X['CORE_COURSE_GRADE_2_F']=X['CORE_COURSE_GRADE_2_F'].fillna(0)
X[['CORE_COURSE_GRADE_1_S','CORE_COURSE_GRADE_2_S']]=X[['CORE_COURSE_GRADE_1_S','CORE_COURSE_GRADE_2_S']].fillna(0)
X['HIGH_SCHL_GPA']=X['HIGH_SCHL_GPA'].fillna(X['HIGH_SCHL_GPA'].mean())


# %% [markdown] id="p61CHAt66B72"
# ## Replace every nan value in the dataset into the mean of the column

# %% colab={"base_uri": "https://localhost:8080/", "height": 273} id="hLaeT6hP6B72" outputId="45b1f362-cc1f-4ae9-adac-cfbce5f650ff" executionInfo={"status": "ok", "timestamp": 1726559303405, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
print(X.shape)
X[:5]

# %% colab={"base_uri": "https://localhost:8080/", "height": 962} id="8JW4Bl1_6B73" outputId="730d2d59-6d8d-44e4-b843-55d9a705733b" executionInfo={"status": "ok", "timestamp": 1726559303716, "user_tz": -180, "elapsed": 330, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X.isna().sum()

# %% [markdown] id="xusGPMd56B73"
# ## Convert the all text in the dataset into numbers with dummies

# %% id="Vsm0PyQnSMSa" executionInfo={"status": "ok", "timestamp": 1726559303721, "user_tz": -180, "elapsed": 334, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Convert only categories aftre search for them by Object  varables in X into dummies

import pandas as pd
X_new = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)

# %% colab={"base_uri": "https://localhost:8080/"} id="1C95NurySfDa" executionInfo={"status": "ok", "timestamp": 1726559303721, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="4ca26f92-d9d3-4fa8-c6ec-346208e87e17"
X_new.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 273} id="Uv8VYLHOTcuG" executionInfo={"status": "ok", "timestamp": 1726559303721, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="bacbca82-9891-4b8d-b4f9-24d2c3a78a5e"
X_new.head()

# %% [markdown] id="bJj3upIO6B74"
# ## feature importance to y  process

# %% colab={"base_uri": "https://localhost:8080/", "height": 432} id="A4vrHxua6B75" outputId="2b07b69b-9271-49b2-f5ca-4c73a5efd1c3" executionInfo={"status": "ok", "timestamp": 1726559307083, "user_tz": -180, "elapsed": 3381, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
#import libraries
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt
#Fit the model
model = RandomForestClassifier()
model.fit(X_new,y)
#plot graph of feature importances
feat_importances = pd.Series(model.feature_importances_, index=X_new.columns)
feat_importances.nlargest(50).plot(kind='barh')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="4zym2DAf6B76" outputId="d6895702-b456-4e53-bad5-5133bd184181" executionInfo={"status": "ok", "timestamp": 1726559307083, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
print(X_new.shape)

# %% id="-U82sM83sKBb" executionInfo={"status": "ok", "timestamp": 1726559307083, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Apply min max scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_new)


# %% colab={"base_uri": "https://localhost:8080/"} id="rFtwh40NqLUO" executionInfo={"status": "ok", "timestamp": 1726559307083, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="408e2645-c81d-4835-824e-ef9ef6b1c920"
# prompt: apply feature selection to choose the best 25 features

from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=50)
X_selected = selector.fit_transform(X_scaled, y)

# Get the names of the selected features
selected_features = X_new.columns[selector.get_support()]
print(selected_features)

# %% id="UcOmG3icyqkY" executionInfo={"status": "ok", "timestamp": 1726559307083, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Create the dataframe using X_selected and selected_features  names

import pandas as pd
X_selected = pd.DataFrame(X_selected, columns=selected_features)

# %% colab={"base_uri": "https://localhost:8080/", "height": 273} id="uHPfNUCty2Bd" executionInfo={"status": "ok", "timestamp": 1726559307083, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="22c6ad57-ce2d-421a-9e6f-5e45c4dd174d"
X_selected.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="_fuoE4eryTDN" executionInfo={"status": "ok", "timestamp": 1726559307083, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="323db812-737e-4df9-9d4c-dce43e9dd246"
X_selected.shape

# %% [markdown] id="uc1dxWx56B78"
# ## split the data into training,  testing

# %% colab={"base_uri": "https://localhost:8080/"} id="pO_EwuVN6B78" outputId="655d1bff-b2f4-4364-bad5-6a6c18f603f1" executionInfo={"status": "ok", "timestamp": 1726559307083, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size = 0.2, random_state = 44,shuffle=True )

print("x train shape is : ",X_train.shape)
print("x test shape is : ",X_test.shape)
print("y train shape is : ",y_train.shape)
print("y test shape is : ",y_test.shape)


# %% colab={"base_uri": "https://localhost:8080/"} id="6auY_QE1UDZs" executionInfo={"status": "ok", "timestamp": 1726558578308, "user_tz": -180, "elapsed": 549, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a17295ce-307b-4587-b788-48b8b234cc95"
# Apply smoten on X_train and y_train

from imblearn.combine import SMOTEENN
from collections import Counter

# Initialize the SMOTEENN object
smote_enn = SMOTEENN(random_state=42)

# Apply the SMOTEENN to balance the dataset
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# Print the class distribution before and after resampling
print("Original class distribution:", Counter(y_train))
print("Resampled class distribution:", Counter(y_resampled))

print("X_train_resampled shape:", X_resampled.shape)
print("y_train_resampled shape:", y_resampled.shape)


# %% id="g4td8yXT9GZy" executionInfo={"status": "ok", "timestamp": 1726559329670, "user_tz": -180, "elapsed": 288, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X_resampled= X_train
y_resampled= y_train

# %% colab={"base_uri": "https://localhost:8080/"} id="sP2-Y9Xkss03" executionInfo={"status": "ok", "timestamp": 1726559354527, "user_tz": -180, "elapsed": 9439, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0c6f40ea-392e-4596-a9de-01e90ad4bfed"
# apply all single  (Logistic regression,  KNN, Naive bayies, SVM, Decision tree), ensemble (RF, Extratrees, bagging m XGBoosting , gradient boosting , adaboost , voting , and stacking) ML models usign grid search

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Single Models
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier()
}

# Hyperparameter Grids (Example - Customize for each model)
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'KNN': {'n_neighbors': [3, 5, 7]},
    'Naive Bayes': {}, # Add an empty dictionary for Naive Bayes since it doesn't usually require hyperparameter tuning
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Decision Tree': {'max_depth': [None, 5, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200]},
    'Extra Trees': {'n_estimators': [50, 100, 200]},
    'Bagging': {'n_estimators': [10, 20, 30]},
    'AdaBoost': {'n_estimators': [50, 100, 200]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
}

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model Evaluation and Selection
best_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring='accuracy')
    grid_search.fit(X_resampled, y_resampled)
    best_models[name] = grid_search.best_estimator_
    print(f"{name}: Best Parameters - {grid_search.best_params_}, Best Score - {grid_search.best_score_}")

# %% colab={"base_uri": "https://localhost:8080/"} id="gb1iNbkC6NAE" executionInfo={"status": "ok", "timestamp": 1726559398162, "user_tz": -180, "elapsed": 43647, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="07563650-af76-46d2-f69a-9e393f77f1a1"
# Ensemble Models
ensemble_models = {
    'Random Forest': RandomForestClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier()
}

for name, model in ensemble_models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring='accuracy')
    grid_search.fit(X_resampled, y_resampled)
    best_models[name] = grid_search.best_estimator_
    print(f"{name}: Best Parameters - {grid_search.best_params_}, Best Score - {grid_search.best_score_}")


# %% colab={"base_uri": "https://localhost:8080/"} id="g6CjBMpS7s4O" executionInfo={"status": "ok", "timestamp": 1726559398162, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b147bd36-17e2-41d9-e69a-b4f1baea3922"
best_models

# %% id="bf5ufopS6tGS" executionInfo={"status": "ok", "timestamp": 1726559399264, "user_tz": -180, "elapsed": 1115, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Voting and Stacking (Example - Customize estimators and voting method)
estimators = [('rf', RandomForestClassifier(n_estimators=200)),
 ('xgb', best_models['XGBoost']), ('bagging', best_models['Bagging'])
 ]
voting_clf = VotingClassifier(estimators=estimators, voting='soft')  # or 'hard'
voting_clf.fit(X_resampled, y_resampled)
best_models['Voting'] = voting_clf

# %% colab={"base_uri": "https://localhost:8080/"} id="NoCW3g948pR_" executionInfo={"status": "ok", "timestamp": 1726559464001, "user_tz": -180, "elapsed": 614, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="16b2a073-5229-4630-a629-bff17c851ee4"
# Evaluate Best Models on Test Set
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

# %% colab={"base_uri": "https://localhost:8080/"} id="jS43NKBA9_Pg" executionInfo={"status": "ok", "timestamp": 1726559717217, "user_tz": -180, "elapsed": 14240, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0b9268b8-cbe9-4b71-ea8d-934e708fd65a"
# Need to implement stacking alogorithm

from sklearn.ensemble import StackingClassifier

# Define base models and meta-learner
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200)),
    ('xgb', best_models['XGBoost']),
    ('bagging', best_models['Bagging']),
]
meta_learner = LogisticRegression()

# Create the stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_learner, cv=5)

# Train the stacking classifier
stacking_clf.fit(X_resampled, y_resampled)

# Add the stacking model to the best_models dictionary
best_models['Stacking'] = stacking_clf

# Evaluate the stacking model on the test set
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
print(f"--- Stacking ---")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

