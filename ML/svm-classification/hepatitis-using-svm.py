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

# %% [markdown] papermill={"duration": 0.035688, "end_time": "2021-03-23T01:54:07.851182", "exception": false, "start_time": "2021-03-23T01:54:07.815494", "status": "completed"} id="MVR5KCZvGKz9"
# **Attribute information:**
#
# 1. **target**: DIE (1), LIVE (2)
# 2. **age**: 10, 20, 30, 40, 50, 60, 70, 80
# 3. **gender**: male (1), female (2)
#
#            ------ no = 2,   yes = 1 ------
#
# 4. **steroid**: no, yes
# 5. **antivirals**: no, yes
# 6. **fatique**: no, yes
# 7. **malaise**: no, yes
# 8. **anorexia**: no, yes
# 9. **liverBig**: no, yes
# 10. **liverFirm**: no, yes
# 11. **spleen**: no, yes
# 12. **spiders**: no, yes
# 13. **ascites**: no, yes
# 14. **varices**: no, yes
# 15. **histology**: no, yes
#
#
# 16. **bilirubin**: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00 --
# 17. **alk**: 33, 80, 120, 160, 200, 250 ---
# 18. **sgot**: 13, 100, 200, 300, 400, 500, ---
# 19. **albu**: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0, ---
# 20. **protime**: 10, 20, 30, 40, 50, 60, 70, 80, 90, ---
#
#         NA's are represented with "?"

# %% [markdown] papermill={"duration": 0.032359, "end_time": "2021-03-23T01:54:07.916534", "exception": false, "start_time": "2021-03-23T01:54:07.884175", "status": "completed"} id="s3yYF0uFGKz_"
# ## Dataset Reading and Pre-Processing steps

# %% [markdown] papermill={"duration": 0.032665, "end_time": "2021-03-23T01:54:07.981840", "exception": false, "start_time": "2021-03-23T01:54:07.949175", "status": "completed"} id="PatFKUD5GKz_"
# import required libraries

# %% papermill={"duration": 0.807829, "end_time": "2021-03-23T01:54:08.822879", "exception": false, "start_time": "2021-03-23T01:54:08.015050", "status": "completed"} id="iqYguBfeGKz_" executionInfo={"status": "ok", "timestamp": 1725612005291, "user_tz": -180, "elapsed": 326, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# %% papermill={"duration": 0.039963, "end_time": "2021-03-23T01:54:08.896509", "exception": false, "start_time": "2021-03-23T01:54:08.856546", "status": "completed"} id="ZNq74f8UGK0A" executionInfo={"status": "ok", "timestamp": 1725612005585, "user_tz": -180, "elapsed": 3, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
#Code to ignore warnings
import warnings
warnings.filterwarnings("ignore")

# %% colab={"base_uri": "https://localhost:8080/"} id="aeGQPiNWInC9" executionInfo={"status": "ok", "timestamp": 1725612007421, "user_tz": -180, "elapsed": 1838, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="78176cc2-52b3-4d64-aeb7-a01ea87f6840"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="rdrp3Y8BInwT" executionInfo={"status": "ok", "timestamp": 1725612007421, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9426a931-eb8e-4f3a-b9ee-dc04dd9ca2df"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/SVM

# %% [markdown] papermill={"duration": 0.032669, "end_time": "2021-03-23T01:54:08.962033", "exception": false, "start_time": "2021-03-23T01:54:08.929364", "status": "completed"} id="cSCxc-86GK0A"
# ###### 1. Read the HEPATITIS dataset and check the data shapes

# %% papermill={"duration": 0.051653, "end_time": "2021-03-23T01:54:09.046619", "exception": false, "start_time": "2021-03-23T01:54:08.994966", "status": "completed"} id="ocpq53FtGK0A" executionInfo={"status": "ok", "timestamp": 1725612007421, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
## Read "hepatitis.csv" using pandas
# target =  1: Die; 2: Live
data = pd.read_csv("hepatitis.csv", na_values="?")

# %% papermill={"duration": 0.043763, "end_time": "2021-03-23T01:54:09.123553", "exception": false, "start_time": "2021-03-23T01:54:09.079790", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="Aq5ltJJZGK0A" executionInfo={"status": "ok", "timestamp": 1725612007421, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="88594b7e-8ee8-4f06-84ff-5a720d20933c"
data.shape

# %% papermill={"duration": 0.068924, "end_time": "2021-03-23T01:54:09.225985", "exception": false, "start_time": "2021-03-23T01:54:09.157061", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 412} id="PjdW3j3ZGK0B" executionInfo={"status": "ok", "timestamp": 1725612007421, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="98815ca0-da16-4737-8f5d-42da4e0b3286"
data.head(10)

# %% [markdown] papermill={"duration": 0.034303, "end_time": "2021-03-23T01:54:09.294534", "exception": false, "start_time": "2021-03-23T01:54:09.260231", "status": "completed"} id="SxQHn87AGK0B"
# ###### 2. Check basic summary statistics of the data

# %% papermill={"duration": 0.096756, "end_time": "2021-03-23T01:54:09.425807", "exception": false, "start_time": "2021-03-23T01:54:09.329051", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 349} id="Nnnphv-EGK0B" executionInfo={"status": "ok", "timestamp": 1725612007421, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="bf3d985d-c6f0-478d-8fe8-fcd22c378074"
data.describe()

# %% [markdown] papermill={"duration": 0.034535, "end_time": "2021-03-23T01:54:09.495693", "exception": false, "start_time": "2021-03-23T01:54:09.461158", "status": "completed"} id="Pv6I0Vb4GK0B"
# ###### 3. Check for value counts in target variable

# %% papermill={"duration": 0.045879, "end_time": "2021-03-23T01:54:09.576253", "exception": false, "start_time": "2021-03-23T01:54:09.530374", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 178} id="oTIEsK_kGK0B" executionInfo={"status": "ok", "timestamp": 1725612007421, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b6d6980f-eb44-4097-cb61-f22a1d40996e"
data.target.value_counts()

# %% [markdown] papermill={"duration": 0.034928, "end_time": "2021-03-23T01:54:09.646147", "exception": false, "start_time": "2021-03-23T01:54:09.611219", "status": "completed"} id="LswjnGRWGK0B"
# #### 4. Check the datatype of each variable

# %% papermill={"duration": 0.044248, "end_time": "2021-03-23T01:54:09.725997", "exception": false, "start_time": "2021-03-23T01:54:09.681749", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 742} id="8zxgw0xwGK0C" executionInfo={"status": "ok", "timestamp": 1725612007758, "user_tz": -180, "elapsed": 350, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="876f67a1-0d09-4c1a-b4b9-dc3c2dd8849a"
data.dtypes

# %% colab={"base_uri": "https://localhost:8080/"} id="bGqsU0_GceAu" executionInfo={"status": "ok", "timestamp": 1725612007758, "user_tz": -180, "elapsed": 34, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="476f287f-ff3a-411f-acc5-55816ee53a86"
data.info()

# %% papermill={"duration": 0.059401, "end_time": "2021-03-23T01:54:09.821013", "exception": false, "start_time": "2021-03-23T01:54:09.761612", "status": "completed"} id="agvg2QbAGK0C" executionInfo={"status": "ok", "timestamp": 1725612007758, "user_tz": -180, "elapsed": 31, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
cat_cols = data.columns[data.nunique() < 5]

# %% colab={"base_uri": "https://localhost:8080/"} id="fqMGBn1RJBQA" executionInfo={"status": "ok", "timestamp": 1725612007759, "user_tz": -180, "elapsed": 31, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="7433e99d-f0d5-4d38-8ab9-0bfa1e945500"
cat_cols

# %% papermill={"duration": 0.045868, "end_time": "2021-03-23T01:54:09.902560", "exception": false, "start_time": "2021-03-23T01:54:09.856692", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="pHXbLKAvGK0C" executionInfo={"status": "ok", "timestamp": 1725612007759, "user_tz": -180, "elapsed": 25, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="42cf0509-69a5-4cc5-e1e2-15b89ca7d875"
num_cols = data.columns[data.nunique() >= 5]
num_cols

# %% [markdown] papermill={"duration": 0.035429, "end_time": "2021-03-23T01:54:09.973727", "exception": false, "start_time": "2021-03-23T01:54:09.938298", "status": "completed"} id="4OmAPsj1GK0C"
# #### 5. Drop columns which are not significant

# %% papermill={"duration": 0.048224, "end_time": "2021-03-23T01:54:10.059160", "exception": false, "start_time": "2021-03-23T01:54:10.010936", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="I7VfZr8DGK0C" executionInfo={"status": "ok", "timestamp": 1725612007759, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="964f8810-0d0b-4116-8ce4-96fc604da362"
data.drop(["ID"], axis = 1, inplace=True)
num_cols = data.columns[data.nunique() >= 5]
num_cols

# %% papermill={"duration": 0.05788, "end_time": "2021-03-23T01:54:10.152881", "exception": false, "start_time": "2021-03-23T01:54:10.095001", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 226} id="PY5efqMGGK0C" executionInfo={"status": "ok", "timestamp": 1725612008093, "user_tz": -180, "elapsed": 348, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="74019cd4-9dee-46f9-cb6b-f07162128e12"
data.head()

# %% [markdown] papermill={"duration": 0.036463, "end_time": "2021-03-23T01:54:10.225853", "exception": false, "start_time": "2021-03-23T01:54:10.189390", "status": "completed"} id="rBvmgvt0GK0C"
# #### 6. Identify the Categorical Columns and store them in a variable cat_cols and numerical into num_cols

# %% papermill={"duration": 0.044189, "end_time": "2021-03-23T01:54:10.315257", "exception": false, "start_time": "2021-03-23T01:54:10.271068", "status": "completed"} id="gmu28Q_OGK0C" executionInfo={"status": "ok", "timestamp": 1725612008093, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
num_cols = ["age", "bili", "alk", "sgot", "albu", "protime"]
cat_cols = ['gender', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liverBig',
            'liverFirm', 'spleen', 'spiders', 'ascites', 'varices', 'histology']

# %% [markdown] papermill={"duration": 0.036554, "end_time": "2021-03-23T01:54:10.388446", "exception": false, "start_time": "2021-03-23T01:54:10.351892", "status": "completed"} id="a4uO7p04GK0C"
# #### 7. Checking the null values

# %% papermill={"duration": 0.047513, "end_time": "2021-03-23T01:54:10.472680", "exception": false, "start_time": "2021-03-23T01:54:10.425167", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 711} id="SNOt9oxdGK0C" executionInfo={"status": "ok", "timestamp": 1725612008093, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d2537732-8d4c-4ec5-a750-b0277407717c"
data.isna().sum()

# %% papermill={"duration": 0.046681, "end_time": "2021-03-23T01:54:10.556229", "exception": false, "start_time": "2021-03-23T01:54:10.509548", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 711} id="-WQY84nBGK0C" executionInfo={"status": "ok", "timestamp": 1725612008093, "user_tz": -180, "elapsed": 22, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ca691c50-a4ca-48bd-a4b3-6e93127cec1e"
data.isnull().sum()

# %% [markdown] papermill={"duration": 0.038211, "end_time": "2021-03-23T01:54:10.631681", "exception": false, "start_time": "2021-03-23T01:54:10.593470", "status": "completed"} id="J91SVPmBGK0C"
# #### 8. Split the data into X and y

# %% papermill={"duration": 0.05205, "end_time": "2021-03-23T01:54:10.728050", "exception": false, "start_time": "2021-03-23T01:54:10.676000", "status": "completed"} id="1XHXtpPAGK0D" executionInfo={"status": "ok", "timestamp": 1725612008094, "user_tz": -180, "elapsed": 22, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X = data.drop(["target"], axis = 1)

# %% papermill={"duration": 0.05007, "end_time": "2021-03-23T01:54:10.821365", "exception": false, "start_time": "2021-03-23T01:54:10.771295", "status": "completed"} id="fDfd9E3XGK0D" executionInfo={"status": "ok", "timestamp": 1725612008094, "user_tz": -180, "elapsed": 22, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
y = data["target"]

# %% papermill={"duration": 0.052828, "end_time": "2021-03-23T01:54:10.917527", "exception": false, "start_time": "2021-03-23T01:54:10.864699", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="2WRCis7EGK0D" executionInfo={"status": "ok", "timestamp": 1725612008094, "user_tz": -180, "elapsed": 22, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="70f99da2-42d5-462e-bb80-38605e1edbc9"
print(X.shape, y.shape)

# %% [markdown] papermill={"duration": 0.041616, "end_time": "2021-03-23T01:54:11.002163", "exception": false, "start_time": "2021-03-23T01:54:10.960547", "status": "completed"} id="ounhQigLGK0D"
# #### 9. Split the data into X_train, X_test, y_train, y_test with test_size = 0.20 using sklearn

# %% papermill={"duration": 0.050267, "end_time": "2021-03-23T01:54:11.093477", "exception": false, "start_time": "2021-03-23T01:54:11.043210", "status": "completed"} id="gFFi5c4xGK0D" executionInfo={"status": "ok", "timestamp": 1725612008094, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

# %% papermill={"duration": 0.049446, "end_time": "2021-03-23T01:54:11.184336", "exception": false, "start_time": "2021-03-23T01:54:11.134890", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="ANUBhFerGK0D" executionInfo={"status": "ok", "timestamp": 1725612008392, "user_tz": -180, "elapsed": 317, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="869a6f90-80fb-4759-a936-3c514667fdf7"
## Print the shape of X_train, X_test, y_train, y_test
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %% [markdown] papermill={"duration": 0.045168, "end_time": "2021-03-23T01:54:11.270239", "exception": false, "start_time": "2021-03-23T01:54:11.225071", "status": "completed"} id="ntmQjif3GK0D"
# #### 10. Check null values in train and test, check value_counts in y_train and y_test

# %% papermill={"duration": 0.050181, "end_time": "2021-03-23T01:54:11.361397", "exception": false, "start_time": "2021-03-23T01:54:11.311216", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="IcG8fHDVGK0D" executionInfo={"status": "ok", "timestamp": 1725612008392, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="08b401b0-94a6-4ee3-f20a-7a78d845afba"
print(y_train.value_counts()/X_train.shape[0])

# %% papermill={"duration": 0.052668, "end_time": "2021-03-23T01:54:11.459968", "exception": false, "start_time": "2021-03-23T01:54:11.407300", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="HW3y4L9hGK0D" executionInfo={"status": "ok", "timestamp": 1725612008392, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="2bfc7fca-8e34-41c8-8dcc-06217e172871"
print(y_test.value_counts()/X_test.shape[0])

# %% papermill={"duration": 0.05497, "end_time": "2021-03-23T01:54:11.559730", "exception": false, "start_time": "2021-03-23T01:54:11.504760", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 680} id="IG8S0iGrGK0D" executionInfo={"status": "ok", "timestamp": 1725612008392, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="3e820bde-b7fb-4ae7-c11f-d57bd66e0c17"
# null values in train
X_train.isna().sum()

# %% papermill={"duration": 0.050251, "end_time": "2021-03-23T01:54:11.652223", "exception": false, "start_time": "2021-03-23T01:54:11.601972", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 680} id="MOecqxo3GK0E" executionInfo={"status": "ok", "timestamp": 1725612008694, "user_tz": -180, "elapsed": 314, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="77e6f9a4-20e8-4063-d57b-2911d3a54114"
# null values in test
X_test.isna().sum()

# %% [markdown] papermill={"duration": 0.040544, "end_time": "2021-03-23T01:54:11.733191", "exception": false, "start_time": "2021-03-23T01:54:11.692647", "status": "completed"} id="kKalpIxmGK0E"
# #### 11. Impute the Categorical Columns with mode and Numerical columns with mean

# %% papermill={"duration": 0.049567, "end_time": "2021-03-23T01:54:11.823842", "exception": false, "start_time": "2021-03-23T01:54:11.774275", "status": "completed"} id="vGquajOWGK0E" executionInfo={"status": "ok", "timestamp": 1725612008694, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df_cat_train = X_train[cat_cols]
df_cat_test = X_test[cat_cols]

# %% papermill={"duration": 0.050171, "end_time": "2021-03-23T01:54:11.917601", "exception": false, "start_time": "2021-03-23T01:54:11.867430", "status": "completed"} id="gPKN_2QzGK0H" executionInfo={"status": "ok", "timestamp": 1725612008694, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Impute on train
# df_cat_train = df_cat_train.fillna(df_cat_train.mode().iloc[0])

# Impute on test
# df_cat_test = df_cat_test.fillna(df_cat_train.mode().iloc[0])

# %% papermill={"duration": 0.258197, "end_time": "2021-03-23T01:54:12.219186", "exception": false, "start_time": "2021-03-23T01:54:11.960989", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 74} id="g5v9kHMPGK0H" executionInfo={"status": "ok", "timestamp": 1725612008694, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="42d2eabd-c726-4098-c735-aff83d0684e0"
from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_imputer.fit(df_cat_train)

# %% papermill={"duration": 0.090542, "end_time": "2021-03-23T01:54:12.363641", "exception": false, "start_time": "2021-03-23T01:54:12.273099", "status": "completed"} id="7XPnKVLhGK0H" executionInfo={"status": "ok", "timestamp": 1725612008694, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df_cat_train = pd.DataFrame(cat_imputer.transform(df_cat_train), columns=cat_cols)

# %% papermill={"duration": 0.096446, "end_time": "2021-03-23T01:54:12.531683", "exception": false, "start_time": "2021-03-23T01:54:12.435237", "status": "completed"} id="dKqOmX8AGK0H" executionInfo={"status": "ok", "timestamp": 1725612009148, "user_tz": -180, "elapsed": 468, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df_cat_test = pd.DataFrame(cat_imputer.transform(df_cat_test), columns=cat_cols)

# %% papermill={"duration": 0.079335, "end_time": "2021-03-23T01:54:12.686471", "exception": false, "start_time": "2021-03-23T01:54:12.607136", "status": "completed"} id="d6EsTXvqGK0H" executionInfo={"status": "ok", "timestamp": 1725612009148, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df_num_train = X_train[num_cols]
df_num_test = X_test[num_cols]

# %% papermill={"duration": 0.080102, "end_time": "2021-03-23T01:54:12.834567", "exception": false, "start_time": "2021-03-23T01:54:12.754465", "status": "completed"} id="ypPIKskhGK0H" executionInfo={"status": "ok", "timestamp": 1725612009148, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Impute on train
# df_num_train = df_num_train.fillna(df_num_train.mean())

#Impute on test
# df_num_test = df_num_test.fillna(df_num_train.mean())

# %% papermill={"duration": 0.10783, "end_time": "2021-03-23T01:54:13.035711", "exception": false, "start_time": "2021-03-23T01:54:12.927881", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 74} id="JrY7FcNHGK0H" executionInfo={"status": "ok", "timestamp": 1725612009148, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f6ac2f9a-618e-408a-82fb-b39f25d6b1c4"
num_imputer = SimpleImputer(strategy='median')
num_imputer.fit(df_num_train[num_cols])

# %% papermill={"duration": 0.054928, "end_time": "2021-03-23T01:54:13.152700", "exception": false, "start_time": "2021-03-23T01:54:13.097772", "status": "completed"} id="9R-NmOUqGK0H" executionInfo={"status": "ok", "timestamp": 1725612009518, "user_tz": -180, "elapsed": 389, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df_num_train = pd.DataFrame ( num_imputer.transform(df_num_train), columns= num_cols)

# %% papermill={"duration": 0.073313, "end_time": "2021-03-23T01:54:13.285126", "exception": false, "start_time": "2021-03-23T01:54:13.211813", "status": "completed"} id="zsJRuoyeGK0I" executionInfo={"status": "ok", "timestamp": 1725612009518, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df_num_test =  pd.DataFrame(num_imputer.transform(df_num_test), columns=num_cols)

# %% papermill={"duration": 0.083574, "end_time": "2021-03-23T01:54:13.431786", "exception": false, "start_time": "2021-03-23T01:54:13.348212", "status": "completed"} id="bUYbnhxiGK0I" executionInfo={"status": "ok", "timestamp": 1725612009518, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Combine numeric and categorical in train
X_train = pd.concat([df_num_train, df_cat_train], axis = 1)

# Combine numeric and categorical in test
X_test = pd.concat([df_num_test, df_cat_test], axis = 1)

# %% papermill={"duration": 0.05184, "end_time": "2021-03-23T01:54:13.541692", "exception": false, "start_time": "2021-03-23T01:54:13.489852", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 680} id="GLCzm7LrGK0I" executionInfo={"status": "ok", "timestamp": 1725612009518, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="1d683320-fb0e-4081-9efa-617ffe5393cf"
X_train.isna().sum()

# %% papermill={"duration": 0.051688, "end_time": "2021-03-23T01:54:13.635582", "exception": false, "start_time": "2021-03-23T01:54:13.583894", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 680} id="En3VGAVyGK0I" executionInfo={"status": "ok", "timestamp": 1725612009518, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c4d556cd-2b35-4a26-c345-7dc921d49223"
X_test.isna().sum()

# %% [markdown] papermill={"duration": 0.04269, "end_time": "2021-03-23T01:54:13.720906", "exception": false, "start_time": "2021-03-23T01:54:13.678216", "status": "completed"} id="O6FntweyGK0I"
# #### Convert all the categorical columns to Integer Format before dummification (2.0 as 2 etc.)

# %% papermill={"duration": 0.059586, "end_time": "2021-03-23T01:54:13.822988", "exception": false, "start_time": "2021-03-23T01:54:13.763402", "status": "completed"} id="ce0UwZrkGK0I" executionInfo={"status": "ok", "timestamp": 1725612009518, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Train
X_train[cat_cols] = X_train[cat_cols].astype('int')

# Test
X_test[cat_cols] = X_test[cat_cols].astype('int')

# %% colab={"base_uri": "https://localhost:8080/", "height": 226} id="9dSkrgSQelKU" executionInfo={"status": "ok", "timestamp": 1725612009518, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8662e123-0f46-4e87-b571-2a1081092113"
X_train.head()

# %% [markdown] papermill={"duration": 0.042444, "end_time": "2021-03-23T01:54:13.908132", "exception": false, "start_time": "2021-03-23T01:54:13.865688", "status": "completed"} id="k-1eDqLSGK0I"
# #### 12. Dummify the Categorical columns

# %% papermill={"duration": 0.068831, "end_time": "2021-03-23T01:54:14.019728", "exception": false, "start_time": "2021-03-23T01:54:13.950897", "status": "completed"} id="R37mwNGBGK0I" executionInfo={"status": "ok", "timestamp": 1725612009842, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
## Convert Categorical Columns to Dummies
# Train
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)

# Test
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# %% papermill={"duration": 0.050304, "end_time": "2021-03-23T01:54:14.113449", "exception": false, "start_time": "2021-03-23T01:54:14.063145", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="yzZlRJroGK0I" executionInfo={"status": "ok", "timestamp": 1725612009843, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="5cf03490-61ea-4ceb-e6ab-887e748746a9"
X_train.columns

# %% papermill={"duration": 0.051026, "end_time": "2021-03-23T01:54:14.208202", "exception": false, "start_time": "2021-03-23T01:54:14.157176", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="dhmntuqyGK0I" executionInfo={"status": "ok", "timestamp": 1725612009843, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ed17f66e-e750-4a5e-9d2f-f5eb6433f364"
X_test.columns

# %% [markdown] papermill={"duration": 0.043075, "end_time": "2021-03-23T01:54:14.294924", "exception": false, "start_time": "2021-03-23T01:54:14.251849", "status": "completed"} id="ZQUSAGkcGK0J"
# #### 13. Scale the numeric attributes ["age", "bili", "alk", "sgot", "albu", "protime"]

# %% papermill={"duration": 0.049738, "end_time": "2021-03-23T01:54:14.388795", "exception": false, "start_time": "2021-03-23T01:54:14.339057", "status": "completed"} id="M9x11UfzGK0J" executionInfo={"status": "ok", "timestamp": 1725612009843, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.preprocessing import StandardScaler

# %% colab={"base_uri": "https://localhost:8080/", "height": 226} id="mOqPB-SFfKin" executionInfo={"status": "ok", "timestamp": 1725612009843, "user_tz": -180, "elapsed": 13, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="67f900e3-9e07-4c61-a4df-1a515ffce345"
X_train.head()

# %% papermill={"duration": 0.061592, "end_time": "2021-03-23T01:54:14.494019", "exception": false, "start_time": "2021-03-23T01:54:14.432427", "status": "completed"} id="iBTtkG36GK0J" executionInfo={"status": "ok", "timestamp": 1725612010165, "user_tz": -180, "elapsed": 1, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
#num_cols = ["age", "bili", "alk", "sgot", "albu", "protime"]
scaler = StandardScaler()

scaler.fit(X_train.loc[:,num_cols])

# scale on train
X_train.loc[:,num_cols] = scaler.transform(X_train.loc[:,num_cols])
#X_train[num_cols] = scaler.transform(X_train[num_cols])

# scale on test
X_test.loc[:,num_cols] = scaler.transform(X_test.loc[:,num_cols])

# %% colab={"base_uri": "https://localhost:8080/", "height": 383} id="49KYMvYyfaWF" executionInfo={"status": "ok", "timestamp": 1725612021896, "user_tz": -180, "elapsed": 325, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d359160f-9e11-44b7-ff66-d94b976d4485"
X_train[:10]

# %% colab={"base_uri": "https://localhost:8080/", "height": 383} id="pZ-cwrNWfkHy" executionInfo={"status": "ok", "timestamp": 1725612056024, "user_tz": -180, "elapsed": 305, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d21d4759-216a-4ecd-a753-51a32c4df56a"
X_test[:10]

# %% [markdown] papermill={"duration": 0.043471, "end_time": "2021-03-23T01:54:14.581447", "exception": false, "start_time": "2021-03-23T01:54:14.537976", "status": "completed"} id="mmDFiEeIGK0J"
# ## MODEL BUILDING - SVM

# %% papermill={"duration": 0.049876, "end_time": "2021-03-23T01:54:14.675069", "exception": false, "start_time": "2021-03-23T01:54:14.625193", "status": "completed"} id="bpgmblgfGK0J" executionInfo={"status": "ok", "timestamp": 1725612065837, "user_tz": -180, "elapsed": 299, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.svm import SVC

# %% papermill={"duration": 0.050388, "end_time": "2021-03-23T01:54:14.770530", "exception": false, "start_time": "2021-03-23T01:54:14.720142", "status": "completed"} id="SGk6m4fJGK0J" executionInfo={"status": "ok", "timestamp": 1725612067860, "user_tz": -180, "elapsed": 2, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Create a SVC classifier using a linear kernel
linear_svm = SVC(kernel='linear', C=1, random_state=0)

# %% papermill={"duration": 0.056121, "end_time": "2021-03-23T01:54:14.871019", "exception": false, "start_time": "2021-03-23T01:54:14.814898", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 74} id="Bwi0IF-4GK0J" executionInfo={"status": "ok", "timestamp": 1725612070925, "user_tz": -180, "elapsed": 396, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="bbf76eeb-59af-4dda-df4c-e2c41edc36d1"
# Train the classifier
linear_svm.fit(X=X_train, y= y_train)

# %% papermill={"duration": 0.072494, "end_time": "2021-03-23T01:54:14.987651", "exception": false, "start_time": "2021-03-23T01:54:14.915157", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="67l2wwqNGK0J" executionInfo={"status": "ok", "timestamp": 1725612086333, "user_tz": -180, "elapsed": 315, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="861a50cb-69ae-49ef-dfab-217a38f1fd14"
## Predict
train_predictions = linear_svm.predict(X_train)
test_predictions = linear_svm.predict(X_test)

### Train data accuracy
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

print("TRAIN Conf Matrix : \n", confusion_matrix(y_train, train_predictions))
print("\nTRAIN DATA ACCURACY",accuracy_score(y_train,train_predictions))
print("\nTrain data f1-score for class '1'",f1_score(y_train,train_predictions,pos_label=1))
print("\nTrain data f1-score for class '2'",f1_score(y_train,train_predictions,pos_label=2))

### Test data accuracy
print("\n\n--------------------------------------\n\n")

print("TEST Conf Matrix : \n", confusion_matrix(y_test, test_predictions))
print("\nTEST DATA ACCURACY",accuracy_score(y_test,test_predictions))
print("\nTest data f1-score for class '1'",f1_score(y_test,test_predictions,pos_label=1))
print("\nTest data f1-score for class '2'",f1_score(y_test,test_predictions,pos_label=2))

# %% [markdown] papermill={"duration": 0.045787, "end_time": "2021-03-23T01:54:15.078788", "exception": false, "start_time": "2021-03-23T01:54:15.033001", "status": "completed"} id="ZFz4XOMYGK0J"
# ####  Non Linear SVM (RBF)

# %% [markdown] papermill={"duration": 0.044588, "end_time": "2021-03-23T01:54:15.168552", "exception": false, "start_time": "2021-03-23T01:54:15.123964", "status": "completed"} id="rayAPu5KGK0J"
# Radial Basis Function is a commonly used kernel in SVC:<br>
#
# <img src="rbf_kernel.png">
#
# where <math xmlns="http://www.w3.org/1998/Math/MathML">
#   <mrow class="MJX-TeXAtom-ORD">
#     <mo stretchy="false">|</mo>
#   </mrow>
#   <mrow class="MJX-TeXAtom-ORD">
#     <mo stretchy="false">|</mo>
#   </mrow>
#   <mrow class="MJX-TeXAtom-ORD">
#     <mi mathvariant="bold">x</mi>
#       <sub>i</sub>
#   </mrow>
#   <mo>&#x2212;<!-- − --></mo>
#   <mrow class="MJX-TeXAtom-ORD">
#     <msup>
#       <mi mathvariant="bold">x</mi>
#       <sub>j</sub>
#     </msup>
#   </mrow>
#   <mrow class="MJX-TeXAtom-ORD">
#     <mo stretchy="false">|</mo>
#   </mrow>
#   <msup>
#     <mrow class="MJX-TeXAtom-ORD">
#       <mo stretchy="false">|</mo>
#     </mrow>
#     <mrow class="MJX-TeXAtom-ORD">
#       <sup>2</sup>
#     </mrow>
#   </msup>
# </math>  is the squared Euclidean distance between two data points x<sub>i</sub> and x<sub>j</sub>
#
# It is only important to know that an SVC classifier using an RBF kernel has two parameters: gamma and C.
#
# <strong>Gamma:</strong>
#
# - Gamma is a parameter of the RBF kernel and can be thought of as the ‘spread’ of the kernel and therefore the decision region. When gamma is low, the ‘curve’ of the decision boundary is very low and thus the decision region is very broad. When gamma is high, the ‘curve’ of the decision boundary is high, which creates islands of decision-boundaries around data points.
#
# <strong>C:</strong>
#
# - C is a parameter of the SVC learner and is the penalty for misclassifying a data point. When C is small, the classifier is okay with misclassified data points (high bias, low variance). When C is large, the classifier is heavily penalized for misclassified data and therefore bends over backwards avoid any misclassified data points (low bias, high variance).
#
#
# <strong>Kernel Trick:</strong><br>
# Image you have a two-dimensional non-linearly separable dataset, you would like to classify it using SVM. It looks like not possible because the data is not linearly separable. However, if we transform the two-dimensional data to a higher dimension, say, three-dimension or even ten-dimension, we would be able to find a hyperplane to separate the data.
#
# <img src="kernel_trick.png">
#
# The problem is, if we have a large dataset containing, say, millions of examples, the transformation will take a long time to run.<br>
# To solve this problem, we actually only care about the result of the dot product (x<sub>i</sub>.x<sub>j</sub>)<br>
# <br>If there is a function which could calculate the dot product and the result is the same as when we transform the data into higher dimension, it would be fantastic. This function is called a kernel function.<br>
# <br>In essence, what the kernel trick does for us is to offer a more efficient and less expensive way to transform data into higher dimensions.

# %% papermill={"duration": 0.053716, "end_time": "2021-03-23T01:54:15.267305", "exception": false, "start_time": "2021-03-23T01:54:15.213589", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 74} id="larmAFPWGK0K" executionInfo={"status": "ok", "timestamp": 1725612127660, "user_tz": -180, "elapsed": 296, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="cb9f4d30-33c8-44ea-ef3a-e2224d4cf08f"
## Create an SVC object and print it to see the arguments
svc = SVC(kernel='rbf', random_state=0, gamma=0.01, C=1)
svc

# %% papermill={"duration": 0.057692, "end_time": "2021-03-23T01:54:15.370412", "exception": false, "start_time": "2021-03-23T01:54:15.312720", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 74} id="41hxAsZ-GK0K" executionInfo={"status": "ok", "timestamp": 1725612130229, "user_tz": -180, "elapsed": 733, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d2c0fe12-45d4-4464-b791-8385a1a74fa5"
## Train the model
svc.fit(X=X_train, y= y_train)

# %% papermill={"duration": 0.07239, "end_time": "2021-03-23T01:54:15.488094", "exception": false, "start_time": "2021-03-23T01:54:15.415704", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="GbmUC7apGK0K" executionInfo={"status": "ok", "timestamp": 1725612131828, "user_tz": -180, "elapsed": 7, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b10c0a5e-74f5-4320-8fd7-6e055f63358b"
## Predict
train_predictions = svc.predict(X_train)
test_predictions = svc.predict(X_test)

### Train data accuracy

print("TRAIN Conf Matrix : \n", confusion_matrix(y_train, train_predictions))
print("\nTRAIN DATA ACCURACY",accuracy_score(y_train,train_predictions))
print("\nTrain data f1-score for class '1'",f1_score(y_train,train_predictions,pos_label=1))
print("\nTrain data f1-score for class '2'",f1_score(y_train,train_predictions,pos_label=2))

### Test data accuracy
print("\n\n--------------------------------------\n\n")

print("TEST Conf Matrix : \n", confusion_matrix(y_test, test_predictions))
print("\nTEST DATA ACCURACY",accuracy_score(y_test,test_predictions))
print("\nTest data f1-score for class '1'",f1_score(y_test,test_predictions,pos_label=1))
print("\nTest data f1-score for class '2'",f1_score(y_test,test_predictions,pos_label=2))

# %% [markdown] papermill={"duration": 0.045684, "end_time": "2021-03-23T01:54:15.580370", "exception": false, "start_time": "2021-03-23T01:54:15.534686", "status": "completed"} id="yeGDNa4cGK0K"
# ### SVM with Grid Search for Paramater Tuning

# %% papermill={"duration": 0.053999, "end_time": "2021-03-23T01:54:15.680075", "exception": false, "start_time": "2021-03-23T01:54:15.626076", "status": "completed"} id="aX5Q-r-9GK0K" executionInfo={"status": "ok", "timestamp": 1725612155649, "user_tz": -180, "elapsed": 317, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
## Use Grid Search for parameter tuning

from sklearn.model_selection import GridSearchCV

svc_grid = SVC()

param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 5, 3, 0.02 ],
                'gamma': [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'kernel':['linear', 'rbf' , 'guassian','poly'],
             }

svc_cv_grid = GridSearchCV(estimator = svc_grid, param_grid = param_grid, cv = 5, verbose=1)

# %% papermill={"duration": 5.221163, "end_time": "2021-03-23T01:54:20.947157", "exception": false, "start_time": "2021-03-23T01:54:15.725994", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 135} id="IcB8EpbhGK0K" executionInfo={"status": "ok", "timestamp": 1725612168098, "user_tz": -180, "elapsed": 8381, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="894634dd-49c8-48bc-dc80-9c27ddf16c38"
## Fit the grid search model
svc_cv_grid.fit(X=X_train, y=y_train)

# %% papermill={"duration": 0.063351, "end_time": "2021-03-23T01:54:21.067359", "exception": false, "start_time": "2021-03-23T01:54:21.004008", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="yAwJlknLGK0K" executionInfo={"status": "ok", "timestamp": 1725612168098, "user_tz": -180, "elapsed": 5, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="4c9da6b6-d625-456d-c5c8-bf641e6af18f"
# Get the best parameters
svc_cv_grid.best_params_

# %% papermill={"duration": 0.062049, "end_time": "2021-03-23T01:54:21.185312", "exception": false, "start_time": "2021-03-23T01:54:21.123263", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 74} id="SQKQX7rdGK0L" executionInfo={"status": "ok", "timestamp": 1725612169681, "user_tz": -180, "elapsed": 11, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e5cda130-5f8a-492d-ec87-48034236d265"
svc_best = svc_cv_grid.best_estimator_
svc_best

# %% papermill={"duration": 0.078943, "end_time": "2021-03-23T01:54:21.320156", "exception": false, "start_time": "2021-03-23T01:54:21.241213", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="qSOUwxzrGK0L" executionInfo={"status": "ok", "timestamp": 1725612174541, "user_tz": -180, "elapsed": 283, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="62ea7572-b359-4355-91be-2422cde59f7d"
## Predict
train_predictions = svc_best.predict(X_train)
test_predictions = svc_best.predict(X_test)

print("TRAIN DATA ACCURACY",accuracy_score(y_train,train_predictions))
print("\nTrain data f1-score for class '1'",f1_score(y_train,train_predictions,pos_label=1))
print("\nTrain data f1-score for class '2'",f1_score(y_train,train_predictions,pos_label=2))

### Test data accuracy
print("\n\n--------------------------------------\n\n")
print("TEST DATA ACCURACY",accuracy_score(y_test,test_predictions))
print("\nTest data f1-score for class '1'",f1_score(y_test,test_predictions,pos_label=1))
print("\nTest data f1-score for class '2'",f1_score(y_test,test_predictions,pos_label=2))

# %% papermill={"duration": 0.05593, "end_time": "2021-03-23T01:54:21.433598", "exception": false, "start_time": "2021-03-23T01:54:21.377668", "status": "completed"} id="A9VgtNQoGK0L"
