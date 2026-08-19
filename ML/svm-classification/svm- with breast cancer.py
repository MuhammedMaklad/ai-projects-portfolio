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

# %% [markdown] papermill={"duration": 0.01399, "end_time": "2024-02-02T17:01:54.136961", "exception": false, "start_time": "2024-02-02T17:01:54.122971", "status": "completed"} id="275aaeac"
# <div class="table-of-contents" style="background-color:#000000; padding: 20px; margin: 10px; font-size: 110%; border-radius: 25px; box-shadow: 10px 10px 5px 0px rgba(0,0,0,0.75);">
#   <h1 style="color:#17E8C4;">TOC</h1>
#   <ol>
#     <li><a href="#1" style="color: #17E8C4;">1. Import</a></li>
#       <li><a href="#2" style="color: #17E8C4;">2. Data Analysis</a></li>
#     <li><a href="#3" style="color: #17E8C4;">3. Data Preprocessing</a></li>
#     <li><a href="#4" style="color: #17E8C4;">4. Model Implementation</a></li>
#     <li><a href="#5" style="color: #17E8C4;">5. Sklearn Implementation</a></li>
#      <li><a href="#6" style="color: #17E8C4;">6. Thank You</a></li>
#   </ol>
# </div>
#
# <a id="1"></a>
# <h1 style='background:#000000;border:0; color:black;
#     box-shadow: 10px 10px 5px 0px rgba(0,0,0,0.75);
#     transform: rotateX(10deg);
#     '><center style='color: #17E8C4;'>Imports</center></h1>
#

# %% colab={"base_uri": "https://localhost:8080/"} id="2F6UY7E2mX70" executionInfo={"status": "ok", "timestamp": 1725609946429, "user_tz": -180, "elapsed": 2297, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="3ed60492-26d0-4cc2-c8bb-d10e0f4b2f5d"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="_ZuA39IvmcbL" executionInfo={"status": "ok", "timestamp": 1725609946429, "user_tz": -180, "elapsed": 12, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="1185b654-30fd-4ea7-b218-212a4e2061e8"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/SVM

# %% papermill={"duration": 2.477814, "end_time": "2024-02-02T17:01:56.627616", "exception": false, "start_time": "2024-02-02T17:01:54.149802", "status": "completed"} id="a2c98820" executionInfo={"status": "ok", "timestamp": 1725609946429, "user_tz": -180, "elapsed": 12, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# %% [markdown] papermill={"duration": 0.012209, "end_time": "2024-02-02T17:01:56.652567", "exception": false, "start_time": "2024-02-02T17:01:56.640358", "status": "completed"} id="ece4f5ea"
# <a id="2"></a>
# <h1 style='background:#000000;border:0; color:black;
#     box-shadow: 10px 10px 5px 0px rgba(0,0,0,0.75);
#     transform: rotateX(10deg);
#     '><center style='color: #17E8C4;'>Data Analysis</center></h1>
#
# # Data Analysis

# %% papermill={"duration": 0.088004, "end_time": "2024-02-02T17:01:56.753369", "exception": false, "start_time": "2024-02-02T17:01:56.665365", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 273} id="a65a56f9" executionInfo={"status": "ok", "timestamp": 1725609946429, "user_tz": -180, "elapsed": 11, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="2ab21368-3e62-40ce-a613-0b6c590762db"
df = pd.read_csv('breast-cancer.csv')
df.head()

# %% papermill={"duration": 1.086058, "end_time": "2024-02-02T17:01:57.854584", "exception": false, "start_time": "2024-02-02T17:01:56.768526", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 542} id="84c21fe4" executionInfo={"status": "ok", "timestamp": 1725609946429, "user_tz": -180, "elapsed": 11, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="832940b7-e3f1-41be-83af-5e94d233621b"
px.histogram(data_frame=df, x='diagnosis', color='diagnosis',color_discrete_sequence=['#05445E','#75E6DA'])

# %% papermill={"duration": 0.23342, "end_time": "2024-02-02T17:01:58.102047", "exception": false, "start_time": "2024-02-02T17:01:57.868627", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 542} id="c3941fa4" executionInfo={"status": "ok", "timestamp": 1725609947004, "user_tz": -180, "elapsed": 585, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0be12fb7-3344-4a02-a30a-54b1d342c2b2"
px.histogram(data_frame=df,x='area_mean',color='diagnosis',color_discrete_sequence=['#05445E','#75E6DA'])

# %% papermill={"duration": 0.094228, "end_time": "2024-02-02T17:01:58.210533", "exception": false, "start_time": "2024-02-02T17:01:58.116305", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 542} id="251e9ef9" executionInfo={"status": "ok", "timestamp": 1725609947004, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f50b7e2d-c59d-435c-87b7-7619d5cf4aa8"
px.histogram(data_frame=df,x='radius_mean',color='diagnosis',color_discrete_sequence=['#05445E','#75E6DA'])

# %% papermill={"duration": 0.099398, "end_time": "2024-02-02T17:01:58.324598", "exception": false, "start_time": "2024-02-02T17:01:58.225200", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 542} id="9b6f963c" executionInfo={"status": "ok", "timestamp": 1725609947005, "user_tz": -180, "elapsed": 22, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="361401d4-d60b-4240-bb2f-82d85cf660ac"
px.histogram(data_frame=df,x='perimeter_mean',color='diagnosis',color_discrete_sequence=['#05445E','#75E6DA'])

# %% papermill={"duration": 0.102487, "end_time": "2024-02-02T17:01:58.442103", "exception": false, "start_time": "2024-02-02T17:01:58.339616", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 542} id="78737556" executionInfo={"status": "ok", "timestamp": 1725609947005, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="06ab4f3f-ffe5-46d6-d678-8f368f864f36"
px.histogram(data_frame=df,x='smoothness_mean',color='diagnosis',color_discrete_sequence=['#05445E','#75E6DA'])

# %% papermill={"duration": 0.098645, "end_time": "2024-02-02T17:01:58.555991", "exception": false, "start_time": "2024-02-02T17:01:58.457346", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 542} id="e59298a1" executionInfo={"status": "ok", "timestamp": 1725609947005, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="748e9a0e-5a30-4dd7-b935-729a0e02a0da"
px.histogram(data_frame=df,x='texture_mean',color='diagnosis',color_discrete_sequence=['#05445E','#75E6DA'])

# %% papermill={"duration": 0.118603, "end_time": "2024-02-02T17:01:58.690353", "exception": false, "start_time": "2024-02-02T17:01:58.571750", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 542} id="11acdd02" executionInfo={"status": "ok", "timestamp": 1725609947005, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="fafd6275-4870-437c-f802-9a040ac4b31f"
px.scatter(data_frame=df,x='symmetry_worst',color='diagnosis',color_discrete_sequence=['#05445E','#75E6DA'])


# %% papermill={"duration": 0.100941, "end_time": "2024-02-02T17:01:58.807239", "exception": false, "start_time": "2024-02-02T17:01:58.706298", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 542} id="883cd945" executionInfo={"status": "ok", "timestamp": 1725609947414, "user_tz": -180, "elapsed": 427, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="647d05ce-05b2-4efc-f191-b6a2cf2b1d54"
px.scatter(data_frame=df,x='concavity_worst',color='diagnosis',color_discrete_sequence=['#05445E','#75E6DA'])


# %% papermill={"duration": 0.098099, "end_time": "2024-02-02T17:01:58.922394", "exception": false, "start_time": "2024-02-02T17:01:58.824295", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 542} id="89fafbab" executionInfo={"status": "ok", "timestamp": 1725609947414, "user_tz": -180, "elapsed": 27, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9d844c8a-fa5f-43d6-c23d-0766e0cbc31a"
px.scatter(data_frame=df,x='fractal_dimension_worst',color='diagnosis',color_discrete_sequence=['#05445E','#75E6DA'])


# %% [markdown] papermill={"duration": 0.016105, "end_time": "2024-02-02T17:01:58.955200", "exception": false, "start_time": "2024-02-02T17:01:58.939095", "status": "completed"} id="19d3714f"
# <a id="3"></a>
# <h1 style='background:#000000;border:0; color:black;
#     box-shadow: 10px 10px 5px 0px rgba(0,0,0,0.75);
#     transform: rotateX(10deg);
#     '><center style='color: #17E8C4;'>Data Preprocessing</center></h1>
#
# # Data Preprocessing

# %% papermill={"duration": 0.059591, "end_time": "2024-02-02T17:01:59.031117", "exception": false, "start_time": "2024-02-02T17:01:58.971526", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 273} id="3aee4e9a" executionInfo={"status": "ok", "timestamp": 1725609947414, "user_tz": -180, "elapsed": 26, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c9a24b72-c43b-498b-f5a6-537645cfe301"
df = pd.read_csv('breast-cancer.csv')

df.head()

# %% papermill={"duration": 0.028383, "end_time": "2024-02-02T17:01:59.076635", "exception": false, "start_time": "2024-02-02T17:01:59.048252", "status": "completed"} id="57ca5252" executionInfo={"status": "ok", "timestamp": 1725609947414, "user_tz": -180, "elapsed": 25, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df.drop('id', axis=1, inplace=True) #drop redundant columns

# %% papermill={"duration": 0.130464, "end_time": "2024-02-02T17:01:59.224019", "exception": false, "start_time": "2024-02-02T17:01:59.093555", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 990} id="72034b8c" executionInfo={"status": "ok", "timestamp": 1725609947414, "user_tz": -180, "elapsed": 25, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e7e4ebd9-400c-4215-dd6f-6cf0bf5b1a13"
df.describe().T

# %% [markdown] papermill={"duration": 0.022419, "end_time": "2024-02-02T17:01:59.264318", "exception": false, "start_time": "2024-02-02T17:01:59.241899", "status": "completed"} id="7dd961e2"
#
# ## Encode target

# %% papermill={"duration": 0.037038, "end_time": "2024-02-02T17:01:59.323355", "exception": false, "start_time": "2024-02-02T17:01:59.286317", "status": "completed"} id="ff19e339" executionInfo={"status": "ok", "timestamp": 1725609947414, "user_tz": -180, "elapsed": 24, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df['diagnosis'] = (df['diagnosis'] == 'M').astype(int) #encode the label into 1/0

# %% colab={"base_uri": "https://localhost:8080/", "height": 900} id="9auV1iap_9rv" executionInfo={"status": "ok", "timestamp": 1725609947414, "user_tz": -180, "elapsed": 24, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="5cba37a5-5d97-4ec4-d054-cdbf6ff483af"
df.head(25)

# %% colab={"base_uri": "https://localhost:8080/"} id="CnQg3sJTr4Wf" executionInfo={"status": "ok", "timestamp": 1725609947414, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="15f62dc8-e1d9-4194-b804-46c63345e49f"
print(df['diagnosis'].unique())

# %% [markdown] papermill={"duration": 0.017299, "end_time": "2024-02-02T17:01:59.360755", "exception": false, "start_time": "2024-02-02T17:01:59.343456", "status": "completed"} id="47590ef2"
# ## Get highly correlated features

# %% papermill={"duration": 0.029244, "end_time": "2024-02-02T17:01:59.407473", "exception": false, "start_time": "2024-02-02T17:01:59.378229", "status": "completed"} id="1168023b" executionInfo={"status": "ok", "timestamp": 1725609947414, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
corr = df.corr()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="3N8bofVyXKnq" executionInfo={"status": "ok", "timestamp": 1725609947414, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="3b166ac8-4762-4d1e-d508-c4c1ab074b71"
corr

# %% papermill={"duration": 4.457394, "end_time": "2024-02-02T17:02:03.882240", "exception": false, "start_time": "2024-02-02T17:01:59.424846", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 1000} id="2018dce4" executionInfo={"status": "ok", "timestamp": 1725609951091, "user_tz": -180, "elapsed": 3697, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="059fed9a-cc67-4d4d-df95-917a218645b2"
plt.figure(figsize=(20,20))
sns.heatmap(corr, cmap='mako_r',annot=True)
plt.show()

# %% [markdown] id="7FvdjjrxVIa9"
# ## Manual feature selection

# %% papermill={"duration": 0.046741, "end_time": "2024-02-02T17:02:03.954859", "exception": false, "start_time": "2024-02-02T17:02:03.908118", "status": "completed"} id="17674a5d" executionInfo={"status": "ok", "timestamp": 1725609951091, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Get the absolute value of the correlation
cor_target = abs(corr["diagnosis"])

# Select highly correlated features (thresold = 0.2)
relevant_features = cor_target[cor_target>0.1]

# %% colab={"base_uri": "https://localhost:8080/"} id="rSsflhEStOSI" executionInfo={"status": "ok", "timestamp": 1725609951091, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a04cd74c-f7ab-4a99-f6fc-58e24a64ee38"
print(cor_target)

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="tXGrFSMLuOOJ" executionInfo={"status": "ok", "timestamp": 1725609951092, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="397364ca-7bdf-42aa-f60a-587f5533c984"
relevant_features

# %% colab={"base_uri": "https://localhost:8080/"} id="E4Jpl_XHuLNS" executionInfo={"status": "ok", "timestamp": 1725609951092, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="630d819d-d2b5-41bd-f46d-39121b62fb46"
# Collect the names of the features
names = [index for index, value in relevant_features.items()]

# Drop the target variable from the results
names.remove('diagnosis')

# Display the results
print(names)

# %% colab={"base_uri": "https://localhost:8080/"} id="cBT7BAL2Vw0z" executionInfo={"status": "ok", "timestamp": 1725609951092, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="14ebdaa6-2fdc-4e74-d434-059adf9d87a9"
len(names)

# %% [markdown] papermill={"duration": 0.025249, "end_time": "2024-02-02T17:02:04.006431", "exception": false, "start_time": "2024-02-02T17:02:03.981182", "status": "completed"} id="7cd5d17e"
# ## Assign data and labels

# %% papermill={"duration": 0.036892, "end_time": "2024-02-02T17:02:04.069025", "exception": false, "start_time": "2024-02-02T17:02:04.032133", "status": "completed"} id="ed0181a8" executionInfo={"status": "ok", "timestamp": 1725609951092, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X = df[names].values
y = df['diagnosis']


# %% [markdown] papermill={"duration": 0.025741, "end_time": "2024-02-02T17:02:04.121050", "exception": false, "start_time": "2024-02-02T17:02:04.095309", "status": "completed"} id="5a714cbf"
# ## Scale the data

# %% papermill={"duration": 0.037071, "end_time": "2024-02-02T17:02:04.184400", "exception": false, "start_time": "2024-02-02T17:02:04.147329", "status": "completed"} id="2406e74d" executionInfo={"status": "ok", "timestamp": 1725609951092, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
def scale(X):
    """
    Standardizes the data in the array X.

    Parameters:
        X (numpy.ndarray): Features array of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: The standardized features array.
    """
    # Calculate the mean and standard deviation of each feature
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Standardize the data
    X = (X - mean) / std
    return X



# %% papermill={"duration": 0.035292, "end_time": "2024-02-02T17:02:04.245155", "exception": false, "start_time": "2024-02-02T17:02:04.209863", "status": "completed"} id="4fba542a" executionInfo={"status": "ok", "timestamp": 1725609951092, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X = scale(X)

# %% colab={"base_uri": "https://localhost:8080/"} id="oRK_0gEQWO2M" executionInfo={"status": "ok", "timestamp": 1725609951092, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b7c76b14-b698-4b54-cce8-737dc829ce07"
X


# %% [markdown] papermill={"duration": 0.028281, "end_time": "2024-02-02T17:02:04.300412", "exception": false, "start_time": "2024-02-02T17:02:04.272131", "status": "completed"} id="cbe86cf8"
# ## Split into train and Testing

# %% papermill={"duration": 0.039321, "end_time": "2024-02-02T17:02:04.365984", "exception": false, "start_time": "2024-02-02T17:02:04.326663", "status": "completed"} id="0455191d" executionInfo={"status": "ok", "timestamp": 1725609951092, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
def train_test_split(X, y, random_state=41, test_size=0.2):
    """
    Splits the data into training and testing sets.

    Parameters:
        X (numpy.ndarray): Features array of shape (n_samples, n_features).
        y (numpy.ndarray): Target array of shape (n_samples,).
        random_state (int): Seed for the random number generator. Default is 42.
        test_size (float): Proportion of samples to include in the test set. Default is 0.2.

    Returns:
        Tuple[numpy.ndarray]: A tuple containing X_train, X_test, y_train, y_test.
    """
    # Get number of samples
    n_samples = X.shape[0]    #1000

    # Set the seed for the random number generator
    np.random.seed(random_state)

    # Shuffle the indices
    shuffled_indices = np.random.permutation(np.arange(n_samples))   #  0 - 999

    # Determine the size of the test set
    test_size = int(n_samples * test_size)   # 200

    # Split the indices into test and train
    test_indices = shuffled_indices[:test_size]  #0  -  199
    train_indices = shuffled_indices[test_size:] #200 - 999

    print("Test indices:", test_indices)
    print("Train indices:", train_indices)

    # Split the features and target arrays into test and train
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


# %% papermill={"duration": 0.036268, "end_time": "2024-02-02T17:02:04.428434", "exception": false, "start_time": "2024-02-02T17:02:04.392166", "status": "completed"} id="46976d6b" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1725609951092, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0c3d9107-10f1-4cb1-8c0f-e68def4669ab"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #split the  data into traing and validating


# %% [markdown] papermill={"duration": 0.026068, "end_time": "2024-02-02T17:02:04.480242", "exception": false, "start_time": "2024-02-02T17:02:04.454174", "status": "completed"} id="f6259447"
# # Model Implementation

# %% papermill={"duration": 0.036643, "end_time": "2024-02-02T17:02:04.670458", "exception": false, "start_time": "2024-02-02T17:02:04.633815", "status": "completed"} id="919523d3" executionInfo={"status": "ok", "timestamp": 1725610079081, "user_tz": -180, "elapsed": 279, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
def accuracy(y_true, y_pred):
    """
    Computes the accuracy of a classification model.

    Parameters:
    ----------
        y_true (numpy array): A numpy array of true labels for each data point.
        y_pred (numpy array): A numpy array of predicted labels for each data point.

    Returns:
    ----------
        float: The accuracy of the model
    """
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return (correct_predictions / total_samples)


# %% [markdown] papermill={"duration": 0.025344, "end_time": "2024-02-02T17:02:11.585326", "exception": false, "start_time": "2024-02-02T17:02:11.559982", "status": "completed"} id="b766aff7"
# # Sklearn Implementation
#

# %% papermill={"duration": 0.540443, "end_time": "2024-02-02T17:02:12.151461", "exception": false, "start_time": "2024-02-02T17:02:11.611018", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="d8cbaaf4" executionInfo={"status": "ok", "timestamp": 1725610140517, "user_tz": -180, "elapsed": 1730, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="db6b7c43-5cfb-44ec-8888-aba349b8eaf5"
from sklearn.svm import SVC
skmodel = SVC()
skmodel.fit(X_train, y_train)
sk_predictions = skmodel.predict(X_test)

accuracy(y_test, sk_predictions)

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="A1gmy4AXw-5F" executionInfo={"status": "ok", "timestamp": 1725610214717, "user_tz": -180, "elapsed": 584, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="859f6a54-9bfe-479e-dbd1-7ab14db8a89f"
# prompt: need to get the confusion matriux  plotted with classes names not numbers

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, sk_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot()
plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} id="MoOgJkb5xJ5A" executionInfo={"status": "ok", "timestamp": 1725610222618, "user_tz": -180, "elapsed": 269, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="7f879eac-d6d9-4695-8bde-82c3fec8daaa"
from sklearn.metrics import classification_report
print(classification_report(y_test, sk_predictions, digits=4))


# %% colab={"base_uri": "https://localhost:8080/"} id="mhPwqj9TxX7A" executionInfo={"status": "ok", "timestamp": 1725610248618, "user_tz": -180, "elapsed": 270, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="09971088-83e2-45b5-efa6-f2c04226e3e1"
# Calculate overall accuracy
overall_accuracy = accuracy(y_test, sk_predictions)

# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel()

# Calculate specificity
specificity = tn / (tn + fp)

print("Overall Accuracy:", overall_accuracy)
print("Specificity:", specificity)


# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="ERMOlXioxhkK" executionInfo={"status": "ok", "timestamp": 1725610319550, "user_tz": -180, "elapsed": 616, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f9b736a9-1f3a-4ecf-85b2-a49ef76af25e"
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for positive class
y_scores = skmodel.decision_function(X_test)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calculate AUC score
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} id="Ur73EliNY-70" executionInfo={"status": "ok", "timestamp": 1725610471199, "user_tz": -180, "elapsed": 3870, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a60ac78d-5e27-4b44-d50d-6d55aa496313"
# Need to apply svm using Grid search for all parameters

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel type
}

# Create the SVM model
model = SVC()

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, digits=4))

