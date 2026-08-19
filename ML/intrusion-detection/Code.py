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

# %% id="OoXj_oJnxU4L" executionInfo={"status": "ok", "timestamp": 1726821232486, "user_tz": -180, "elapsed": 320, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import roc_curve, auc

# %% id="JmDqpxoxC5IQ" executionInfo={"status": "ok", "timestamp": 1726821232783, "user_tz": -180, "elapsed": 3, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
import warnings
warnings.filterwarnings("ignore")

# %% colab={"base_uri": "https://localhost:8080/"} id="RDa5iB_7dtsy" executionInfo={"status": "ok", "timestamp": 1726821234803, "user_tz": -180, "elapsed": 2023, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f5d7307b-1fac-464a-e9a7-11bcbef273c0"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="qn7lWmWkdyEL" executionInfo={"status": "ok", "timestamp": 1726821234803, "user_tz": -180, "elapsed": 40, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="23557431-63c3-4b2f-b81d-bcdcd5c227d6"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/Intrusion detection

# %% [markdown] id="0pup0O35C5IQ"
# The dataset is available in Kaggle at link: https://www.kaggle.com/code/bhaktapri/network-intrusion-detection/input

# %% [markdown] id="IHoO5Mu_3Mbs"
# ## Data preprocessing

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 39, "status": "ok", "timestamp": 1726821234803, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="gHeevwAFxdgu" outputId="5a063dec-0d05-4d93-c6f1-31879b5e255b"
# Load the dataset
data = pd.read_csv('Train_data.csv')
data_test = pd.read_csv('Test_data.csv')

new_file=data_test

data.info()

# %% id="vhG_mOAhL1JI" executionInfo={"status": "ok", "timestamp": 1726821234803, "user_tz": -180, "elapsed": 38, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# prompt: I want to rename the last column in data from class to Class

data.rename(columns={'class': 'Class'}, inplace=True)


# %% colab={"base_uri": "https://localhost:8080/", "height": 178} executionInfo={"elapsed": 38, "status": "ok", "timestamp": 1726821234803, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="QSOHavwyB_EE" outputId="77fdcad2-e1e0-4cf3-bd4d-c6a64121cdec"
data.Class.value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 37, "status": "ok", "timestamp": 1726821234803, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="5Yn4roxNB_Mm" outputId="81225175-9434-40c4-f576-b93d60bc73af"
data.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 358, "status": "ok", "timestamp": 1726821235126, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="VgiFJKGp9kMH" outputId="45f8cd1a-84dc-4e7a-cadb-265846a0e302"
data.head()

# %% id="Xby23AX_xgYK" executionInfo={"status": "ok", "timestamp": 1726821235126, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Step 2: Data Encoding
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Identify object columns
object_columns = data.select_dtypes(include=['object']).columns
object_columns_test = data_test.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each object column
for col in object_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Apply label encoding to each object column
for col in object_columns_test:
    data_test[col] = label_encoder.fit_transform(data_test[col])

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1726821235126, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="TW4iFpepBYBS" outputId="a2d50b1e-986f-4ec7-bbe5-8d0d9a5ec121"
data.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 706} executionInfo={"elapsed": 6171, "status": "ok", "timestamp": 1726821241275, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="kCOebbZABqDY" outputId="f337ddcf-51fd-4d46-a543-2fa81f32fa89"
# Step 3 EDA
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = data.corr()

# Create a heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Correlation Heatmap')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"elapsed": 372, "status": "ok", "timestamp": 1726821241624, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1QsM1oPNBqRu" outputId="fd096081-da10-4ef7-cdd3-5977b651a8c1"
# Visualize the distribution of the target variable 'Grade'
plt.figure(figsize=(8, 5))
sns.countplot(x='Class', data=data)
plt.title('Distribution of Output')
plt.show()

# %% id="DV9ajIEByZ7d" executionInfo={"status": "ok", "timestamp": 1726821241624, "user_tz": -180, "elapsed": 32, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Step 4 Extarct input and output
X=data.iloc[:,:-1]  # all columns without the last one
y=data.iloc[:,-1]   # last column only

# %% id="X9zWWFA5yxgs" executionInfo={"status": "ok", "timestamp": 1726821241624, "user_tz": -180, "elapsed": 32, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Step 5 Data scaling
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()    #  convert to (-3 to +3)

# Fit the scaler to your data and transform it
X_scaled = scaler.fit_transform(X)
X_scaled_test=scaler.fit_transform(data_test)


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1726821241624, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="XpoeHrFK0V1-" outputId="770827a6-f9b7-436f-e0dc-142e9deae86c"
print(X_scaled[:1])

# %% id="C3oXxcprC5IW" outputId="9517ad33-1a66-4baa-8089-86a0a5721a72" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1726821241624, "user_tz": -180, "elapsed": 31, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
print(X_scaled_test[:1])

# %% id="7RkxZbgNzBVB" executionInfo={"status": "ok", "timestamp": 1726821241624, "user_tz": -180, "elapsed": 30, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Step 6 Feature selection

from sklearn.feature_selection import SelectKBest, f_regression

# Initialize SelectKBest with the scoring function
selector = SelectKBest(score_func=f_regression, k=25)

# Fit and transform your data
X = selector.fit_transform(X_scaled, y)
data_test=selector.transform(X_scaled_test)

# %% id="JVdDvmMdWxvo" executionInfo={"status": "ok", "timestamp": 1726821241624, "user_tz": -180, "elapsed": 30, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# prompt: need to split the X_train to train and test 80:20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1726821241625, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="e3dg1PSuzdhT" outputId="a4edbf9e-ea46-4709-ca14-411187461640"
print(X_train.shape)
print(X_test.shape)

# %% [markdown] id="ekpp9bHK1HKD"
# ## Modeling

# %% [markdown] id="dR_4EldR3Bzl"
# # Random forest

# %% id="1GpeRZj21GlC" executionInfo={"status": "ok", "timestamp": 1726821242804, "user_tz": -180, "elapsed": 1209, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 623} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1726821242804, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="mY0FYfkz2Fg9" outputId="f6f32ac8-efce-4a6d-cc2f-973594b8d63f"
# Classification report
class_report = classification_report(y_test, y_pred,digits=4)
print("Classification Report:")
print(class_report)

from sklearn import metrics

cm=confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)

cm_display.plot()
plt.show()

# %% [markdown] id="UyXBtX1gNLNp"
# ## Logistic Regression

# %% id="Cx6NDnmRMkII" executionInfo={"status": "ok", "timestamp": 1726821243115, "user_tz": -180, "elapsed": 335, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
logreg_model = LogisticRegression(random_state=42)

# Fit the model to the training data
logreg_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = logreg_model.predict(X_test)


# %% colab={"base_uri": "https://localhost:8080/", "height": 623} executionInfo={"elapsed": 386, "status": "ok", "timestamp": 1726821243494, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="gNGRlDWCMkN7" outputId="a928c448-346b-4d21-9861-b1986c6d8180"
# Classification report
class_report = classification_report(y_test, y_pred,digits=4)
print("Classification Report:")
print(class_report)

from sklearn import metrics

cm=confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)

cm_display.plot()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1726821243494, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="lTTFjOzeCclz" outputId="c8ccc876-228f-472d-8553-59e4baf9566a"
# prompt: predict the X_test using the best model and print first ten outputs

y_pred_test = rf_classifier.predict(data_test)
print(y_pred_test[:10])


# %% id="8AtX5-yHDPhA" executionInfo={"status": "ok", "timestamp": 1726821243789, "user_tz": -180, "elapsed": 318, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# prompt: save the ouput of y_pred_test in new column as output in the Test_data.csv file

new_file['output']=y_pred_test
new_file.to_csv('Test_data_output.csv', index=False)

