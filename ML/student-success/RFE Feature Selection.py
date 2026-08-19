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

# %% [markdown] id="730cf795"
# <h1> Student Success Prediction using Machine learning</h>

# %% papermill={"duration": 1.53618, "end_time": "2022-01-19T01:39:23.515347", "exception": false, "start_time": "2022-01-19T01:39:21.979167", "status": "completed"} id="72e78d72"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score,learning_curve, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import auc,confusion_matrix, roc_curve, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# %% colab={"base_uri": "https://localhost:8080/"} id="a-EFKoOW3bU0" executionInfo={"status": "ok", "timestamp": 1722677620284, "user_tz": -180, "elapsed": 20220, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="35a47df0-5723-4570-bd4d-e94b9f4480a4"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="XpUR1fFX3gkE" executionInfo={"status": "ok", "timestamp": 1722677645339, "user_tz": -180, "elapsed": 633, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f056bf0e-d054-4bcc-eb22-56bd36095e54"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/Student Success

# %% [markdown] id="5a1c257f"
# ### Read the dataset

# %% papermill={"duration": 0.070758, "end_time": "2022-01-19T01:39:23.736923", "exception": false, "start_time": "2022-01-19T01:39:23.666165", "status": "completed"} id="b25c241b"
data = pd.read_csv("dataset.csv", sep=";")

# %% papermill={"duration": 0.074315, "end_time": "2022-01-19T01:39:23.861711", "exception": false, "start_time": "2022-01-19T01:39:23.787396", "status": "completed"} id="6df63fbd" outputId="664db16a-ef4f-41d5-d0f8-374a07693754" colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"status": "ok", "timestamp": 1722677646013, "user_tz": -180, "elapsed": 11, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.head()

# %% papermill={"duration": 0.059785, "end_time": "2022-01-19T01:39:24.073298", "exception": false, "start_time": "2022-01-19T01:39:24.013513", "status": "completed"} id="83503634" outputId="538feade-26d8-4dab-c603-bbdce9da6fab" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722677646013, "user_tz": -180, "elapsed": 9, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.shape

# %% papermill={"duration": 0.078689, "end_time": "2022-01-19T01:39:24.203703", "exception": false, "start_time": "2022-01-19T01:39:24.125014", "status": "completed"} id="06963df7" outputId="d0c0db12-9283-45f1-8462-0443ab3e3eb6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722677646013, "user_tz": -180, "elapsed": 8, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.info()

# %% papermill={"duration": 0.094948, "end_time": "2022-01-19T01:39:24.349908", "exception": false, "start_time": "2022-01-19T01:39:24.254960", "status": "completed"} id="100e5863" outputId="ae36d7a8-7101-4aec-e4a0-9407c8695e93" colab={"base_uri": "https://localhost:8080/", "height": 495} executionInfo={"status": "ok", "timestamp": 1722677646478, "user_tz": -180, "elapsed": 472, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.describe(include="all")

# %% [markdown] id="07d6aabf"
# ### check if the dataset has null values

# %% papermill={"duration": 0.06262, "end_time": "2022-01-19T01:39:24.464107", "exception": false, "start_time": "2022-01-19T01:39:24.401487", "status": "completed"} id="619b37db" outputId="622742b2-5768-4138-8c98-b17a27725bf9" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722677646478, "user_tz": -180, "elapsed": 10, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.isnull().sum()

# %% [markdown] id="196311a6"
# ### Count the number of every class in the output

# %% papermill={"duration": 0.227446, "end_time": "2022-01-19T01:39:26.564642", "exception": false, "start_time": "2022-01-19T01:39:26.337196", "status": "completed"} id="4fd223d5" outputId="97999d58-08aa-4d81-fa01-7ee55f434fbe" colab={"base_uri": "https://localhost:8080/", "height": 752} executionInfo={"status": "ok", "timestamp": 1722677646852, "user_tz": -180, "elapsed": 383, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
sns.countplot(data.Target, palette="bwr")
plt.show()
data.Target.value_counts(normalize=True)

# %% [markdown] id="1dec189e"
# ## Convert Dropout =0, Graduate=1, Enrolled=3

# %% id="0bebdbcc"
data = pd.read_csv("dataset.csv", sep=";")
data["Target"]=data.Target.map(dict( Dropout =0, Graduate=1, Enrolled=2))

# %% [markdown] id="683e404c"
# ### select only Dropout and Graduate because the number of Enrolled students is very low

# %% id="62cf8a44"
data = data[data['Target'] != 2]

# %% [markdown] id="1d82725c"
# ### Read the input and the output

# %% papermill={"duration": 0.073658, "end_time": "2022-01-19T01:39:34.690351", "exception": false, "start_time": "2022-01-19T01:39:34.616693", "status": "completed"} id="011f71a6"
X = data.drop("Target",axis=1)
y = data["Target"]

# %% papermill={"duration": 0.081611, "end_time": "2022-01-19T01:39:34.840205", "exception": false, "start_time": "2022-01-19T01:39:34.758594", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="2640555e" executionInfo={"status": "ok", "timestamp": 1722677647197, "user_tz": -180, "elapsed": 11, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d5a57112-f61f-4cdd-f871-f6bbaf862b4b"
X.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="855e83f2" executionInfo={"status": "ok", "timestamp": 1722677647197, "user_tz": -180, "elapsed": 10, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b32edb79-ff09-40f9-da31-d16f7ad1bf82"
y.head()

# %% [markdown] id="daed8339"
# ## Data scaling

# %% id="dc204ef3"
from sklearn import preprocessing
X= preprocessing.StandardScaler().fit_transform(X)

# %% [markdown] id="7451e7ec"
# ## Data splitting

# %% id="d2e48059"
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=10)

# %% [markdown] papermill={"duration": 0.068839, "end_time": "2022-01-19T01:39:35.132728", "exception": false, "start_time": "2022-01-19T01:39:35.063889", "status": "completed"} id="83ff4275"
# ## Build models

# %% id="a48ed886"
from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
rf_3 = RandomForestClassifier()
rfe = RFE(estimator=rf_3, n_features_to_select=20, step=1)
rfe = rfe.fit(X_train, y_train)
y_pred=rfe.predict(X_test)

# %% id="fRG3FEX--klT" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722677658554, "user_tz": -180, "elapsed": 602, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="5d3cb836-2789-4b9f-8d4e-be0ae2d07475"
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#  Draw the roc curev and auc using the results of random forest
# ROC Curve and AUC
y_pred_proba = rfe.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# %% id="R5ErOlxnzqJe"
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV


eclf = VotingClassifier(estimators=[
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression()),
    ], voting='soft')


eclf.fit(X_train,y_train)


y_pred=eclf.predict(X_test)

# %% id="NLFAzlii8p-5" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722677660471, "user_tz": -180, "elapsed": 814, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="af50e2b9-2d0b-4df5-93a4-10032dcc06c0"
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
y_pred_proba = eclf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] papermill={"duration": 0.065493, "end_time": "2022-01-19T01:39:35.678299", "exception": false, "start_time": "2022-01-19T01:39:35.612806", "status": "completed"} id="b6b3ab59"
# ## Logistic Regression Model

# %% papermill={"duration": 3.120202, "end_time": "2022-01-19T01:39:38.863804", "exception": false, "start_time": "2022-01-19T01:39:35.743602", "status": "completed"} id="0dcac604"
LR=LogisticRegression(max_iter=1000)

rfe = RFE(estimator=LR, n_features_to_select=20, step=1)
rfe = rfe.fit(X_train, y_train)
y_pred=rfe.predict(X_test)

# %% [markdown] id="5d182a34"
# ### Metrics calculation

# %% id="2eb737ae" outputId="269d080c-6e9d-4f48-d09d-a9a7726bdf76" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722677661967, "user_tz": -180, "elapsed": 985, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
y_pred_proba = rfe.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] papermill={"duration": 0.068996, "end_time": "2022-01-19T01:39:48.918816", "exception": false, "start_time": "2022-01-19T01:39:48.849820", "status": "completed"} id="2751fc1b"
# ## Random Forest implementation

# %% papermill={"duration": 4.941656, "end_time": "2022-01-19T01:39:53.930617", "exception": false, "start_time": "2022-01-19T01:39:48.988961", "status": "completed"} id="1eba2000"
from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
rf_3 = RandomForestClassifier()
rfe = RFE(estimator=rf_3, n_features_to_select=20, step=1)
rfe = rfe.fit(X_train, y_train)
y_pred=rfe.predict(X_test)

# %% papermill={"duration": 0.167409, "end_time": "2022-01-19T01:39:54.168655", "exception": false, "start_time": "2022-01-19T01:39:54.001246", "status": "completed"} id="be21b196" outputId="ee81694f-99d0-49c6-c6cd-1e602a30987d" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722677672400, "user_tz": -180, "elapsed": 593, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
y_pred_proba = rfe.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] id="43ca72d0"
# ## XGBoost Classifier

# %% id="491bb18b" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722677683695, "user_tz": -180, "elapsed": 11302, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="7311e0e0-b2c6-480d-a2ed-303a7956d362"
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

rfe = RFE(estimator=xgb, n_features_to_select=20, step=1)
rfe = rfe.fit(X_train, y_train)
y_pred=rfe.predict(X_test)



# %% id="27d9a51d" outputId="83c42e22-e2a4-439d-eff8-e126f3cc6e1a" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722677684673, "user_tz": -180, "elapsed": 987, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
y_pred_proba = rfe.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] id="3Qdj14WDuIsK"
# ## GradientBoostingClassifier

# %% id="a0441d03"
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
GBC=GradientBoostingClassifier()

rfe = RFE(estimator=GBC, step=1)
rfe = rfe.fit(X_train, y_train)
y_pred=rfe.predict(X_test)

# %% [markdown] id="gcfcOSP-z47O"
#

# %% id="1cbdc1db" outputId="89a11fc8-04a1-47ec-f490-4f759df3f65b" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722677702192, "user_tz": -180, "elapsed": 935, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
y_pred_proba = rfe.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# %% [markdown] id="9e957fad"
# ## Linear_SGD classifier

# %% [markdown] id="aaq6gIdRuS_o"
#

# %% id="d22e18c6"
# Implementing Linear_SGD classifier

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(max_iter=1000, loss='log_loss') # Change loss to 'log_loss' for probability estimates
rfe = RFE(estimator=clf, n_features_to_select=20, step=1)
rfe = rfe.fit(X_train, y_train)
y_pred=rfe.predict(X_test)


# %% id="e1943129" outputId="9964044d-1ffa-41c9-c254-2a8cf52e7d2e" colab={"base_uri": "https://localhost:8080/", "height": 573} executionInfo={"status": "ok", "timestamp": 1722678016788, "user_tz": -180, "elapsed": 1539, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="ucD3i-Memn9q" executionInfo={"status": "ok", "timestamp": 1722678055875, "user_tz": -180, "elapsed": 27813, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e2a106e7-b90a-406c-8621-3793336d82d1"
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

# define the base models
level0 = list()
level0.append(('LR', LogisticRegression(C= 0.1, penalty='l2')))
level0.append(('RF',RandomForestClassifier(n_estimators= 300)))
level0.append(('SGD',SGDClassifier(alpha= 0.01)))
level0.append(('XGB',XGBClassifier(subsample= 0.8, min_child_weight= 5, max_depth= 5, gamma= 1, colsample_bytree= 0.8)))
level0.append(('MLP',MLPClassifier(activation= 'relu', alpha= 0.0001, hidden_layer_sizes= (20,), learning_rate= 'adaptive', solver= 'adam')))
level0.append(('Adaboost',AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6, min_samples_leaf=1, min_samples_split=2, splitter= 'random'))))
level0.append(('bagging',BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=6, min_samples_leaf=1, min_samples_split=2, splitter= 'random'))))

# define meta learner model
level1= SVC(C= 0.7999999999999999, gamma= 0.04, kernel= 'rbf')
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %% id="LPBHz3Ye9DQU" colab={"base_uri": "https://localhost:8080/", "height": 573} executionInfo={"status": "ok", "timestamp": 1722678057179, "user_tz": -180, "elapsed": 563, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e41616e8-92f7-4e3c-9523-19007ab2a535"
# Need to print the classiiofcation report , confusion matrix using heatmap

import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred,digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

