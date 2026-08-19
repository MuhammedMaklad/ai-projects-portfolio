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

# %% executionInfo={"elapsed": 322, "status": "ok", "timestamp": 1726822948230, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="72e78d72" papermill={"duration": 1.53618, "end_time": "2022-01-19T01:39:23.515347", "exception": false, "start_time": "2022-01-19T01:39:21.979167", "status": "completed"}
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1904, "status": "ok", "timestamp": 1726822950501, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="7u4Y4aEU2yCH" outputId="b77091a6-386a-4de6-bc98-3a96c6336694"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1726822950501, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="UsqaEtSv2xNU" outputId="13232b53-8c17-4ed4-d1fa-a0649a9cb154"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/Student Success

# %% [markdown] id="5a1c257f"
# ### Read the dataset

# %% executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1726822950501, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="b25c241b" papermill={"duration": 0.070758, "end_time": "2022-01-19T01:39:23.736923", "exception": false, "start_time": "2022-01-19T01:39:23.666165", "status": "completed"}
data = pd.read_csv("dataset.csv", sep=";")

# %% colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"elapsed": 16, "status": "ok", "timestamp": 1726822950501, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="6df63fbd" outputId="b57db2a8-b739-4f75-bdd3-e564e228ac09" papermill={"duration": 0.074315, "end_time": "2022-01-19T01:39:23.861711", "exception": false, "start_time": "2022-01-19T01:39:23.787396", "status": "completed"}
data.head(10)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1726822950501, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="83503634" outputId="8ab2e604-6745-4398-d333-29ac71da8a07" papermill={"duration": 0.059785, "end_time": "2022-01-19T01:39:24.073298", "exception": false, "start_time": "2022-01-19T01:39:24.013513", "status": "completed"}
data.shape

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 16, "status": "ok", "timestamp": 1726822950502, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="06963df7" outputId="81c73661-d46f-499d-db39-1a6cac89e71e" papermill={"duration": 0.078689, "end_time": "2022-01-19T01:39:24.203703", "exception": false, "start_time": "2022-01-19T01:39:24.125014", "status": "completed"}
data.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 495} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1726822950502, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="100e5863" outputId="1a950e6e-b352-4051-bb14-4af049914163" papermill={"duration": 0.094948, "end_time": "2022-01-19T01:39:24.349908", "exception": false, "start_time": "2022-01-19T01:39:24.254960", "status": "completed"}
data.describe(include="all")

# %% [markdown] id="07d6aabf"
# ### check if the dataset has null values

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 312, "status": "ok", "timestamp": 1726822950801, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="619b37db" outputId="b69c5a2c-7b2d-4646-ffb5-9a6df0e0bbbc" papermill={"duration": 0.06262, "end_time": "2022-01-19T01:39:24.464107", "exception": false, "start_time": "2022-01-19T01:39:24.401487", "status": "completed"}
data.isnull().sum()

# %% [markdown] id="196311a6"
# ### Count the number of every class in the output

# %% colab={"base_uri": "https://localhost:8080/", "height": 748} executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1726822950801, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4fd223d5" outputId="95b6e143-831c-4bca-8622-22f0a7d56a26" papermill={"duration": 0.227446, "end_time": "2022-01-19T01:39:26.564642", "exception": false, "start_time": "2022-01-19T01:39:26.337196", "status": "completed"}
sns.countplot(data.Target, palette="bwr")
plt.show()
data.Target.value_counts(normalize=True)

# %% [markdown] id="1dec189e"
# ## Convert Dropout =0, Graduate=1, Enrolled=3

# %% executionInfo={"elapsed": 340, "status": "ok", "timestamp": 1726822951129, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="0bebdbcc"
data = pd.read_csv("dataset.csv", sep=";")
data["Target"]=data.Target.map(dict( Dropout =0, Graduate=1, Enrolled=2))

# %% [markdown] id="683e404c"
# ### select only Dropout and Graduate because the number of Enrolled students is very low

# %% executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1726822951130, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="62cf8a44"
data = data[data['Target'] != 2]

# %% [markdown] id="1d82725c"
# ### Read the input and the output

# %% executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1726822951130, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="011f71a6" papermill={"duration": 0.073658, "end_time": "2022-01-19T01:39:34.690351", "exception": false, "start_time": "2022-01-19T01:39:34.616693", "status": "completed"}
X = data.drop("Target",axis=1)
y = data["Target"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1726822951130, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="2640555e" outputId="ea35bbbe-a12a-49ff-959d-115594fc5315" papermill={"duration": 0.081611, "end_time": "2022-01-19T01:39:34.840205", "exception": false, "start_time": "2022-01-19T01:39:34.758594", "status": "completed"}
X.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1726822951130, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="855e83f2" outputId="0eb89518-1f9f-4949-9132-ee0a2e5690ed"
y.head()

# %% executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1726822951130, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="_IBxhVbXtTt3"
# prompt: apply min max scaler on X and convert the output to datafram with the colimns names

import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # 0 to 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1726822951130, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="wFhRSnj33SS8" outputId="38ed84a0-83f8-4cb1-9977-65a5cfaf8ec0"
from sklearn.feature_selection import SelectKBest, chi2

SelKpest = SelectKBest(chi2, k=20)
X_new=SelKpest.fit_transform(X_scaled,y)
print("the new shape is : ", X_new.shape)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1726822951130, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="s518uq-Up6kp" outputId="43041848-122d-460c-e0c6-ec3064b160f9"
SelKpest.get_support()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1726822951130, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Kf3gSv7ip9cj" outputId="3ab98766-8c1f-49d1-a664-0b44c15e83cc"
print(X.columns[SelKpest.get_support()].to_list())
X=X_new


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 2691, "status": "ok", "timestamp": 1726822953802, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="FZtOiIRVJZcE" outputId="eee44fb6-1083-4e0b-d89d-813c0979741e"
data=np.column_stack([X, y])
data = pd.DataFrame(data)
f,ax = plt.subplots(figsize=(14,14))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

# %% [markdown] id="daed8339"
# ## Data scaling

# %% [markdown] id="7451e7ec"
# ## Data splitting

# %% executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1726822953802, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="d2e48059"
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=10)

# %% [markdown] id="83ff4275" papermill={"duration": 0.068839, "end_time": "2022-01-19T01:39:35.132728", "exception": false, "start_time": "2022-01-19T01:39:35.063889", "status": "completed"}
# ## Build models

# %% [markdown] id="bf112f5f" papermill={"duration": 0.068374, "end_time": "2022-01-19T01:39:41.850613", "exception": false, "start_time": "2022-01-19T01:39:41.782239", "status": "completed"}
# ## Decision Tree

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9018, "status": "ok", "timestamp": 1726822962809, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="db51e325" outputId="f8fc96da-d88d-4f26-e558-3ebb382b180e" papermill={"duration": 5.668179, "end_time": "2022-01-19T01:39:47.587646", "exception": false, "start_time": "2022-01-19T01:39:41.919467", "status": "completed"}
model = DecisionTreeClassifier(random_state=42)
params = {'splitter': ["best", "random"],
          'max_depth': np.arange(1,10,1),
          'min_samples_leaf': np.arange(1,5,1),
          'min_samples_split': np.arange(2,5,1)
         }
cv_dt = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1,cv=5)
cv_dt.fit(X_train, y_train)

best_DT=cv_dt.best_estimator_
y_pred = best_DT.predict(X_test)
print(cv_dt.best_params_, cv_dt.best_score_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 880, "status": "ok", "timestamp": 1726822963684, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="3f3ea1b5" outputId="6a63be5e-b69a-48bc-d46a-bbe95a3a7651" papermill={"duration": 0.105385, "end_time": "2022-01-19T01:39:47.762516", "exception": false, "start_time": "2022-01-19T01:39:47.657131", "status": "completed"}
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
y_pred_proba = best_DT.predict_proba(X_test)[:, 1]
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

# %% [markdown] id="2751fc1b" papermill={"duration": 0.068996, "end_time": "2022-01-19T01:39:48.918816", "exception": false, "start_time": "2022-01-19T01:39:48.849820", "status": "completed"}
# ## Random Forest implementation

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20213, "status": "ok", "timestamp": 1726822983894, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1eba2000" outputId="52df9ca5-ce6c-49d3-d92f-9b9b89b57290" papermill={"duration": 4.941656, "end_time": "2022-01-19T01:39:53.930617", "exception": false, "start_time": "2022-01-19T01:39:48.988961", "status": "completed"}
model = RandomForestClassifier(random_state=42)
params = {'n_estimators': np.arange(100,500,100)}
cv_rf = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1, cv=5,verbose=3)
cv_rf.fit(X_train, y_train)
best_RF=cv_rf.best_estimator_
y_pred = best_RF.predict(X_test)
print(cv_rf.best_params_, cv_rf.best_score_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 508, "status": "ok", "timestamp": 1726822984397, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="be21b196" outputId="bef335fb-b963-4a44-cfe4-ffdc8cc9d1b4" papermill={"duration": 0.167409, "end_time": "2022-01-19T01:39:54.168655", "exception": false, "start_time": "2022-01-19T01:39:54.001246", "status": "completed"}
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
y_pred_proba = best_RF.predict_proba(X_test)[:, 1]
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6166, "status": "ok", "timestamp": 1726822990552, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="491bb18b" outputId="3ab9df97-a08c-488b-86f7-8a54d79af72a"
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy',
                                   n_jobs=-1, cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )

random_search.fit(X_train,y_train)
best_XGB=random_search.best_estimator_
y_pred = best_XGB.predict(X_test)
print(random_search.best_params_, random_search.best_score_)


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 834, "status": "ok", "timestamp": 1726822991382, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="27d9a51d" outputId="241be05e-b9db-4f6c-bd8d-f8889f8e7866"
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
y_pred_proba = best_XGB.predict_proba(X_test)[:, 1]
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

# %% [markdown] id="ccb106db"
# ## MLP implementation

# %% executionInfo={"elapsed": 359384, "status": "ok", "timestamp": 1726823350764, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="fd470897"
from sklearn.neural_network import MLPClassifier
mlp_gs = MLPClassifier(max_iter=300)
parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels
best_MLP=clf.best_estimator_
y_pred=best_MLP.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 886, "status": "ok", "timestamp": 1726823351642, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="06b09c77" outputId="f669da37-748f-413a-d8ab-0b5c09e91be8"
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
y_pred_proba = best_MLP.predict_proba(X_test)[:, 1]
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

# %% [markdown] id="qgWLiOxHN0WG"
# ## VotingClassifier

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 120473, "status": "ok", "timestamp": 1726823661611, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="a60b691c" outputId="d6deb65e-9862-479e-9a80-5f389c453fe1"
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


eclf = VotingClassifier(estimators=[
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier(n_estimators= 300)),
    ('dt', DecisionTreeClassifier(max_depth=6, min_samples_leaf=1, min_samples_split=2, splitter= 'random')),
    ('xgb', XGBClassifier(subsample= 0.8, min_child_weight= 5, max_depth= 5, gamma= 1, colsample_bytree= 0.8)),
    ('mlp', MLPClassifier(activation= 'relu', alpha= 0.0001, hidden_layer_sizes= (20,), learning_rate= 'adaptive', solver= 'adam')),
    ('bagging',BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=6, min_samples_leaf=1, min_samples_split=2, splitter= 'random'))),
    ('adaboost',AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6, min_samples_leaf=1, min_samples_split=2, splitter= 'random')))
    ], voting='soft')

#Use the key for the classifier followed by __ and the attribute
params = {'lr__C': [1.0, 100.0],
      'svm__C': [2,3,4],
          }

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5,n_jobs=-1, verbose=3)

grid.fit(X_train,y_train)
print (grid.best_params_)
best_VC=grid.best_estimator_
y_pred=best_VC.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 1242, "status": "ok", "timestamp": 1726823663877, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="0331d0db" outputId="be1f85b5-9521-40d0-e7f6-8f64c08a5c79"
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
y_pred_proba = best_VC.predict_proba(X_test)[:, 1]
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

# %% [markdown] id="honrPIv3txEi"
# ## Stacking Classifier

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 19142, "status": "ok", "timestamp": 1726823683018, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="tSFmnxIwi5b3" outputId="60328b48-e160-4b3e-c3ef-6e1315e9a054"
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
level0.append(('DT', DecisionTreeClassifier(max_depth=6, min_samples_leaf=1, min_samples_split=2, splitter= 'random')))
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 566} executionInfo={"elapsed": 338, "status": "ok", "timestamp": 1726823683355, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1X8Igtd-n9mf" outputId="59a32b54-a022-4e03-aa82-bbf1f4875173"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 130752, "status": "ok", "timestamp": 1726823814099, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="9MwKxAaGEl8i" outputId="33da73a2-dd88-4e96-e64c-3f27b3c0858a"
from sklearn.svm import SVC
model = SVC(probability=True)
params = {'kernel': ['linear', 'rbf'], 'C': np.arange(0.5,1,0.1), 'gamma': np.arange(0.01,0.05,0.01)}
cv_svm = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1,cv=5)
cv_svm.fit(X_train, y_train)
best_SVM=cv_svm.best_estimator_
y_pred = best_SVM.predict(X_test)
print(cv_svm.best_params_, cv_svm.best_score_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 388, "status": "ok", "timestamp": 1726823814481, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="iW0JYKG6oB9d" outputId="6da819b6-2283-47c8-d3c1-8eadb40d697a"
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
y_pred_proba = best_SVM.predict_proba(X_test)[:, 1]
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

# %% [markdown] id="ZQjLN7Hg9zLH"
# Adaboost , XGboost = 92.5%
