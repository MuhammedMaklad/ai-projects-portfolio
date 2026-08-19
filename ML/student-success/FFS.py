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

# %% papermill={"duration": 1.53618, "end_time": "2022-01-19T01:39:23.515347", "exception": false, "start_time": "2022-01-19T01:39:21.979167", "status": "completed"} id="72e78d72" executionInfo={"status": "ok", "timestamp": 1727419319520, "user_tz": -180, "elapsed": 4740, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% colab={"base_uri": "https://localhost:8080/"} id="7u4Y4aEU2yCH" executionInfo={"status": "ok", "timestamp": 1727419344151, "user_tz": -180, "elapsed": 24633, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="257a10dd-28c2-459f-91f5-ee4c510a4d7e"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="UsqaEtSv2xNU" executionInfo={"status": "ok", "timestamp": 1727419345505, "user_tz": -180, "elapsed": 1363, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d7fc35f3-957e-4e73-a750-f23940d9f381"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/Student Success

# %% [markdown] id="5a1c257f"
# ### Read the dataset

# %% papermill={"duration": 0.070758, "end_time": "2022-01-19T01:39:23.736923", "exception": false, "start_time": "2022-01-19T01:39:23.666165", "status": "completed"} id="b25c241b" executionInfo={"status": "ok", "timestamp": 1727419346313, "user_tz": -180, "elapsed": 809, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data = pd.read_csv("dataset.csv", sep=";")

# %% papermill={"duration": 0.074315, "end_time": "2022-01-19T01:39:23.861711", "exception": false, "start_time": "2022-01-19T01:39:23.787396", "status": "completed"} id="6df63fbd" outputId="8e405936-31fb-41a3-8021-1dc33ffb09b3" colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"status": "ok", "timestamp": 1727419346313, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.head()

# %% papermill={"duration": 0.059785, "end_time": "2022-01-19T01:39:24.073298", "exception": false, "start_time": "2022-01-19T01:39:24.013513", "status": "completed"} id="83503634" outputId="94d05725-02cb-4787-9904-adbabc002e7c" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1727419346313, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.shape

# %% papermill={"duration": 0.078689, "end_time": "2022-01-19T01:39:24.203703", "exception": false, "start_time": "2022-01-19T01:39:24.125014", "status": "completed"} id="06963df7" outputId="f91c0e16-8b60-464a-ec16-ab0534b9da65" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1727419346313, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.info()

# %% papermill={"duration": 0.094948, "end_time": "2022-01-19T01:39:24.349908", "exception": false, "start_time": "2022-01-19T01:39:24.254960", "status": "completed"} id="100e5863" outputId="0a20550c-4f01-42f1-d1b1-512d9376fdde" colab={"base_uri": "https://localhost:8080/", "height": 495} executionInfo={"status": "ok", "timestamp": 1727419346314, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.describe(include="all")

# %% [markdown] id="07d6aabf"
# ### check if the dataset has null values

# %% papermill={"duration": 0.06262, "end_time": "2022-01-19T01:39:24.464107", "exception": false, "start_time": "2022-01-19T01:39:24.401487", "status": "completed"} id="619b37db" outputId="3e9b373d-791a-4a1b-b1aa-77e9cf13010c" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1727419346314, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.isnull().sum()

# %% [markdown] id="196311a6"
# ### Count the number of every class in the output

# %% papermill={"duration": 0.227446, "end_time": "2022-01-19T01:39:26.564642", "exception": false, "start_time": "2022-01-19T01:39:26.337196", "status": "completed"} id="4fd223d5" outputId="0fd5db94-0df4-449c-ddd3-1bdcb8bab9b0" colab={"base_uri": "https://localhost:8080/", "height": 748} executionInfo={"status": "ok", "timestamp": 1727419347078, "user_tz": -180, "elapsed": 777, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
sns.countplot(data.Target, palette="bwr")
plt.show()
data.Target.value_counts(normalize=True)

# %% [markdown] id="1dec189e"
# ## Convert Dropout =0, Graduate=1, Enrolled=3

# %% id="0bebdbcc" executionInfo={"status": "ok", "timestamp": 1727419347078, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data = pd.read_csv("dataset.csv", sep=";")
data["Target"]=data.Target.map(dict( Dropout =0, Graduate=1, Enrolled=2))

# %% [markdown] id="683e404c"
# ### select only Dropout and Graduate because the number of Enrolled students is very low

# %% id="62cf8a44" executionInfo={"status": "ok", "timestamp": 1727419347079, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data = data[data['Target'] != 2]

# %% [markdown] id="1d82725c"
# ### Read the input and the output

# %% papermill={"duration": 0.073658, "end_time": "2022-01-19T01:39:34.690351", "exception": false, "start_time": "2022-01-19T01:39:34.616693", "status": "completed"} id="011f71a6" executionInfo={"status": "ok", "timestamp": 1727419347079, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X = data.drop("Target",axis=1)
y = data["Target"]

# %% papermill={"duration": 0.081611, "end_time": "2022-01-19T01:39:34.840205", "exception": false, "start_time": "2022-01-19T01:39:34.758594", "status": "completed"} id="2640555e" outputId="67fdce5b-994e-45f1-81b1-2efc1ac8612c" colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"status": "ok", "timestamp": 1727419347079, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X.head()

# %% id="855e83f2" outputId="f6d5eb13-6f1e-4174-b749-74150365517e" colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"status": "ok", "timestamp": 1727419347079, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
y.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="wFhRSnj33SS8" executionInfo={"status": "ok", "timestamp": 1727419488543, "user_tz": -180, "elapsed": 141480, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="2e3a7a6e-c006-4d8e-d8e0-23cc35e3c378"
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn,direction='forward')
X_new=sfs.fit_transform(X, y)
print(X.columns[sfs.get_support()].to_list())
X=X_new

# %% colab={"base_uri": "https://localhost:8080/", "height": 853} id="FZtOiIRVJZcE" executionInfo={"status": "ok", "timestamp": 1727419490644, "user_tz": -180, "elapsed": 2104, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b02b52e8-df83-421b-f5e8-5e91ac87d951"
data=np.column_stack([X, y])
data = pd.DataFrame(data)
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

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

# %% [markdown] papermill={"duration": 0.068374, "end_time": "2022-01-19T01:39:41.850613", "exception": false, "start_time": "2022-01-19T01:39:41.782239", "status": "completed"} id="bf112f5f"
# ## Decision Tree

# %% papermill={"duration": 5.668179, "end_time": "2022-01-19T01:39:47.587646", "exception": false, "start_time": "2022-01-19T01:39:41.919467", "status": "completed"} id="db51e325" outputId="5900c128-c52b-4a29-fe6a-1ce8a23d8791" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722680137411, "user_tz": -180, "elapsed": 18913, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% papermill={"duration": 0.105385, "end_time": "2022-01-19T01:39:47.762516", "exception": false, "start_time": "2022-01-19T01:39:47.657131", "status": "completed"} id="3f3ea1b5" outputId="809e1d56-5348-4424-e8c8-27428376aab3" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722680137412, "user_tz": -180, "elapsed": 12, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% [markdown] papermill={"duration": 0.068996, "end_time": "2022-01-19T01:39:48.918816", "exception": false, "start_time": "2022-01-19T01:39:48.849820", "status": "completed"} id="2751fc1b"
# ## Random Forest implementation

# %% papermill={"duration": 4.941656, "end_time": "2022-01-19T01:39:53.930617", "exception": false, "start_time": "2022-01-19T01:39:48.988961", "status": "completed"} id="1eba2000" outputId="8c929035-b81e-4bfd-a5d9-d9cc74fdd966" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722680141168, "user_tz": -180, "elapsed": 3765, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
model = RandomForestClassifier(random_state=42)
params = {'n_estimators': np.arange(100,500,100)}
cv_rf = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1, cv=5,verbose=3)
cv_rf.fit(X_train, y_train)
best_RF=cv_rf.best_estimator_
y_pred = best_RF.predict(X_test)
print(cv_rf.best_params_, cv_rf.best_score_)

# %% papermill={"duration": 0.167409, "end_time": "2022-01-19T01:39:54.168655", "exception": false, "start_time": "2022-01-19T01:39:54.001246", "status": "completed"} id="be21b196" outputId="8ba75fc3-0108-435b-83f5-657dca05811a" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722680142092, "user_tz": -180, "elapsed": 928, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% id="491bb18b" outputId="0b46c4f9-a9f2-47ab-a875-f0dc9d3f83ce" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722680149568, "user_tz": -180, "elapsed": 7478, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )

random_search.fit(X_train,y_train)
best_XGB=random_search.best_estimator_
y_pred = best_XGB.predict(X_test)
print(random_search.best_params_, random_search.best_score_)


# %% id="27d9a51d" outputId="756f4c44-b701-4ff4-d512-c07c4d0e8751" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722680150222, "user_tz": -180, "elapsed": 665, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% id="fd470897"
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

# %% id="06b09c77" outputId="e05f55f6-7b10-4ead-ecf2-94f52387f71a" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722680518625, "user_tz": -180, "elapsed": 910, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% id="a60b691c" outputId="270c69c9-f44d-45e6-fe47-dbe1cc284508" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722680536080, "user_tz": -180, "elapsed": 17459, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV


eclf = VotingClassifier(estimators=[
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression()),
    ], voting='soft')

#Use the key for the classifier followed by __ and the attribute
params = {'lr__C': [1.0, 100.0],
      'svm__C': [2,3,4],}

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5,n_jobs=-1, verbose=3)

grid.fit(X_train,y_train)
print (grid.best_params_)
best_VC=grid.best_estimator_
y_pred=best_VC.predict(X_test)

# %% id="0331d0db" outputId="f613b763-d8e6-4eac-f604-b8cfdf63610b" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722680536645, "user_tz": -180, "elapsed": 575, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% colab={"base_uri": "https://localhost:8080/"} id="tSFmnxIwi5b3" executionInfo={"status": "ok", "timestamp": 1722680552617, "user_tz": -180, "elapsed": 15973, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9325b621-7c34-4fec-9a8d-38c0064da59e"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 573} id="1X8Igtd-n9mf" executionInfo={"status": "ok", "timestamp": 1722680553064, "user_tz": -180, "elapsed": 449, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0b02f7c6-ea71-44c4-e3ba-ff77a90982a7"
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

# %% colab={"base_uri": "https://localhost:8080/"} id="9MwKxAaGEl8i" executionInfo={"status": "ok", "timestamp": 1722680649091, "user_tz": -180, "elapsed": 96033, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="795653c5-9511-4cd2-9f10-6a7816983f4b"
from sklearn.svm import SVC
model = SVC(probability=True)
params = {'kernel': ['linear', 'rbf'], 'C': np.arange(0.5,1,0.1), 'gamma': np.arange(0.01,0.05,0.01)}
cv_svm = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1,cv=5)
cv_svm.fit(X_train, y_train)
best_SVM=cv_svm.best_estimator_
y_pred = best_SVM.predict(X_test)
print(cv_svm.best_params_, cv_svm.best_score_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="iW0JYKG6oB9d" executionInfo={"status": "ok", "timestamp": 1722680649699, "user_tz": -180, "elapsed": 613, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="fe54b65e-9bfb-4637-9c5b-e24dedfbc83e"
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
