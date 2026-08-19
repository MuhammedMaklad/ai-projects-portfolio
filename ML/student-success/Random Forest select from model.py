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

# %% colab={"base_uri": "https://localhost:8080/"} id="7u4Y4aEU2yCH" executionInfo={"status": "ok", "timestamp": 1722678129588, "user_tz": -180, "elapsed": 19955, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="170f5a85-e643-446b-8251-197974e1c0fb"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="UsqaEtSv2xNU" executionInfo={"status": "ok", "timestamp": 1722678130012, "user_tz": -180, "elapsed": 427, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9d8b6f47-1304-4769-fd0c-b2aaf0b16e9b"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/Student Success

# %% [markdown] id="5a1c257f"
# ### Read the dataset

# %% papermill={"duration": 0.070758, "end_time": "2022-01-19T01:39:23.736923", "exception": false, "start_time": "2022-01-19T01:39:23.666165", "status": "completed"} id="b25c241b"
data = pd.read_csv("dataset.csv", sep=";")

# %% papermill={"duration": 0.074315, "end_time": "2022-01-19T01:39:23.861711", "exception": false, "start_time": "2022-01-19T01:39:23.787396", "status": "completed"} id="6df63fbd" outputId="f446acad-4353-481b-970f-55eba5888475" colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"status": "ok", "timestamp": 1722678130657, "user_tz": -180, "elapsed": 345, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.head()

# %% papermill={"duration": 0.059785, "end_time": "2022-01-19T01:39:24.073298", "exception": false, "start_time": "2022-01-19T01:39:24.013513", "status": "completed"} id="83503634" outputId="7364604a-53fd-47d5-ef13-fb5b58973bab" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722678130657, "user_tz": -180, "elapsed": 9, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.shape

# %% papermill={"duration": 0.078689, "end_time": "2022-01-19T01:39:24.203703", "exception": false, "start_time": "2022-01-19T01:39:24.125014", "status": "completed"} id="06963df7" outputId="b3dd85d7-e609-40eb-e5dd-cb39be29a70d" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722678130657, "user_tz": -180, "elapsed": 8, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.info()

# %% papermill={"duration": 0.094948, "end_time": "2022-01-19T01:39:24.349908", "exception": false, "start_time": "2022-01-19T01:39:24.254960", "status": "completed"} id="100e5863" outputId="1ec4c32d-1cd2-4892-b189-eb4f12787323" colab={"base_uri": "https://localhost:8080/", "height": 495} executionInfo={"status": "ok", "timestamp": 1722678131169, "user_tz": -180, "elapsed": 519, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.describe(include="all")

# %% [markdown] id="07d6aabf"
# ### check if the dataset has null values

# %% papermill={"duration": 0.06262, "end_time": "2022-01-19T01:39:24.464107", "exception": false, "start_time": "2022-01-19T01:39:24.401487", "status": "completed"} id="619b37db" outputId="4c03cc86-aaa1-4e90-b9c3-9d31451deb70" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722678131169, "user_tz": -180, "elapsed": 8, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.isnull().sum()

# %% [markdown] id="196311a6"
# ### Count the number of every class in the output

# %% papermill={"duration": 0.227446, "end_time": "2022-01-19T01:39:26.564642", "exception": false, "start_time": "2022-01-19T01:39:26.337196", "status": "completed"} id="4fd223d5" outputId="84c1cf3f-c25e-4a08-dfe3-46c4a4263c57" colab={"base_uri": "https://localhost:8080/", "height": 752} executionInfo={"status": "ok", "timestamp": 1722678131863, "user_tz": -180, "elapsed": 701, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% papermill={"duration": 0.081611, "end_time": "2022-01-19T01:39:34.840205", "exception": false, "start_time": "2022-01-19T01:39:34.758594", "status": "completed"} id="2640555e" outputId="0757caf4-3755-47a6-f52c-d164ec20f8e0" colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"status": "ok", "timestamp": 1722678131863, "user_tz": -180, "elapsed": 10, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X.head()

# %% id="855e83f2" outputId="7b605913-507d-40d6-a1fb-074a84471067" colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"status": "ok", "timestamp": 1722678131863, "user_tz": -180, "elapsed": 9, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
y.head()

# %% [markdown] id="tJRzsrMjPdFW"
# every mode can give me importances for every feature beside the target.

# %% colab={"base_uri": "https://localhost:8080/"} id="wFhRSnj33SS8" executionInfo={"status": "ok", "timestamp": 1722678132218, "user_tz": -180, "elapsed": 364, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="7c581027-3ce5-4829-b440-db26c0cdaeff"
from sklearn.feature_selection import SelectFromModel

selFromModel=SelectFromModel(RandomForestClassifier(n_estimators = 50))
X_new=selFromModel.fit_transform(X,y)
print("the new shape is : ", X_new.shape)
print(X.columns[selFromModel.get_support()].to_list())
X=X_new


# %% colab={"base_uri": "https://localhost:8080/", "height": 700} id="FZtOiIRVJZcE" executionInfo={"status": "ok", "timestamp": 1722678134065, "user_tz": -180, "elapsed": 1850, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="2657a186-948b-433d-9be6-ad80cc2e9068"
data=np.column_stack([X, y])
data = pd.DataFrame(data)
f,ax = plt.subplots(figsize=(8,8))
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

# %% papermill={"duration": 5.668179, "end_time": "2022-01-19T01:39:47.587646", "exception": false, "start_time": "2022-01-19T01:39:41.919467", "status": "completed"} id="db51e325" outputId="a2a00975-40a0-47c5-fdda-e65dfbd67cb2" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722678151861, "user_tz": -180, "elapsed": 10779, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% papermill={"duration": 0.105385, "end_time": "2022-01-19T01:39:47.762516", "exception": false, "start_time": "2022-01-19T01:39:47.657131", "status": "completed"} id="3f3ea1b5" outputId="2a512b97-8bd8-4af3-a042-c043b4d03468" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722678152508, "user_tz": -180, "elapsed": 658, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% papermill={"duration": 4.941656, "end_time": "2022-01-19T01:39:53.930617", "exception": false, "start_time": "2022-01-19T01:39:48.988961", "status": "completed"} id="1eba2000" outputId="6a9fbca6-1526-4364-dd7c-1b112602e272" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722678251711, "user_tz": -180, "elapsed": 39462, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
model = RandomForestClassifier(random_state=42)
params = {'n_estimators': np.arange(100,500,100)}
cv_rf = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1, cv=5,verbose=3)
cv_rf.fit(X_train, y_train)
best_RF=cv_rf.best_estimator_
y_pred = best_RF.predict(X_test)
print(cv_rf.best_params_, cv_rf.best_score_)

# %% papermill={"duration": 0.167409, "end_time": "2022-01-19T01:39:54.168655", "exception": false, "start_time": "2022-01-19T01:39:54.001246", "status": "completed"} id="be21b196" outputId="711b3932-14ec-4ed6-d961-0c31e82e6826" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722678253423, "user_tz": -180, "elapsed": 1716, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% id="491bb18b" outputId="f98b0032-3754-4b35-ccda-956868b63e98" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722678315925, "user_tz": -180, "elapsed": 8364, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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


# %% id="27d9a51d" outputId="61f44cec-cbbd-4b10-c58f-a5281e28a017" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722678317522, "user_tz": -180, "elapsed": 1601, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% id="06b09c77" outputId="e368dab7-d8b9-409f-ed86-350d7fc409fa" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722678769019, "user_tz": -180, "elapsed": 558, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% id="a60b691c" outputId="f8904691-dcc1-4f8a-e3f1-ecea9bbd456f" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722678789467, "user_tz": -180, "elapsed": 19795, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% id="0331d0db" outputId="4fcd648f-281c-494f-b917-26b8416dda85" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722678789955, "user_tz": -180, "elapsed": 494, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% colab={"base_uri": "https://localhost:8080/"} id="tSFmnxIwi5b3" executionInfo={"status": "ok", "timestamp": 1722678807441, "user_tz": -180, "elapsed": 17496, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="13a40792-cdc1-497b-b642-dff57ff6eca4"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 573} id="1X8Igtd-n9mf" executionInfo={"status": "ok", "timestamp": 1722678807983, "user_tz": -180, "elapsed": 552, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ccf056dd-1e17-47d6-c3a5-e1295d396345"
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

# %% colab={"base_uri": "https://localhost:8080/"} id="9MwKxAaGEl8i" executionInfo={"status": "ok", "timestamp": 1722679138473, "user_tz": -180, "elapsed": 105042, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c173bfde-de24-44b3-b2aa-e47619141d55"
from sklearn.svm import SVC
model = SVC(probability=True)
params = {'kernel': ['linear', 'rbf'], 'C': np.arange(0.5,1,0.1), 'gamma': np.arange(0.01,0.05,0.01)}
cv_svm = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1,cv=5)
cv_svm.fit(X_train, y_train)
best_SVM=cv_svm.best_estimator_
y_pred = best_SVM.predict(X_test)
print(cv_svm.best_params_, cv_svm.best_score_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="iW0JYKG6oB9d" executionInfo={"status": "ok", "timestamp": 1722679139405, "user_tz": -180, "elapsed": 936, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="868a48c0-a309-4e72-90a6-c5512c41ab27"
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
