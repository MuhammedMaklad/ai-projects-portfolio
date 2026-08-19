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
from sklearn.feature_selection import RFECV

# %% colab={"base_uri": "https://localhost:8080/"} id="9wwK1vQS3z5g" executionInfo={"status": "ok", "timestamp": 1722667881992, "user_tz": -180, "elapsed": 19791, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="7ae35ecf-eb2e-4f95-972a-a3feebda3cc9"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="y6rZHL6k31d8" executionInfo={"status": "ok", "timestamp": 1722667882464, "user_tz": -180, "elapsed": 476, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8c5b2ca4-61d8-48af-fa29-67381748fe37"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/Student Success

# %% [markdown] id="5a1c257f"
# ### Read the dataset

# %% papermill={"duration": 0.070758, "end_time": "2022-01-19T01:39:23.736923", "exception": false, "start_time": "2022-01-19T01:39:23.666165", "status": "completed"} id="b25c241b"
data = pd.read_csv("dataset.csv", sep=";")

# %% papermill={"duration": 0.074315, "end_time": "2022-01-19T01:39:23.861711", "exception": false, "start_time": "2022-01-19T01:39:23.787396", "status": "completed"} id="6df63fbd" outputId="976ba306-1550-45b8-8385-14b17e267a52" colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"status": "ok", "timestamp": 1722667882869, "user_tz": -180, "elapsed": 8, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.head()

# %% papermill={"duration": 0.059785, "end_time": "2022-01-19T01:39:24.073298", "exception": false, "start_time": "2022-01-19T01:39:24.013513", "status": "completed"} id="83503634" outputId="fac14a0b-5596-40d9-81e3-83f35fc24d07" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722534538113, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.shape

# %% papermill={"duration": 0.078689, "end_time": "2022-01-19T01:39:24.203703", "exception": false, "start_time": "2022-01-19T01:39:24.125014", "status": "completed"} id="06963df7" outputId="1e66866f-42c7-49a9-e338-ca1fdcac6ab8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722534538113, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.info()

# %% papermill={"duration": 0.094948, "end_time": "2022-01-19T01:39:24.349908", "exception": false, "start_time": "2022-01-19T01:39:24.254960", "status": "completed"} id="100e5863" outputId="da711806-7943-4ddf-9e47-69a15f5964a4" colab={"base_uri": "https://localhost:8080/", "height": 495} executionInfo={"status": "ok", "timestamp": 1722534538114, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.describe(include="all")

# %% [markdown] id="07d6aabf"
# ### check if the dataset has null values

# %% papermill={"duration": 0.06262, "end_time": "2022-01-19T01:39:24.464107", "exception": false, "start_time": "2022-01-19T01:39:24.401487", "status": "completed"} id="619b37db" outputId="23150d8e-2782-47e9-e6b8-cb54ec720e74" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722534538114, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.isnull().sum()

# %% [markdown] id="196311a6"
# ### Count the number of every class in the output

# %% papermill={"duration": 0.227446, "end_time": "2022-01-19T01:39:26.564642", "exception": false, "start_time": "2022-01-19T01:39:26.337196", "status": "completed"} id="4fd223d5" outputId="736cae61-4987-4f9c-d1ef-3d56d2ca8102" colab={"base_uri": "https://localhost:8080/", "height": 752} executionInfo={"status": "ok", "timestamp": 1722534538414, "user_tz": -180, "elapsed": 315, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
sns.countplot(data.Target, palette="bwr")
plt.show()
data.Target.value_counts(normalize=True)

# %% [markdown] id="dfd2ccb0"
# ### Draw the correlation between the attributes

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

# %% papermill={"duration": 0.081611, "end_time": "2022-01-19T01:39:34.840205", "exception": false, "start_time": "2022-01-19T01:39:34.758594", "status": "completed"} colab={"base_uri": "https://localhost:8080/", "height": 307} id="2640555e" executionInfo={"status": "ok", "timestamp": 1722534538805, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="3278da61-09bd-4e15-ad87-03c8cfcac814"
X.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 495} id="sFBuuUEdAigR" executionInfo={"status": "ok", "timestamp": 1722668056414, "user_tz": -180, "elapsed": 422, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="eeb5e7b4-a095-4aa4-99ef-7050da68f133"
X[:100]

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} id="855e83f2" executionInfo={"status": "ok", "timestamp": 1722534538805, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="908efb1a-f851-4148-aa2c-d1a6bfe79486"
y.head()

# %% [markdown] id="daed8339"
# ## Data scaling

# %% id="dc204ef3"
from sklearn import preprocessing
X= preprocessing.StandardScaler().fit_transform(X)

# %% id="vi9YLXduDW9X"
# Convert X to Dataframe with original columns names

import pandas as pd
X = pd.DataFrame(X, columns=data.drop("Target",axis=1).columns)


# %% [markdown] id="7451e7ec"
# ## Data splitting

# %% id="d2e48059"
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=10)

# %% [markdown] papermill={"duration": 0.068839, "end_time": "2022-01-19T01:39:35.132728", "exception": false, "start_time": "2022-01-19T01:39:35.063889", "status": "completed"} id="83ff4275"
# ## Build models with RFS

# %% [markdown] papermill={"duration": 0.065493, "end_time": "2022-01-19T01:39:35.678299", "exception": false, "start_time": "2022-01-19T01:39:35.612806", "status": "completed"} id="b6b3ab59"
# ## Logistic Regression Model

# %% [markdown] id="x8rTGlOTpqjD"
# **Recursive feature elimination**
#
# **Initial Training:** Train a model using all features and evaluate its performance using cross-validation.
#
# **Feature Ranking:** Rank features based on their importance scores from the model.
#
# **Feature Elimination:** Remove the least important feature(s).
#
# **Re-Evaluation:** Re-train the model with the remaining features and evaluate using cross-validation.
#
# **Repeat:** Repeat steps 2-4 until the optimal number of features is reached, yielding the best cross-validation score.

# %% papermill={"duration": 3.120202, "end_time": "2022-01-19T01:39:38.863804", "exception": false, "start_time": "2022-01-19T01:39:35.743602", "status": "completed"} id="0dcac604"
LR=LogisticRegression(max_iter=1000)

rfe = RFECV(estimator=LR, step=20,cv=5)
rfe = rfe.fit(X_train, y_train)
y_pred=rfe.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="YAv8IQvQDKbU" executionInfo={"status": "ok", "timestamp": 1722534548847, "user_tz": -180, "elapsed": 5, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="3aaf8780-70d2-4e82-fc26-25a10dd3dc8a"
# Need to know the best features choosen in the rfe

print('Optimal number of features: {}'.format(rfe.n_features_))
print('Selected features: {}'.format(list(X.columns[rfe.support_])))


# %% [markdown] id="5d182a34"
# ### Metrics calculation

# %% id="2eb737ae" outputId="5ac0bbf3-d0a7-45ad-9783-aa3eea75345e" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722534618471, "user_tz": -180, "elapsed": 828, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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


# %% [markdown] papermill={"duration": 0.068996, "end_time": "2022-01-19T01:39:48.918816", "exception": false, "start_time": "2022-01-19T01:39:48.849820", "status": "completed"} id="2751fc1b"
# ## Random Forest implementation

# %% papermill={"duration": 4.941656, "end_time": "2022-01-19T01:39:53.930617", "exception": false, "start_time": "2022-01-19T01:39:48.988961", "status": "completed"} id="1eba2000" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722534698797, "user_tz": -180, "elapsed": 9069, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e626aba7-edba-4eda-bff7-d4e212d4b6cd"
from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
rf_3 = RandomForestClassifier()
rfe = RFECV(estimator=rf_3, step=20,cv=5)
rfe = rfe.fit(X_train, y_train)
y_pred=rfe.predict(X_test)

print('Optimal number of features: {}'.format(rfe.n_features_))
print('Selected features: {}'.format(list(X.columns[rfe.support_])))

# %% papermill={"duration": 0.167409, "end_time": "2022-01-19T01:39:54.168655", "exception": false, "start_time": "2022-01-19T01:39:54.001246", "status": "completed"} id="be21b196" outputId="c176b214-a495-4c2f-e32e-79aea2c8cb6d" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722534701270, "user_tz": -180, "elapsed": 905, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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


# %% [markdown] id="43ca72d0"
# ## XGBoost Classifier

# %% id="491bb18b" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1722534932259, "user_tz": -180, "elapsed": 11206, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="55263512-172c-4ee5-b6d3-6beea390ec78"
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

rfe = RFECV(estimator=xgb, step=20,cv=5)
rfe = rfe.fit(X_train, y_train)
y_pred=rfe.predict(X_test)



# %% colab={"base_uri": "https://localhost:8080/"} id="JYUhmhP1GxzK" executionInfo={"status": "ok", "timestamp": 1722535326860, "user_tz": -180, "elapsed": 943, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="64f44229-6eee-42f7-efb5-b89638d1a74f"
print('Optimal number of features: {}'.format(rfe.n_features_))
print('Selected features: {}'.format(list(X.columns[rfe.support_])))

# %% id="27d9a51d" outputId="7273b666-8a5b-45a3-fae7-4e6ee425dcc2" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722535298806, "user_tz": -180, "elapsed": 1859, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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


# %% [markdown] id="696d6ebb"
# ## GradientBoostingClassifier

# %% id="a0441d03"
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
GBC=GradientBoostingClassifier()

rfe = RFECV(estimator=GBC, step=1,cv=5)
rfe = rfe.fit(X_train, y_train)
y_pred=rfe.predict(X_test)

print('Optimal number of features: {}'.format(rfe.n_features_))
print('Selected features: {}'.format(list(X.columns[rfe.support_])))

# %% id="1cbdc1db" outputId="468706f8-6bef-4bd8-b2ef-1fcb1a831a38" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1722535610726, "user_tz": -180, "elapsed": 675, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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


# %% [markdown] id="3192ea68"
# ## SVC

# %% id="8c590426"
from sklearn.svm import SVC
svm_4 = SVC(kernel='linear', probability=True)
svmcv = RFECV(estimator=svm_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
svmcv = svmcv.fit(X_train,y_train)
y_pred=svmcv.predict(X_test)

print('Optimal number of features: {}'.format(svmcv.n_features_))
print('Selected features: {}'.format(list(X.columns[svmcv.support_])))


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="aa0e2255" executionInfo={"status": "ok", "timestamp": 1722536310733, "user_tz": -180, "elapsed": 736, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="377f750b-2ff9-42c3-efc7-ee35d964e6d4"
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

# Transform X_test to have the same number of features as the trained model
X_test_transformed = svmcv.transform(X_test)

y_pred_proba = svmcv.estimator_.predict_proba(X_test_transformed)[:, 1] # Access the underlying estimator and predict on transformed data
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


# %% [markdown] id="TDdo84CSu33v"
# ## Stacking Classifier

# %% colab={"base_uri": "https://localhost:8080/"} id="cVdHRyvpzc5w" executionInfo={"status": "ok", "timestamp": 1722536417268, "user_tz": -180, "elapsed": 84983, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="4b37d40f-84a8-41fd-84cb-f290413bd288"
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# define the base models
level0 = list()
level0.append(('LR', LogisticRegression()))
level0.append(('KNN', KNeighborsClassifier()))
level0.append(('DT', DecisionTreeClassifier()))
level0.append(('SVM', SVC()))
level0.append(('MLP', MLPClassifier(max_iter=1000)))
level0.append(('SGD',SGDClassifier()))
level0.append(('XGB',XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)))
level0.append(('Adaboost',AdaBoostClassifier(base_estimator=DecisionTreeClassifier())))

# define meta learner model
level1= LogisticRegression()
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="ZTNgtpTO-REx" executionInfo={"status": "ok", "timestamp": 1722536418349, "user_tz": -180, "elapsed": 1091, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="3b56e8ae-b330-4655-e9b7-cb4169fb6352"
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
y_pred_proba = model.predict_proba(X_test)[:, 1]
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

