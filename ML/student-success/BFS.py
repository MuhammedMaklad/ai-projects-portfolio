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

# %% executionInfo={"elapsed": 4868, "status": "ok", "timestamp": 1722680524554, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="72e78d72"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 26147, "status": "ok", "timestamp": 1722680550699, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="7u4Y4aEU2yCH" outputId="e6075ce5-1abe-47ac-916c-5b6325c2635d"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 814, "status": "ok", "timestamp": 1722680551509, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="UsqaEtSv2xNU" outputId="ba562155-c96a-4709-b881-6b40e8a8e126"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/Student Success

# %% [markdown] id="5a1c257f"
# ### Read the dataset

# %% executionInfo={"elapsed": 1631, "status": "ok", "timestamp": 1722680553137, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="b25c241b"
data = pd.read_csv("dataset.csv", sep=";")

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"elapsed": 486, "status": "ok", "timestamp": 1722680553622, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="6df63fbd" outputId="e396adac-61d1-4f3c-d3e3-8003683976e8"
data.head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1722680553622, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="83503634" outputId="cb82ac6e-ce95-4189-8aa5-ab3bad8c8daf"
data.shape

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1722680553623, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="06963df7" outputId="81df6422-8626-4aa6-bc82-31fee36815fb"
data.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 495} executionInfo={"elapsed": 1177, "status": "ok", "timestamp": 1722680554790, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="100e5863" outputId="0ba89308-6e3d-4ff2-cf6a-5b77cb8c8181"
data.describe(include="all")

# %% [markdown] id="07d6aabf"
# ### check if the dataset has null values

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1722680554791, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="619b37db" outputId="efb358b6-c501-4bd0-9d68-bded3e0e23d1"
data.isnull().sum()

# %% [markdown] id="196311a6"
# ### Count the number of every class in the output

# %% colab={"base_uri": "https://localhost:8080/", "height": 752} executionInfo={"elapsed": 1769, "status": "ok", "timestamp": 1722680556552, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4fd223d5" outputId="c56061bd-6140-452a-dcf0-8782bd5f03cf"
sns.countplot(data.Target, palette="bwr")
plt.show()
data.Target.value_counts(normalize=True)

# %% [markdown] id="1dec189e"
# ## Convert Dropout =0, Graduate=1, Enrolled=3

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1722680556552, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="0bebdbcc"
data = pd.read_csv("dataset.csv", sep=";")
data["Target"]=data.Target.map(dict( Dropout =0, Graduate=1, Enrolled=2))

# %% [markdown] id="683e404c"
# ### select only Dropout and Graduate because the number of Enrolled students is very low

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1722680556552, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="62cf8a44"
data = data[data['Target'] != 2]

# %% [markdown] id="1d82725c"
# ### Read the input and the output

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1722680556552, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="011f71a6"
X = data.drop("Target",axis=1)
y = data["Target"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"elapsed": 422, "status": "ok", "timestamp": 1722680556972, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="2640555e" outputId="e56ed9a0-f6f6-40aa-b9f4-6bc8bf72468d"
X.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1722680556972, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="855e83f2" outputId="fbc4f6d0-5d7b-4d5e-e53e-218e2cfc225a"
y.head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 67797, "status": "ok", "timestamp": 1722680624762, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="wFhRSnj33SS8" outputId="6fb5020b-e382-45d2-b2d1-650211fd23df"
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn , direction='backward')
X_new=sfs.fit_transform(X, y)
print(X.columns[sfs.get_support()].to_list())
X=X_new

# %% colab={"base_uri": "https://localhost:8080/", "height": 700} executionInfo={"elapsed": 2216, "status": "ok", "timestamp": 1722680626975, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="FZtOiIRVJZcE" outputId="7263af6e-dab6-4109-c869-6d7c8d1da99a"
data=np.column_stack([X, y])
data = pd.DataFrame(data)
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

# %% [markdown] id="daed8339"
# ## Data scaling

# %% executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1722680626976, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="dc204ef3"
from sklearn import preprocessing
X= preprocessing.StandardScaler().fit_transform(X)

# %% [markdown] id="7451e7ec"
# ## Data splitting

# %% executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1722680626976, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="d2e48059"
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=10)

# %% [markdown] id="83ff4275"
# ## Build models

# %% [markdown] id="bf112f5f"
# ## Decision Tree

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10535, "status": "ok", "timestamp": 1722680637501, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="db51e325" outputId="e70c9d6b-6f18-43d9-9c58-39fdd5e6f80e"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 1016, "status": "ok", "timestamp": 1722680638505, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="3f3ea1b5" outputId="2a7527f5-0f3f-4d00-d7c5-6a02e8f7640a"
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

# %% [markdown] id="2751fc1b"
# ## Random Forest implementation

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 25769, "status": "ok", "timestamp": 1722680664270, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1eba2000" outputId="b0691c29-fad6-4fc7-e75a-77c0dc594e15"
model = RandomForestClassifier(random_state=42)
params = {'n_estimators': np.arange(100,500,100)}
cv_rf = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1, cv=5,verbose=3)
cv_rf.fit(X_train, y_train)
best_RF=cv_rf.best_estimator_
y_pred = best_RF.predict(X_test)
print(cv_rf.best_params_, cv_rf.best_score_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 462, "status": "ok", "timestamp": 1722680664727, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="be21b196" outputId="fe094ba9-3566-4885-af41-98ba69c1dbef"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7277, "status": "ok", "timestamp": 1722680672272, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="491bb18b" outputId="d511bdb2-be6e-41f1-a40a-fcef8520e6e1"
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


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 985, "status": "ok", "timestamp": 1722680673247, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="27d9a51d" outputId="df171559-343a-4701-b4e0-19e77097581f"
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

# %% colab={"background_save": true} id="fd470897"
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

# %% colab={"background_save": true} id="06b09c77"
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

# %% colab={"background_save": true} id="a60b691c"
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

# %% colab={"background_save": true} id="0331d0db"
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

# %% colab={"background_save": true} id="tSFmnxIwi5b3"
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

# %% colab={"background_save": true} id="1X8Igtd-n9mf"
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

# %% colab={"background_save": true} id="9MwKxAaGEl8i"
from sklearn.svm import SVC
model = SVC(probability=True)
params = {'kernel': ['linear', 'rbf'], 'C': np.arange(0.5,1,0.1), 'gamma': np.arange(0.01,0.05,0.01)}
cv_svm = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1,cv=5)
cv_svm.fit(X_train, y_train)
best_SVM=cv_svm.best_estimator_
y_pred = best_SVM.predict(X_test)
print(cv_svm.best_params_, cv_svm.best_score_)

# %% colab={"background_save": true} id="iW0JYKG6oB9d"
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
