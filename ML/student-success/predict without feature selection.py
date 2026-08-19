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

# %% papermill={"duration": 1.53618, "end_time": "2022-01-19T01:39:23.515347", "exception": false, "start_time": "2022-01-19T01:39:21.979167", "status": "completed"} id="72e78d72" executionInfo={"status": "ok", "timestamp": 1726821896475, "user_tz": -180, "elapsed": 2489, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% colab={"base_uri": "https://localhost:8080/"} id="Pa4In3dR2VMp" executionInfo={"status": "ok", "timestamp": 1726821929228, "user_tz": -180, "elapsed": 32757, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b73d4dc1-614f-44c9-9e64-4e65e86a63a9"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="yi2lgH0t2WdW" executionInfo={"status": "ok", "timestamp": 1726821929228, "user_tz": -180, "elapsed": 11, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="1dded671-1597-4c7d-c14f-9596651c4408"
# cd /content/drive/MyDrive/Student Success

# %% [markdown] id="5a1c257f"
# ### Read the dataset

# %% papermill={"duration": 0.070758, "end_time": "2022-01-19T01:39:23.736923", "exception": false, "start_time": "2022-01-19T01:39:23.666165", "status": "completed"} id="b25c241b" executionInfo={"status": "ok", "timestamp": 1726821930277, "user_tz": -180, "elapsed": 1056, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data = pd.read_csv("dataset.csv", sep=";")

# %% papermill={"duration": 0.074315, "end_time": "2022-01-19T01:39:23.861711", "exception": false, "start_time": "2022-01-19T01:39:23.787396", "status": "completed"} id="6df63fbd" outputId="a7591a45-b9db-44e7-a0a0-ee914214f369" colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"status": "ok", "timestamp": 1726821930277, "user_tz": -180, "elapsed": 27, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.head()

# %% papermill={"duration": 0.059785, "end_time": "2022-01-19T01:39:24.073298", "exception": false, "start_time": "2022-01-19T01:39:24.013513", "status": "completed"} id="83503634" outputId="7c185b1c-1628-4c7b-ce0e-5ae2d57a6688" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1726821930277, "user_tz": -180, "elapsed": 25, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.shape

# %% papermill={"duration": 0.078689, "end_time": "2022-01-19T01:39:24.203703", "exception": false, "start_time": "2022-01-19T01:39:24.125014", "status": "completed"} id="06963df7" outputId="98104f2d-da0e-4352-a0d5-5bd293091df5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1726821930277, "user_tz": -180, "elapsed": 24, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.info()

# %% papermill={"duration": 0.094948, "end_time": "2022-01-19T01:39:24.349908", "exception": false, "start_time": "2022-01-19T01:39:24.254960", "status": "completed"} id="100e5863" outputId="f04db8c7-7f83-4251-b01b-62cdc6b9c362" colab={"base_uri": "https://localhost:8080/", "height": 495} executionInfo={"status": "ok", "timestamp": 1726821930277, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.describe(include="all")

# %% [markdown] id="07d6aabf"
# ### check if the dataset has null values

# %% papermill={"duration": 0.06262, "end_time": "2022-01-19T01:39:24.464107", "exception": false, "start_time": "2022-01-19T01:39:24.401487", "status": "completed"} id="619b37db" outputId="da177d43-c7f7-47d4-e46d-c1bd41ce3311" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1726821930277, "user_tz": -180, "elapsed": 22, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data.isnull().sum()

# %% [markdown] id="196311a6"
# ### Count the number of every class in the output

# %% colab={"base_uri": "https://localhost:8080/", "height": 517} id="Wv22PqNynSjg" executionInfo={"status": "ok", "timestamp": 1726822057790, "user_tz": -180, "elapsed": 329, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="eb197efa-d953-4683-fc25-e76597117cd5"
import matplotlib.pyplot as plt

# Create the plot with specified colors
colors = ['#FF9999', '#99CCFF', '#9900FF']  # Define your custom colors
plt.bar(data.Target.value_counts().index, data.Target.value_counts(), color=colors)

# Display the plot
plt.show()

# Display value counts
print(data.Target.value_counts(normalize=True))

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} id="fzPQJTiooJUX" executionInfo={"status": "ok", "timestamp": 1726822282593, "user_tz": -180, "elapsed": 343, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="83a2adc3-794a-48b9-9eb6-3350cd0ece99"
# prompt: encoding the Target into numbers using labe encoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Target'] = le.fit_transform(data['Target'])
data.head()


# %% [markdown] id="dfd2ccb0"
# ### Draw the correlation between the attributes

# %% id="1d980cce" outputId="d265fb8a-34a2-489a-cba9-d95af8f18571" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1726822306292, "user_tz": -180, "elapsed": 9496, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
#correlation map
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# %% [markdown] id="1dec189e"
# ## Convert Dropout =0, Graduate=1, Enrolled=3

# %% id="0bebdbcc"
data = pd.read_csv("dataset.csv", sep=";")
data["Target"]=data.Target.map(dict( Dropout =0, Graduate=1, Enrolled=3))

# %% [markdown] id="683e404c"
# ### select only Dropout and Graduate because the number of Enrolled students is very low

# %% id="62cf8a44"
data = data[data['Target'] != 3]

# %% [markdown] id="1d82725c"
# ### Read the input and the output

# %% papermill={"duration": 0.073658, "end_time": "2022-01-19T01:39:34.690351", "exception": false, "start_time": "2022-01-19T01:39:34.616693", "status": "completed"} id="011f71a6"
X = data.drop("Target",axis=1)
y = data["Target"]

# %% papermill={"duration": 0.081611, "end_time": "2022-01-19T01:39:34.840205", "exception": false, "start_time": "2022-01-19T01:39:34.758594", "status": "completed"} id="2640555e" outputId="c81b321a-bdaa-4382-8d5b-3788714ab3f5"
X.head()

# %% id="855e83f2" outputId="fb0593dc-7478-45dd-96ba-a6cf88308a45"
y.head()

# %% [markdown] id="daed8339"
# ## Data scaling

# %% id="dc204ef3"
from sklearn import preprocessing
X= preprocessing.StandardScaler().fit_transform(X)

# %% [markdown] id="7451e7ec"
# ## Data splitting

# %% id="d2e48059"
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.15, random_state=10)

# %% [markdown] papermill={"duration": 0.068839, "end_time": "2022-01-19T01:39:35.132728", "exception": false, "start_time": "2022-01-19T01:39:35.063889", "status": "completed"} id="83ff4275"
# ## Build models

# %% [markdown] papermill={"duration": 0.065493, "end_time": "2022-01-19T01:39:35.678299", "exception": false, "start_time": "2022-01-19T01:39:35.612806", "status": "completed"} id="b6b3ab59"
# ## Logistic Regression Model

# %% papermill={"duration": 3.120202, "end_time": "2022-01-19T01:39:38.863804", "exception": false, "start_time": "2022-01-19T01:39:35.743602", "status": "completed"} id="0dcac604" outputId="c5173105-326a-476b-fd51-728a769f906e"
grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression(max_iter=1000)
logreg_cv=GridSearchCV(logreg,grid,cv=5)
logreg_cv.fit(X_train,y_train)

y_pred = logreg_cv.predict(X_test)
print(logreg_cv.best_params_, logreg_cv.best_score_)


# %% [markdown] id="5d182a34"
# ### Metrics calculation

# %% id="2eb737ae" outputId="98483a73-e395-4b91-8ef4-52189682677a"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% [markdown] papermill={"duration": 0.065624, "end_time": "2022-01-19T01:39:39.162419", "exception": false, "start_time": "2022-01-19T01:39:39.096795", "status": "completed"} id="031687d1"
# ## SVM

# %% papermill={"duration": 1.395977, "end_time": "2022-01-19T01:39:40.624683", "exception": false, "start_time": "2022-01-19T01:39:39.228706", "status": "completed"} id="aabcae4d" outputId="0d3ffe16-0175-46ae-9a82-b3173cf66dfd"
model = SVC()
params = {'kernel': ['linear', 'rbf'], 'C': np.arange(0.5,1,0.1), 'gamma': np.arange(0.01,0.05,0.01)}
cv_svm = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1,cv=5)
cv_svm.fit(X_train, y_train)
y_pred = cv_svm.predict(X_test)
print(cv_svm.best_params_, cv_svm.best_score_)

# %% papermill={"duration": 0.117567, "end_time": "2022-01-19T01:39:40.820605", "exception": false, "start_time": "2022-01-19T01:39:40.703038", "status": "completed"} id="42dac6ce" outputId="e1891365-1277-427a-900a-7585bfa820ed"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% [markdown] papermill={"duration": 0.072167, "end_time": "2022-01-19T01:39:40.964784", "exception": false, "start_time": "2022-01-19T01:39:40.892617", "status": "completed"} id="848c6075"
# ## K Nearest Neighbors

# %% papermill={"duration": 0.352778, "end_time": "2022-01-19T01:39:41.387125", "exception": false, "start_time": "2022-01-19T01:39:41.034347", "status": "completed"} id="4a926abe" outputId="1ab44fad-b166-4d38-e483-4fe8ebf4893f"
model = KNeighborsClassifier()
params = {'n_neighbors': np.arange(5,20,5), "weights": ['uniform', 'distance']}
cv_knn = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1,cv=5)
cv_knn.fit(X_train, y_train)
y_pred = cv_knn.predict(X_test)
print(cv_knn.best_params_, cv_knn.best_score_)

# %% papermill={"duration": 0.114034, "end_time": "2022-01-19T01:39:41.712415", "exception": false, "start_time": "2022-01-19T01:39:41.598381", "status": "completed"} id="ebc30959" outputId="750dac8f-1472-429d-8dc7-bf67a7881545"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% [markdown] papermill={"duration": 0.068374, "end_time": "2022-01-19T01:39:41.850613", "exception": false, "start_time": "2022-01-19T01:39:41.782239", "status": "completed"} id="bf112f5f"
# ## Decision Tree

# %% papermill={"duration": 5.668179, "end_time": "2022-01-19T01:39:47.587646", "exception": false, "start_time": "2022-01-19T01:39:41.919467", "status": "completed"} id="db51e325" outputId="26de5c7d-e13b-49ff-b3ac-beba4a8b4697"
model = DecisionTreeClassifier(random_state=42)
params = {'splitter': ["best", "random"],
          'max_depth': np.arange(1,10,1),
          'min_samples_leaf': np.arange(1,5,1),
          'min_samples_split': np.arange(2,5,1)
         }
cv_dt = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1,cv=5)
cv_dt.fit(X_train, y_train)
y_pred = cv_dt.predict(X_test)
print(cv_dt.best_params_, cv_dt.best_score_)

# %% papermill={"duration": 0.105385, "end_time": "2022-01-19T01:39:47.762516", "exception": false, "start_time": "2022-01-19T01:39:47.657131", "status": "completed"} id="3f3ea1b5" outputId="8c0e1d64-bda9-463b-f430-3b5074a11bcd"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% [markdown] papermill={"duration": 0.068996, "end_time": "2022-01-19T01:39:48.918816", "exception": false, "start_time": "2022-01-19T01:39:48.849820", "status": "completed"} id="2751fc1b"
# ## Random Forest implementation

# %% papermill={"duration": 4.941656, "end_time": "2022-01-19T01:39:53.930617", "exception": false, "start_time": "2022-01-19T01:39:48.988961", "status": "completed"} id="1eba2000" outputId="e88b05b6-4d33-48cb-8185-64e539ae527f"
model = RandomForestClassifier(random_state=42)
params = {'n_estimators': np.arange(100,500,100)}
cv_rf = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1, cv=5,verbose=3)
cv_rf.fit(X_train, y_train)
y_pred = cv_rf.predict(X_test)
print(cv_rf.best_params_, cv_rf.best_score_)

# %% papermill={"duration": 0.167409, "end_time": "2022-01-19T01:39:54.168655", "exception": false, "start_time": "2022-01-19T01:39:54.001246", "status": "completed"} id="be21b196" outputId="41e9079c-4322-444c-c091-8f5b43015b85"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% [markdown] papermill={"duration": 0.072111, "end_time": "2022-01-19T01:39:54.311507", "exception": false, "start_time": "2022-01-19T01:39:54.239396", "status": "completed"} id="747b897f"
# ### Adaboost Classifier

# %% papermill={"duration": 0.077878, "end_time": "2022-01-19T01:39:54.459436", "exception": false, "start_time": "2022-01-19T01:39:54.381558", "status": "completed"} id="05a942dd" outputId="30b5e8da-06b2-4721-b2f8-88f2c60f4555"
model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

parameters = {'base_estimator__max_depth':[i for i in range(2,11,2)],
              'base_estimator__min_samples_leaf':[5,10],
              'n_estimators':[10,50,250,1000],
              'learning_rate':[0.01,0.1]}

Adaboost = GridSearchCV(model, parameters,verbose=3,scoring='f1',n_jobs=-1,cv=5)
Adaboost.fit(X_train,y_train)
y_pred = Adaboost.predict(X_test)
print(Adaboost.best_params_, Adaboost.best_score_)

# %% papermill={"duration": 0.076359, "end_time": "2022-01-19T01:39:54.608030", "exception": false, "start_time": "2022-01-19T01:39:54.531671", "status": "completed"} id="04b410ee" outputId="c207aa48-20aa-43b5-f8c2-7fb71a6512c5"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% [markdown] id="43ca72d0"
# ## XGBoost Classifier

# %% id="491bb18b" outputId="441fb588-07fb-4dd7-cd1b-cf031502795a"
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
y_pred = random_search.predict(X_test)
print(random_search.best_params_, random_search.best_score_)


# %% id="27d9a51d" outputId="9432e143-2601-4aae-fe28-40b96b0d965c"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% [markdown] id="ccb106db"
# ## MLP implementation

# %% id="fd470897" outputId="d3d99226-6c37-43f9-96ec-8e6b921a92ca"
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
y_pred=clf.predict(X_test)

# %% id="06b09c77" outputId="3a00c0a3-a410-4a29-bc7f-5b2ea4e93556"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% id="a0441d03" outputId="ee55b263-3c20-4324-dbc3-d3996ebd3822"
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# A sample parameter

parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.075, 0.15],
    "min_samples_split": np.linspace(0.3, 0.5, 2),
    "min_samples_leaf": np.linspace(0.3, 0.5, 2),
    "max_depth":[8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[ 0.75, 0.95],
    "n_estimators":[10]
    }
#passing the scoring function in the GridSearchCV
clf = GridSearchCV(GradientBoostingClassifier(), parameters,cv=5, n_jobs=-1, verbose=3)

clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

# %% id="1cbdc1db" outputId="cabb114b-1453-47fb-d30c-5165133c05d4"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% id="a60b691c" outputId="55c37d73-e470-4659-d9c6-580a81705b58"
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

y_pred=grid.predict(X_test)




# %% id="0331d0db" outputId="14c58e2a-a708-4fb5-9cbc-0ae9fd2f0fe4"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% id="d22e18c6" outputId="2cce1367-be84-45cf-ec00-ff9696c737b5"
# Implementing Linear_SGD classifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(max_iter=1000)
Cs = [0.0001,0.001, 0.01, 0.1, 1, 10]
tuned_parameters = [{'alpha': Cs}]
model = GridSearchCV(clf, tuned_parameters, scoring = 'accuracy', cv=5,n_jobs=-1, verbose=3)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

# %% id="e1943129" outputId="9cd6a567-c785-4a52-bf1f-c2385c583af5"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% id="d1d3c7e3" outputId="6be55375-e875-4375-df82-e2a4d487a18b"
from sklearn.ensemble import BaggingClassifier

bc_params = {"base_estimator__max_depth": [3,5,10,20],
          "base_estimator__max_features": [None, "auto"],
          "base_estimator__min_samples_leaf": [1, 3, 5, 7, 10],
          "base_estimator__min_samples_split": [2, 5, 7],
          'bootstrap_features': [False, True],
          'max_features': [0.5, 0.7, 1.0],
          'max_samples': [0.5, 0.7, 1.0],
          'n_estimators': [2, 5, 10, 20],
}


bc_gs = GridSearchCV(BaggingClassifier(DecisionTreeClassifier()), bc_params, cv=5, verbose=3)
bc_gs.fit(X_train, y_train)
y_pred=bc_gs.predict(X_test)

# %% id="e70bf6c1"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% colab={"base_uri": "https://localhost:8080/"} id="695149d2" executionInfo={"status": "ok", "timestamp": 1653492859723, "user_tz": -120, "elapsed": 2280, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c3229bf0-19c6-4fd7-e811-05ad3c2fca5b"
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
precision_rate = tp / (tp + fp)
recall_rate = tp / (tp + fn)
fscore=2*((precision_rate*recall_rate)/(precision_rate+recall_rate))
print("The precision rate is: ", precision_rate)
print("The recall rate is: ", recall_rate)
print("The F1-score  is: ", fscore)

# %% [markdown] id="sDpkIr6D2B-m"
# Logistic Regression with 93.2 % without feature selection
