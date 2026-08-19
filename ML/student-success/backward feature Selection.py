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

# %% id="72e78d72" executionInfo={"status": "ok", "timestamp": 1722673707908, "user_tz": -180, "elapsed": 2679, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22960, "status": "ok", "timestamp": 1722673730866, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="jxl7GQ8XyD_f" outputId="bc60386c-5d74-45f4-d71b-0cd69e0fd6af"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3179, "status": "ok", "timestamp": 1722673734040, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="kPr8_CE3yIJa" outputId="ad7c3a5a-8751-4469-c187-857bfdee881c"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/Generic Projects/Student Success

# %% [markdown] id="5a1c257f"
# ### Read the dataset

# %% id="b25c241b" executionInfo={"status": "ok", "timestamp": 1722673734041, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data = pd.read_csv("dataset.csv", sep=";")

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1722673734041, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="6df63fbd" outputId="17b94a91-84a4-43cc-83db-1b1430f90f47"
data.head()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1722673734041, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="83503634" outputId="fad295cf-1a13-40ff-f4c3-5fec53e398d8"
data.shape

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1722673734041, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="06963df7" outputId="9e34d854-06d9-4eac-eaa0-bda26ea8454c"
data.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 495} executionInfo={"elapsed": 16, "status": "ok", "timestamp": 1722673734041, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="100e5863" outputId="1a575b80-5ed2-4f4f-9288-7e6922b8460b"
data.describe(include="all")

# %% [markdown] id="07d6aabf"
# ### check if the dataset has null values

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1722673734041, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="619b37db" outputId="011c5b18-5663-4115-aedb-509f4f31fa29"
data.isnull().sum()

# %% [markdown] id="196311a6"
# ### Count the number of every class in the output

# %% colab={"base_uri": "https://localhost:8080/", "height": 752} executionInfo={"elapsed": 408, "status": "ok", "timestamp": 1722673734434, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4fd223d5" outputId="ebdbe414-e647-4e1c-f34b-e4f923f24606"
sns.countplot(data.Target, palette="bwr")
plt.show()
data.Target.value_counts(normalize=True)

# %% [markdown] id="1dec189e"
# ## Convert Dropout =0, Graduate=1, Enrolled=3

# %% id="0bebdbcc" executionInfo={"status": "ok", "timestamp": 1722673734434, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data = pd.read_csv("dataset.csv", sep=";")
data["Target"]=data.Target.map(dict( Dropout =0, Graduate=1, Enrolled=2))

# %% [markdown] id="683e404c"
# ### select only Dropout and Graduate because the number of Enrolled students is very low

# %% id="62cf8a44" executionInfo={"status": "ok", "timestamp": 1722673734434, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
data = data[data['Target'] != 2]

# %% [markdown] id="1d82725c"
# ### Read the input and the output

# %% id="011f71a6" executionInfo={"status": "ok", "timestamp": 1722673734434, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X = data.drop("Target",axis=1)
y = data["Target"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1722673734434, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="2640555e" outputId="f9565059-6b10-4aae-8ae7-722a985f2ace"
X.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1722673734434, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="855e83f2" outputId="7e0529cc-ad4d-4048-86d8-91e8e6314f01"
y.head()

# %% id="d707f925" executionInfo={"status": "ok", "timestamp": 1722673734434, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X.iloc[:,-2:]=scaler.fit_transform(X.iloc[:,-2:])

# %% colab={"base_uri": "https://localhost:8080/", "height": 213} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1722673734434, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="2eeffa50" outputId="569fbce8-7e96-41a7-dee9-2b9ba4faf4f4"
X[:2]

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 40377, "status": "ok", "timestamp": 1722673774798, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1f65d265" outputId="65d1ec3e-243d-461f-e200-de7a99def21f"
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn , direction='backward')
X_new=sfs.fit_transform(X, y)
print(X.columns[sfs.get_support()].to_list())
X=X_new

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 2765, "status": "ok", "timestamp": 1722673777553, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="y0FqZqdI0Q_P" outputId="5cbfd49a-4bbd-48e0-a2b4-ed3c2822c7ff"
#correlation map
data=np.column_stack([X, y])
data = pd.DataFrame(data)
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

# %% [markdown] id="daed8339"
# ## Data scaling

# %% id="dc204ef3" executionInfo={"status": "ok", "timestamp": 1722673811534, "user_tz": -180, "elapsed": 521, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)

# %% [markdown] id="7451e7ec"
# ## Data splitting

# %% id="d2e48059" executionInfo={"status": "ok", "timestamp": 1722673811534, "user_tz": -180, "elapsed": 2, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=10)

# %% [markdown] id="83ff4275"
# ## Build models

# %% [markdown] id="b6b3ab59"
# ## Logistic Regression Model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 568, "status": "ok", "timestamp": 1669463040256, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="0dcac604" outputId="e5c3eb53-7724-4126-da57-ba50336e6533"
grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression(max_iter=1000)
logreg_cv=GridSearchCV(logreg,grid,cv=5)
logreg_cv.fit(X_train,y_train)

y_pred = logreg_cv.predict(X_test)
print(logreg_cv.best_params_, logreg_cv.best_score_)



# %% [markdown] id="5d182a34"
# ### Metrics calculation

# %% colab={"base_uri": "https://localhost:8080/", "height": 352} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1669463040257, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="2eb737ae" outputId="f1334f44-a13b-4f32-e5c3-02e6c5da5486"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
sensitifity = tp / (tp + fn)
specificity =tn/(fp+tn)
precision = tp / (tp + fp)
fscore=2*((precision*sensitifity)/(precision+sensitifity))
sensitifity=tp/(tp+fn)
print("The sensitifity = ", sensitifity)
print("The specificity = ", specificity)
print("The precision = ", precision)
print("The F1-score = ", fscore)

# %% [markdown] id="031687d1"
# ## SVM

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 47987, "status": "ok", "timestamp": 1669463098245, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="aabcae4d" outputId="abc35701-abc0-46d5-a6df-03fefd523677"
model = SVC()
params = {'kernel': ['linear', 'rbf'], 'C': np.arange(0.5,1,0.1), 'gamma': np.arange(0.01,0.05,0.01)}
cv_svm = GridSearchCV(model, param_grid=params, scoring="accuracy", n_jobs=-1,cv=5)
cv_svm.fit(X_train, y_train)
y_pred = cv_svm.predict(X_test)
print(cv_svm.best_params_, cv_svm.best_score_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 352} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1669463098246, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="42dac6ce" outputId="9368ea04-68d2-40c9-f083-059ef0fe76e6"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
sensitifity = tp / (tp + fn)
specificity =tn/(fp+tn)
precision = tp / (tp + fp)
fscore=2*((precision*sensitifity)/(precision+sensitifity))
sensitifity=tp/(tp+fn)
print("The sensitifity = ", sensitifity)
print("The specificity = ", specificity)
print("The precision = ", precision)
print("The F1-score = ", fscore)

# %% [markdown] id="2751fc1b"
# ## Random Forest implementation

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 29883, "status": "ok", "timestamp": 1669463128105, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="1eba2000" outputId="61fa4bc9-7d21-41d4-eb2f-e0a4de157a9f"
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
cv_rf = GridSearchCV(model, param_grid=param_grid, scoring="accuracy", n_jobs=-1, cv=5,verbose=3)
cv_rf.fit(X_train, y_train)
y_pred = cv_rf.predict(X_test)
print(cv_rf.best_params_, cv_rf.best_score_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 352} executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1669463128106, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="be21b196" outputId="b38d6b45-5cd4-4430-e4bf-0341ee1e8e7f"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
sensitifity = tp / (tp + fn)
specificity =tn/(fp+tn)
precision = tp / (tp + fp)
fscore=2*((precision*sensitifity)/(precision+sensitifity))
sensitifity=tp/(tp+fn)
print("The sensitifity = ", sensitifity)
print("The specificity = ", specificity)
print("The precision = ", precision)
print("The F1-score = ", fscore)

# %% [markdown] id="ccb106db"
# ## MLP implementation

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 684103, "status": "ok", "timestamp": 1669463812185, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="fd470897" outputId="e3405500-8663-47d9-f761-0201f22f3822"
from sklearn.neural_network import MLPClassifier
mlp_gs = MLPClassifier(max_iter=1000)
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
print(clf.best_params_, clf.best_score_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 352} executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1669463812186, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="06b09c77" outputId="5ff4f3df-a81e-4b36-be7e-b28af5bc104f"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
sensitifity = tp / (tp + fn)
specificity =tn/(fp+tn)
precision = tp / (tp + fp)
fscore=2*((precision*sensitifity)/(precision+sensitifity))
sensitifity=tp/(tp+fn)
print("The sensitifity = ", sensitifity)
print("The specificity = ", specificity)
print("The precision = ", precision)
print("The F1-score = ", fscore)

# %% [markdown] id="CTUbDWRux7mf"
# GradientBoostingClassifier

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 16911, "status": "ok", "timestamp": 1669463829089, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="a0441d03" outputId="3d63b284-dea9-44c1-9b62-e1aa1271e167"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 356} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1669463829091, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="1cbdc1db" outputId="8d6d9311-4389-40c4-ba62-934cdf952f0b"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
sensitifity = tp / (tp + fn)
specificity =tn/(fp+tn)
precision = tp / (tp + fp)
fscore=2*((precision*sensitifity)/(precision+sensitifity))
sensitifity=tp/(tp+fn)
print("The sensitifity = ", sensitifity)
print("The specificity = ", specificity)
print("The precision = ", precision)
print("The F1-score = ", fscore)

# %% [markdown] id="hhGSe_PLP-Ug"
# ## Voting Classifier

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22153, "status": "ok", "timestamp": 1669463851220, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="XYoRJT2bP93M" outputId="2d70be5b-90c7-4e30-ce13-4e8e77c138d0"
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


# %% colab={"base_uri": "https://localhost:8080/", "height": 352} executionInfo={"elapsed": 554, "status": "ok", "timestamp": 1669463851767, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="4WgA81SUP98u" outputId="af639ba8-d0d9-4c82-f93f-92357a9eed47"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
sensitifity = tp / (tp + fn)
specificity =tn/(fp+tn)
precision = tp / (tp + fp)
fscore=2*((precision*sensitifity)/(precision+sensitifity))
sensitifity=tp/(tp+fn)
print("The sensitifity = ", sensitifity)
print("The specificity = ", specificity)
print("The precision = ", precision)
print("The F1-score = ", fscore)

# %% [markdown] id="B_igOuBCNaL0"
# ## SGD classifier

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1669463851768, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="d22e18c6" outputId="b657fdd4-f896-42f6-a0e4-c7e0bee2b7b1"
# Implementing Linear_SGD classifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(max_iter=1000)
Cs = [0.0001,0.001, 0.01, 0.1, 1, 10]
tuned_parameters = [{'alpha': Cs}]
model = GridSearchCV(clf, tuned_parameters, scoring = 'accuracy', cv=5,n_jobs=-1, verbose=3)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 352} executionInfo={"elapsed": 452, "status": "ok", "timestamp": 1669463852212, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="e1943129" outputId="11c9c33e-d294-4b1f-ea56-ed4532f983ae"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
sensitifity = tp / (tp + fn)
specificity =tn/(fp+tn)
precision = tp / (tp + fp)
fscore=2*((precision*sensitifity)/(precision+sensitifity))
sensitifity=tp/(tp+fn)
print("The sensitifity = ", sensitifity)
print("The specificity = ", specificity)
print("The precision = ", precision)
print("The F1-score = ", fscore)

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9483, "status": "ok", "timestamp": 1669465683153, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="d1d3c7e3" outputId="2df2a479-c901-45bd-c1b2-4b393034832a"
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

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/", "height": 352} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1669465683154, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -120} id="e70bf6c1" outputId="1d236be7-f283-4faa-e30b-aa525d637037"
ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
sensitifity = tp / (tp + fn)
specificity =tn/(fp+tn)
precision = tp / (tp + fp)
fscore=2*((precision*sensitifity)/(precision+sensitifity))
sensitifity=tp/(tp+fn)
print("The sensitifity = ", sensitifity)
print("The specificity = ", specificity)
print("The precision = ", precision)
print("The F1-score = ", fscore)

# %% [markdown] id="rER3cIjypjp8"
# ## Stacking Classifeir

# %% colab={"base_uri": "https://localhost:8080/", "height": 788} id="ETSD95DGLQe7" outputId="e3cb81b0-be7e-49c0-e850-6dbf90d1156e" executionInfo={"status": "ok", "timestamp": 1675600972370, "user_tz": -120, "elapsed": 27319, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
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
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

ac = accuracy_score(y_test,y_pred)
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")
tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
sensitifity = tp / (tp + fn)
specificity =tn/(fp+tn)
precision = tp / (tp + fp)
fscore=2*((precision*sensitifity)/(precision+sensitifity))
sensitifity=tp/(tp+fn)
print("The sensitifity = ", sensitifity)
print("The specificity = ", specificity)
print("The precision = ", precision)
print("The F1-score = ", fscore)

# %% id="GwxtemqBmG8C" colab={"base_uri": "https://localhost:8080/", "height": 995} executionInfo={"status": "ok", "timestamp": 1675600979691, "user_tz": -120, "elapsed": 6470, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c885c5ba-ac05-4568-cdd4-ef59b1643ae9"
# !pip install scikit-plot
# !pip install plot-metric
from plot_metric.functions import BinaryClassification
# Visualisation with plot_metric
bc = BinaryClassification(y_test, y_pred, labels=["Class 1", "Class 2"])

# Figures
plt.figure(figsize=(5,5))
bc.plot_roc_curve()
plt.show()

# %% [markdown] id="PLJiqSJbxNCx"
# Logistic Regression , SVM , MLP are the largest accuracy with 93.8 %
