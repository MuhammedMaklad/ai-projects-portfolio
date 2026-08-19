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
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/", "height": 90} id="6j_vquuH-v32" outputId="6527286c-f7ea-4ceb-acbe-067ba13f159f"
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))


# %% id="MkQLsTcTAEgT"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')

# %% [markdown] id="QkDIXTy-AOmw"
# ## Read Dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="_JB9C9jUARy9" outputId="39057bc8-3517-41cb-aa0d-1e3df2407ac7"
df = pd.read_csv('kyphosis.csv')
df.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="ehx6IHx1AYgi" outputId="f0c09f9b-f24a-4531-87d8-39fb9e9c5f04"
df.info()

# %% [markdown] id="cDivzWXNAbAl"
# - there is no null values
# - all features are numerical

# %% [markdown] id="DIJ-KVe0A7Dk"
# ### Convert Target to numerical

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} id="4jGYw41xA9kC" outputId="05c58b21-1eb9-4df4-a70c-a5dd24583e18"
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Kyphosis'] = le.fit_transform(df['Kyphosis'])
df['Kyphosis'].head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 143} id="nFCdAoQvAiSI" outputId="caefd706-d2bf-4bca-edfd-97516ed012c6"
df.describe().T

# %% [markdown] id="yYrY-zqoAnuA"
# - detect outliers
# - scale featuers

# %% [markdown] id="VEiwYku_BJ_K"
# ### display corr between features and target

# %% colab={"base_uri": "https://localhost:8080/", "height": 435} id="o_RLL7lGAq5A" outputId="85fe8bad-e45e-4d97-b848-e0555da14d82"
sns.heatmap(df.corr(), annot=True)
plt.show()

# %% [markdown] id="Wb6YNgvRBWN_"
# - `Number` feature has influence to our target
# - `Age` feature may influnece to our target (very week influence)
# - `Start` feature influence our target

# %% [markdown] id="igwoWUZtC11X"
# ### Checking for outliers

# %% colab={"base_uri": "https://localhost:8080/", "height": 404} id="CAmiqqjACzO2" outputId="e317f10a-ede2-4292-f3e6-5c2446791bf4"
plt.figure(figsize=(18,16))
for idx, feature in enumerate(['Age', 'Number', 'Start']):
  plt.subplot(3,3,idx+1)
  sns.boxplot(df[feature])
plt.show()

# %% [markdown] id="0OedodNeDKoZ"
# - `Number` feature contain outliers

# %% [markdown] id="oA1BLFa3DN-1"
# ### Treat outliers by Imputing

# %% id="gIQLpdXMDRMK"
# Calculate Q1, Q3, and IQR for the 'Number' feature
Q1 = df['Number'].quantile(0.25)
Q3 = df['Number'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outlier detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Impute outliers with the median value
df['Number'] = np.where((df['Number'] < lower_bound) | (df['Number'] > upper_bound),
                         df['Number'].median(),
                         df['Number'])

# %% colab={"base_uri": "https://localhost:8080/", "height": 404} id="iw1Gcy1fDv6G" outputId="45fdf2b7-725f-421b-c3bd-1e06cf8b7725"
plt.figure(figsize=(18,16))
for idx, feature in enumerate(['Age', 'Number', 'Start']):
  plt.subplot(3,3,idx+1)
  sns.boxplot(df[feature])
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 442} id="9AEb25sND4p9" outputId="cc274c57-eb0a-419f-da2e-2b3c2696817b"
plt.figure(figsize=(18,16))
for idx, feature in enumerate(['Age', 'Number', 'Start']):
  plt.subplot(3,3,idx+1)
  sns.distplot(df[feature])
plt.show()

# %% [markdown] id="jJuYhR6QCWoH"
# ### feature Scalling

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="PZT1AAqBBPUN" outputId="c7084b71-3429-4ee9-9b0d-73ee2f785b07"
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Number', 'Start']] = scaler.fit_transform(df[['Age', 'Number', 'Start']])
df.head()

# %% [markdown] id="DYGfGSFYEcwk"
# ### Explore target

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} id="_0rMheCBEIqN" outputId="4acfec79-4596-47f4-d49d-0084ba9867f0"
df['Kyphosis'].value_counts().plot(kind='bar')
plt.show()

# %% [markdown] id="cMPHwhV4Eln_"
# - data is unbalanced

# %% [markdown] id="V66C6B_vEwix"
# ### balanced our data

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} id="0SsY7sPjEzGs" outputId="6c258730-1840-4465-f447-13a754dcd5b1"
from imblearn.over_sampling import SMOTE

# Separate features (X) and target (y)
X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a new balanced DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced['Kyphosis'] = y_resampled

# Display class counts after balancing
df_balanced['Kyphosis'].value_counts().plot(kind='bar')
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="vD2LtFrTEv6i" outputId="46b367a0-a564-4030-dbe4-7518cae26da5"
df.head()

# %% [markdown] id="8uwrgf57FU6a"
# ## Work on Balanced data

# %% [markdown] id="ckZCZjaJEEuI"
# ### Spliting dataset

# %% id="OzXwbIj8ED9C"
X  = df_balanced.drop('Kyphosis', axis=1)
y = df_balanced['Kyphosis']

# %% id="woVCRPh9Fc2e"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% colab={"base_uri": "https://localhost:8080/"} id="yUESv0YlFfxP" outputId="f247fdc5-c2d6-44ca-a838-2bd5b23b5612"
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(dtc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the best model on the test set
best_dtc = grid_search.best_estimator_
y_pred = best_dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))


# %% colab={"base_uri": "https://localhost:8080/"} id="KxtKmEmvFprB" outputId="a82f9d88-68e5-4a5c-e145-032c8ef07591"
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="ZmTwo7b5F0rG" outputId="79495b4d-7996-46ff-9efb-35ca12c0a2d6"
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()

# %% [markdown] id="VK7uVU4xGSS0"
# ### run --> Random Forest

# %% colab={"base_uri": "https://localhost:8080/", "height": 812} id="AQ2W1dGwGJNA" outputId="0893133c-b669-4bd9-ca58-23a349fa6e6e"
from sklearn.ensemble import RandomForestClassifier

# Define the parameter grid for hyperparameter tuning
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion':['gini', 'entropy']
}

# Create a RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)

# Perform GridSearchCV
grid_search_rf = GridSearchCV(rfc, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search_rf.best_params_)

# Evaluate the best model on the test set
best_rfc = grid_search_rf.best_estimator_
y_pred_rf = best_rfc.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf}")
print(classification_report(y_test, y_pred_rf))

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')
plt.show()


# %% [markdown] id="CQWU1P2xHU5J"
# ## Accuracy Ehanced by using Random forst to .96
