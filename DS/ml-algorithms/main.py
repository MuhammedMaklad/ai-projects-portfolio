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

# %% id="BCx7NhwUTuFj" colab={"base_uri": "https://localhost:8080/", "height": 73} outputId="58a4ec7c-1d4a-4bb0-8aa4-ada3455451e7"
# prompt: code to upload dataset from local machine
from google.colab import files
uploaded1 = files.upload()

# %% id="R6kTH-N9VS2t"
import io
import pandas as pd
df_train = pd.read_csv(io.BytesIO(uploaded1['titanic_train.csv']))

# %% colab={"base_uri": "https://localhost:8080/", "height": 73} id="fn1MgOzEO76S" outputId="aec09c35-1864-4a92-a3ea-ba47f5ad3b15"
uploaded2 = files.upload()

# %% id="TIdRuuV3gBJF"
df_test = pd.read_csv(io.BytesIO(uploaded2['titanic_test.csv']))

# %% id="Wh8YWjGXgC72"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# Suppress warnings temporarily
warnings.filterwarnings("ignore")
# %matplotlib inline

# %% colab={"base_uri": "https://localhost:8080/", "height": 258} id="-xAFikMBgP0O" outputId="287fc36d-7d2a-41cf-b0ec-c405d1bf1f6f"
df_train.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="wbCIBnR-gZ5j" outputId="4f65c11b-7a75-4702-edb9-90744d491353"
df_test.head()

# %% id="R-6rKcOigml0"
# combine df_train and df_test to df
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# %% [markdown] id="2J-0JJG6g9-L"
# ## Explore dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 258} id="cwuESC_6gwOw" outputId="470fa747-0b49-4cc3-e404-deff18ed5d67"
df.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="SECF7CjlgwM-" outputId="8d4ba522-734b-413a-eafc-cdc9826a6d1d"
df.columns.tolist()

# %% colab={"base_uri": "https://localhost:8080/"} id="q614V4gRg1Ge" outputId="3151cd81-3307-4905-db57-6e6bdd1479c7"
df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="iblQfzychBYj" outputId="8c2b0b75-34cc-4947-a175-e97d759413d0"
df.describe()

# %% id="XlqxxijehG89"
# drop PassengerId column
df.drop('PassengerId', axis=1, inplace=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="jb9GGdi6hRGK" outputId="bf42c526-91b0-498f-fa3a-67086b1c602f"
df.columns.tolist()

# %% id="-ms308tSkfBO"
Y = df['Survived']

# %% id="dhfqBx6MkuRX"
df.drop('Survived', axis=1, inplace=True)

# %% id="6IoPDTxjkjSb"
# split our dataset to numerical and categorical feature
numerical_features = df.select_dtypes(include=np.number).columns.tolist()
categorical_features = df.select_dtypes(exclude=np.number).columns.tolist()

# %% colab={"base_uri": "https://localhost:8080/"} id="sf_YPAiok76_" outputId="6a753f2f-19dd-4fc6-a620-ee03adb19426"
numerical_features

# %% colab={"base_uri": "https://localhost:8080/"} id="OAxpNvFDk9iV" outputId="f831fb5c-16f0-4890-cccf-2b70adb45832"
categorical_features

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="XPty66NBlA0X" outputId="c6ae7e3c-de33-45c4-ac32-f07ebf9d7c9c"
df[numerical_features].head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="E8RdktmflFFo" outputId="7f574de1-5af9-4cd8-f80d-7dd8d8565ffc"
df[categorical_features].head()

# %% [markdown] id="NbD1r3QulT0p"
# # Pre Processing

# %% [markdown] id="SjpJovc1lXg4"
# ## pre processing for categorical features

# %% colab={"base_uri": "https://localhost:8080/", "height": 452} id="D51s61zrlWR3" outputId="9fb4e1e2-20ce-4fda-b3dc-3b7307650b32"
# checking for missing values
sns.heatmap(df[categorical_features].isnull(), cbar=True,annot=False, cmap='viridis')
plt.title('Missing Values in Categorical Features')
plt.show()

# %% [markdown] id="WbsygMa7lpN0"
# - `Cabin` feature contain missing value

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="fq4B4NtclunK" outputId="443282e8-1ebf-4b9e-ec62-15ed025b440b"
df['Cabin'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="nxO3_zppmbcp" outputId="f7e7ec9a-c59b-4e88-946c-30536517aef6"
# unique values in Cabin feature
print(f"number of unique values in 'Cabin' feature = {df['Cabin'].isnull().sum()}")

# %% colab={"base_uri": "https://localhost:8080/"} id="4MqphZgnmhla" outputId="3d7ed798-4cdd-4b7f-bb21-2e69aca087d3"
print(f"number of missing value in 'Cabin' feature = {df['Cabin'].isnull().sum()}")

# %% [markdown] id="5oFc8CRYnAdM"
# #### will drop `Cabin` feature
# - contain more than 1014 unique values
# - contain 1014 missing values

# %% id="Gn4ouQFJm_1p"
df.drop("Cabin",axis=1,inplace=True)

# %% id="FjWZn8VcnY4i"
categorical_features.remove("Cabin")

# %% [markdown] id="dphluHlIoB4O"
# #### remove `Name` feature
# - its unique identifier column
# - no need it

# %% id="jt_AjASbnbPe"
df.drop("Name",axis=1,inplace=True)


# %% id="tg-065sLoOU1"
categorical_features.remove("Name")

# %% colab={"base_uri": "https://localhost:8080/"} id="nlKLpmEgoPu0" outputId="f4817e63-7af3-4cb6-955d-4f0f0a197413"
print(f"number unique values in 'Embarked' feature = {df['Embarked'].nunique()}")

# %% [markdown] id="gOkNCLFspXYZ"
# #### Will convert `Embarked` feature into numerical by applying hot one encoding

# %% id="f4tehB6yoagb"
embarked = pd.get_dummies(df['Embarked'],drop_first=False)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="52nj0ch4p46t" outputId="65921b20-0d6c-42fa-997f-34561c02a53a"
embarked.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="kUupE2DuqPL_" outputId="e5eff192-0caf-4afa-a34d-3e19788d7d5b"
# Concatenate the embarked features with the main DataFrame
df = pd.concat([df, embarked], axis=1)
df.head()


# %% id="7NjA0l-TqfY5"
df.drop("Embarked",axis=1,inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="ImGFbMqhqiwi" outputId="f1eac0db-9d6d-4f15-b34e-15e703ad83c5"
df.head()


# %% colab={"base_uri": "https://localhost:8080/"} id="ss4oSs9ZqksO" outputId="d08a8c9c-87fe-448c-d5c9-602ead7eab33"
print(f"number unique values in 'Ticket' feature = {df['Ticket'].nunique()}")

# %% [markdown] id="zDrVLpplqoDj"
# #### Will remove `Ticket` feature
# - contian 929 unique values

# %% id="pNRvFP_eqzAo"
df.drop("Ticket",axis=1,inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="SHXGs7UOq2Ir" outputId="1ba47160-4e72-4ede-d34c-87d705c10140"
df.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="WE0d6znqq8mU" outputId="c96a945b-c3ba-436f-8b65-94239132422b"
print(f'number unique values in "Sex" feature = {df["Sex"].nunique()}')

# %% id="-LY2cxnSrBMR"
sex = pd.get_dummies(df['Sex'],drop_first=False)

# %% id="JLKWs__nrEM2"
df = pd.concat([df, sex], axis=1)
df.drop("Sex",axis=1,inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="SgM5nvxyrJr0" outputId="e32a6301-78e8-40b2-d5d1-4bf8ab3cfc33"
df.head()

# %% [markdown] id="tI7eKt05rNIx"
# ## pre processing value numerical features

# %% colab={"base_uri": "https://localhost:8080/", "height": 452} id="DxvyNjCKrSEm" outputId="34f69a3f-ceb3-4937-b27e-721f990b87bc"
# Checking for missing values
sns.heatmap(df[numerical_features].isnull(), cbar=True,annot=False, cmap='viridis')
plt.title('Missing Values in Numerical Features')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="fFCrK0RPrdpF" outputId="4db35897-cbb4-4bc0-b620-61602777b937"
print(f'number of missing value in "Age" feature = {df["Age"].isnull().sum()}')

# %% colab={"base_uri": "https://localhost:8080/"} id="jUnr2Z7_riFh" outputId="45ec07fe-44fd-4dd7-fee4-3ccd0c6ed621"
print(f'precentage of missing value in "Age" feature = {df["Age"].isnull().sum()/len(df)*100}')

# %% [markdown] id="XScndilxr_zF"
# - will impute missing value in `Age` feature

# %% colab={"base_uri": "https://localhost:8080/"} id="1FRnMNabr-tu" outputId="465c3f95-d529-4db7-b449-6ca8145d6bf1"
# Calculate the correlation between 'Age' and 'Survived'
correlation = df['Age'].corr(df_train['Survived'])
print(f"The correlation between Age and Survived is: {correlation}")


# %% colab={"base_uri": "https://localhost:8080/", "height": 17} id="MYKjnznXOucN" outputId="5f593460-dd61-4084-a9fb-99a0e5513d90"
import cufflinks as cf
cf.go_offline()

# %% colab={"base_uri": "https://localhost:8080/", "height": 636} id="w0cZEQ95PVsH" outputId="7b4b6188-1f7f-425b-95a1-5a0ecbd1e0c7"
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')


# %% id="Tza7DsIqP6cP"
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# %% id="3De2j6nWP8Lv"
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 452} id="4URa3d5xQIbr" outputId="35e2a700-1d04-4317-94c1-96faaae5f246"
sns.heatmap(df[numerical_features].isnull(), cbar=True,annot=False, cmap='viridis')
plt.title('Missing Values in Numerical Features')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="ouEyDTx-QUSf" outputId="8ef8018f-6ae0-4dfc-ee6d-ba80aeaf972e"
df[numerical_features].describe()

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="UGy_T41EQyez" outputId="a4d90388-7366-4e5d-aef4-c7aca231b760"
df.head(10)

# %% [markdown] id="0sSm0ft-SIW-"
# #### checking our target

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} id="zYuaBPYYSHnm" outputId="c65589c4-ae34-430d-f88c-fc6a132299ab"
Y.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="Ol79ARGOSZbj" outputId="4786b194-991c-4987-cedd-1ef694794ca0"
print(f'number of unique values in "Survived" feature = {Y.nunique()}')

# %% colab={"base_uri": "https://localhost:8080/"} id="FHJT2bAdSujQ" outputId="a41691b4-690b-488d-c781-6c45c33faf64"
print(f'number of missing value in "Survived" = {Y.isnull().sum()}')

# %% colab={"base_uri": "https://localhost:8080/", "height": 480} id="_bBRsDtOS8_e" outputId="190a976b-5a6b-4b84-b45a-40b0f26aba99"
Y.value_counts().plot(kind='bar')
plt.title('Survived Distribution')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# %% [markdown] id="4FApCFvVTFD-"
# - datase is unbalanced

# %% id="99tBqWAXTJLx"
Y.fillna(0.0,inplace=True)

# %% [markdown] id="1vJ5aRZyTi-W"
# - we consider that people with missing target is `unsurvived`

# %% colab={"base_uri": "https://localhost:8080/", "height": 480} id="kRZEZZNeTTkC" outputId="6fba7c0d-d439-4f7f-96f1-71cdb55a142e"
Y.value_counts().plot(kind='bar')
plt.title('Survived Distribution')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# %% [markdown] id="ZPgZOObbTuL0"
# #### balance dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 694} id="6lGMjFclUa9B" outputId="498addf4-74a0-4455-9087-92c1cc43d4c4"
# prompt: code for check NaN value in dataset

# Check for NaN values in the entire DataFrame
nan_counts = df.isna().sum()
print(nan_counts)

# Check for NaN values in specific columns
print(df['Age'].isna().sum())  # Example: Check NaN values in the 'Age' column

# Visualize missing values using a heatmap (as you did before)
sns.heatmap(df.isnull(), cbar=True, annot=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# %% id="z9_imlujU-sE"
# Fill NaN values in 'Fare' column with the mean of the column
df['Fare'].fillna(df['Fare'].mean(), inplace=True)


# %% colab={"base_uri": "https://localhost:8080/", "height": 467} id="TA3RD66GUL-v" outputId="78679813-202c-4311-cc11-674af51f8087"
# prompt: code to balance our dataset

from imblearn.over_sampling import SMOTE

# Separate features (X) and target variable (y)
X = df  # Assuming 'df' contains your features after preprocessing
y = Y.astype(int) # Ensure y is integer type

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Now X_resampled and y_resampled contain the balanced dataset
y_resampled.value_counts().plot(kind='bar')
plt.title('Survived Distribution after balancing')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()


# %% id="oHjSqre2RX3b"
# prompt: split dataset to train and test

from sklearn.model_selection import train_test_split

# Assuming 'Y' is your target variable and 'df' your features
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=42)

# %% [markdown] id="Se-bdDNMRn7E"
# ## Building model `Logistic Regression`

# %% colab={"base_uri": "https://localhost:8080/", "height": 80} id="LaFPZC23RiCd" outputId="aedab34c-c6ef-44b0-a724-6bffe00bd304"
# build logistic regression model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize and train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# %% id="mD4W43P0VRGC"
predictions = logreg.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="tN6vEOaKVVGE" outputId="0fa61549-07be-441b-85b2-39cf654a9020"
# code to evaluate model

from sklearn.metrics import classification_report, confusion_matrix

# Evaluate the model
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")


# %% colab={"base_uri": "https://localhost:8080/"} id="q_a9Kq6FVl5v" outputId="60eb4e6b-f5f4-4dea-99fc-b9c6d78c930f"
# build logistic regression model with best params and cross validation

from sklearn.model_selection import GridSearchCV

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga'] # Solvers that support both L1 and L2 penalties
}

# Initialize the logistic regression model
logreg = LogisticRegression()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_logreg = grid_search.best_estimator_

print(f"Best parameters: {best_params}")

# Make predictions using the best model
best_predictions = best_logreg.predict(X_test)

# Evaluate the best model
print(classification_report(y_test, best_predictions))
print(confusion_matrix(y_test, best_predictions))
accuracy = accuracy_score(y_test, best_predictions)
print(f"Accuracy: {accuracy}")


# %% colab={"base_uri": "https://localhost:8080/", "height": 564} id="2ct8LOs7WETR" outputId="fc33aab2-a03b-439f-cfc9-2edb26aadfb3"
# visualize confusion matrix

# Compute confusion matrix
cm = confusion_matrix(y_test, best_predictions)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

