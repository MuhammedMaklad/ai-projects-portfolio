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

# %% [markdown]
# ## CSU-Spring小队 Kernel introduction
# <br>
#
# ### Introduction
#
# 1. 🌱 team: Spring小队<br><br>
# 2. 😄 成员: Aczy156(陈冉飞) Chun yu(陳俊瑜) AFun(纪昱帆)<br><br>
# 3. ⚡ 选题: 中南大学(Central South University) 大三上半学年 工程研究与实习 软件缺陷测试<br><br>
# 4. 🔭 dataset source: <br>
#     * Tronclass 任老师的资料夹中的 NASA资料夹中文件<br>
#     * [NASA open dataset](https://nasa.github.io/data-nasa-gov-frontpage/)<br>
#     * [Kaggle open dataset](https://www.kaggle.com/semustafacevik/software-defect-prediction)<br>
#
# <br>
# ### Dataset and Kernel
#
# Dataset: 整合数据，并upload至Kaggle，数据集已开源。<br>
# URL: [https://www.kaggle.com/aczy156/software-defect-prediction-nasa](https://www.kaggle.com/aczy156/software-defect-prediction-nasa)
# <br><br>
#
# Kernel：(也就是当前此kernel)
# * URL: [https://www.kaggle.com/aczy156/software-defect-prediction-nasa-eda-naive-bayes](https://www.kaggle.com/aczy156/software-defect-prediction-nasa-eda-naive-bayes)
# * Version：
#     * Version1: "build: Baseline(based on machine leanring)"
#     * Version2: "feats: Introduction"
#     * Version3: "fix: EDA plot method"
#     * Version4: "feats: Machine Learning model - Decision Tree"
#     * Version5: "fix: evaluate function"
#     * Version6: "feats: Machine Learning model - boosting ['xgboost', 'lightgbm', 'catboost']"
#     * Version7: "feats: Model Compare by P-R curve and ROC curve"
#     * Version8: "feats: GridSearchCV for better parameters"
#     * Version9: "fix: model tuning."
#     * Versino10: "styles: format kernel."

# %% [markdown]
# ## prework
#
# * import basic dependencies
# * load data

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0"
import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

# %%
# load data
data = pd.read_csv('/kaggle/input/software-defect-prediction-nasa/jm1.csv')
data.shape

# %% [markdown]
# ## EDA
#
# * import some dependencies to plot
# * use plotly to visualization
#     * label classification
#         * count and plot(visualization)
#     * value visualization
#         * use historgram to visualization attribution
#         * relationship
#             * covariance
#             * heatmap
#     * scatter
#     

# %%
# import some dependencies to plot

from plotly.offline import iplot
# init_notebook_mode(connected=True)
import plotly.graph_objs as go


# %% _kg_hide-output=true
# check data
def show_info(data, is_matrix_transpose=False):
    # basic shape
    print('data shape is: {}   sample number {}   attribute number {}\n'.format(data.shape, data.shape[0], data.shape[1]))
    # attribute(key)
    print('data columns number {}  \nall columns: {}\n'.format(len(data.columns) ,data.columns))
    # value's null
    print('data all attribute count null:\n', data.isna().sum())
    # data value analysis and data demo
    if is_matrix_transpose:
        print('data value analysis: ', data.describe().T)
        print('data demo without matrix transpose: ', data.head().T)
    else:
        print('data value analysis: ', data.describe())
        print('data demo without matrix transpose: ', data.head())
        
show_info(data)

# %%
data.head()

# %%
# label classification
defects_true_false = data.groupby('defects')['b'].apply(lambda x: x.count())
print('True: ', defects_true_false[1], 'False: ', defects_true_false[0])
data.defects.value_counts().plot.bar()

# %% [markdown]
# **Then show the covariance.**
#  -- by coveriance matrix

# %% _kg_hide-input=true
# Attribute relationship -- covariance
data.corr()


# %%
# plot columns distribution
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    
plotPerColumnDistribution(data, 10, 5)


# %%
# plot corr
def plotCorrelationMatrix(df, graphWidth):
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.show()

plotCorrelationMatrix(data, 8)


# %%
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    
plotScatterMatrix(data, 20, 10)

# %% [markdown]
# ## Data cleaning
#
# * fillna
# * remove outliars (by use boxplot to visualization)
# * data type transform
#     * remove ~char

# %%
trace1 = go.Box(x=data['uniq_Op'])
box_data = [trace1]
iplot(box_data)

# %% [markdown]
# ### some special columns 
#
# * data cleaning
# * change data type

# %%
# some special columns [type is 'object']
object_type_cols = ['uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']
# data.head()
# data['uniq_Op'] = data['uniq_Op'].astype(np.float64)
# data['uniq_Opnd'] = data['uniq_Opnd'].astype(np.float64)
# data['total_Op'] = data['total_Op'].astype(np.float64)
# data['total_Opnd'] = data['total_Opnd'].astype(np.float64)
# data['branchCount'] = data['branchCount'].astype(np.float64)

# %% [markdown]
# ## Feature engineering
#
# * extract some useful attributions and create new attribution

# %% _kg_hide-output=true
# extract useful attributions and create new attribution
def extract_and_eval(data):
    '''
    input: data
    goal: make an evaluation to every sample and label
    '''
    eval = (data.n < 300) & (data.v < 1000) & (data.d < 50) & (data.e < 500000) & (data.t < 5000)
    data['eval'] = pd.DataFrame(eval)
    data['eval'] = [1 if e == True else 0 for e in data['eval']]

extract_and_eval(data)
show_info(data)

# %% [markdown]
# ## Data Normalization
#
# * load importance
# * use Min-Max Normalization

# %%
from sklearn.preprocessing import MinMaxScaler

# %% _kg_hide-output=true
scale_v = data[['v']]
scale_b = data[['b']]

minmax_scaler = MinMaxScaler()

v_scaled = minmax_scaler.fit_transform(scale_v)
b_scaled = minmax_scaler.fit_transform(scale_b)
data['scaled_v'] = pd.DataFrame(v_scaled)
data['scaled_b'] = pd.DataFrame(b_scaled)

# check data
show_info(data)

# %%
tem_data = data.copy()

# %% [markdown]
# ## Model
#
# * hyper-parameter
# * data prepare
#     * extract target(label)
#     * train train split
#     * cross-validation(because of data size only 1w) => use 10-cv
# * build model
#     * naive bayes
#     * LR
#     * Boosting
#         * xgboost
#         * lightgbm
# * fit, predict, evaluate(precision, recall, f1-score, acc, roc, auc)

# %%
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# machine learning model
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
# boosting
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# %%
# hyper-parameter
validation_size = 0.2
random_seed=7

# %%
# extract target
data['target'] = data['defects'].apply(lambda x: 1 if x == True else 0)
data = data.drop(['defects'], axis=1)

# %%
data.info()

# %%
# def is_number(s):
#     try:
#         float(s)
#         return True
#     except ValueError:
#         pass
 
#     try:
#         import unicodedata
#         unicodedata.numeric(s)
#         return True
#     except (TypeError, ValueError):
#         pass
 
#     return False

# data type change prework
origin_data_type_cols = ['uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']
data = data.drop(origin_data_type_cols, axis=1)
# for col in origin_data_type_cols:
#     data[col] = data[data[col].is_number()]
#     data[col] = data[col].astype(np.float64)
# data.info()

# %%
target = data['target']
data = data.drop(['target'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    data,
    target,
    test_size=validation_size,
    random_state=random_seed
)
X_train.shape, X_val.shape, y_train.shape, y_val.shape

# %%
# model evaluation calculate and score
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score,  mean_squared_error
# model evaluation 
from sklearn.metrics import classification_report, confusion_matrix

# metrics method
def metrics_calculate(model_name, y_val, y_pred):
    '''
    0. basic metrics values ['accuracy', 'precision', 'recall', 'fpr', 'fnr', 'auc']
    1. classification report
    2. confusion matrix
    '''
#     y_val = np.reshape(y_val, -1).astype(np.int32)
#     y_pred = np.where(np.reshape(y_pred, -1) > 0.5, 1, 0)
#     accuracy = accuracy_score(y_val, y_pred)
#     precision = precision_score(y_val, y_pred)
#     recall = recall_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
#     auc = roc_auc_score(y_val, y_pred)
#     print('Model:%s Acc:%.8f Prec:%.8f Recall:%.8f FNR:%.8f FPR:%.8f AUC:%.8f' % (model_name, accuracy, precision, recall, fnr, fpr, auc))
    print(model_name, 'classification report:\n', classification_report(y_val, y_pred))
    print(model_name, 'confusion_matrix:\n', confusion_matrix(y_val, y_pred))
    print('\n%s FNR:%.8f FPR:%.8f\n%s accuracy:%.8f' % (model_name, fnr, fpr, model_name, accuracy_score(y_pred,y_val)))


# %%
# plot metrics model answer(metrics)
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve

def metrics_plot(model_name, model, X_val, y_val):
    # plot P-R curve
    disp = plot_precision_recall_curve(model, X_val, y_val)
#     disp.ax_.set_title('2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))


# %% [markdown]
# ### Boosting
#
# * xgboost
# * lightgbm
# * catboost
#
# #### Choose a model that performs best => gridSearchCV
# * get best parameter

# %%
# %%time

# lightgbm
lgb = LGBMClassifier(
    max_depth=7,
    lambda_l1=0.1,
    lambda_l2=0.01,
    learning_rate=0.01,
    n_estimators=500,
    reg_aplha=1.1,
    colsample_bytree=0.9,
    subsample=0.9,
    n_jobs=5
)
# cv = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
# print('lightgbm cv score: ', cv)

# %%
# fit
lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='accuracy', verbose=False, early_stopping_rounds=50)
# predict
y_pred = lgb.predict(X_val)
# evaluate
metrics_calculate('Boosting lightgbm', y_val, y_pred)

# %%
# %%time

# catboost
cb = CatBoostClassifier(
    depth = 9, 
    reg_lambda=0.1,
    learning_rate = 0.09,
    iterations = 500
)
# cv = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

# %%
# fit
cb.fit(X_train, y_train, eval_set=[(X_val, y_val)],  verbose=False, early_stopping_rounds=50)
# predict
cb.predict(X_val)
# evaluate
metrics_calculate('Catboost', y_val, y_pred)

# %%
# %%time

# xgboost
xgb = XGBClassifier(
    max_depth=9,
    learning_rate=0.01,
    n_estimators=500,
    reg_alpha=1.1,
    colsample_bytree = 0.9, 
    subsample = 0.9,
    n_jobs = 5
)
# cv = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
# print('xgboost cv score: ', cv)

# %%
# fit
# %time xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=2)
# pred
y_pred = xgb.predict(X_val)
# evaluate
metrics_calculate('Boosting xgboost', y_val, y_pred)

# %%
# compare boosting [P-R curve]
PR_curve = plot_precision_recall_curve(xgb, X_val, y_val)
PR_curve = plot_precision_recall_curve(lgb, X_val, y_val, ax=PR_curve.ax_)
PR_curve = plot_precision_recall_curve(cb, X_val, y_val, ax=PR_curve.ax_)
# compare boosting [ROC curve]
ROC_curve = plot_roc_curve(xgb, X_val, y_val)
ROC_curve = plot_roc_curve(lgb, X_val, y_val, ax=ROC_curve.ax_)
ROC_curve = plot_roc_curve(cb, X_val, y_val, ax=ROC_curve.ax_)

# %% jupyter={"source_hidden": true}
# # %%time

# # xgboost -- gridsearchcv
# gs_xgb = XGBClassifier(
#     eta= 0.3, 
#     n_estimators= 500,
#     gamma= 0,
#     max_depth= 6, 
#     min_child_weight= 1,
#     colsample_bytree= 1, 
#     colsample_bylevel= 1, 
#     subsample= 1, 
#     reg_lambda= 1, 
#     reg_alpha= 0,
#     seed= 33
# )

# # scale of tree
# scale_tree_params = {
#     'max_depth':[3,5,7,9],
#     'min_child_weight':[1,3,5]
# }


# # control fit degree
# fit_degree_params = {
#     'subsample':[i/10.0 for i in range(6,10)],
#     'colsample_bytree':[i/10.0 for i in range(6,10)],
#     'min_child_weight':[6,8,10,12],
#     'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
# }

# print('Search for best tree scale parameters')
# grid_search_params(gs_xgb, scale_tree_params, X_train, y_train)

# print('Search for best fit degree parameters')
# grid_search_params(gs_xgb, fit_degree_params, X_train, y_train)

# %% [markdown]
# ### Other models
#
# * Naive bayes
#     * GaussionNB 高斯分布的朴素贝叶斯
#     * MultinomialNB 多项式分布的朴素贝叶斯
#     * BernoulliNB 伯努利分布的朴素贝叶斯

# %%
# get data
# tem_data.head()
X_train, X_val, y_train, y_val = train_test_split(
    tem_data.iloc[:, :-10].values, 
    tem_data['eval'].values, 
    test_size=validation_size,
    random_state=random_seed
)
X_train.shape, X_val.shape, y_train.shape, y_val.shape

# %%
kfold = KFold(n_splits=10, random_state=random_seed)
# get model -- GaussianNB
gaussion_nb = GaussianNB()
# %time cv = cross_val_score(gaussion_nb, X_train, y_train, cv=kfold, scoring='accuracy')
print('Naive Bayes GaussianNB cv score: ', cv)

# %%
# fit
# %time gaussion_nb.fit(X_train, y_train)
# predict
y_pred = gaussion_nb.predict(X_val)
# evaluate
metrics_calculate('Naive Bayes - GaussionNB', y_val, y_pred)

# %%
# get model -- MultinomialNB
multinomial_nb = MultinomialNB()
# %time cv = cross_val_score(multinomial_nb, X_train, y_train, cv=kfold, scoring='accuracy')
print('Naive Bayes MultinomialNB cv score: ', cv)

# %%
# fit
# %time multinomial_nb.fit(X_train, y_train)
# predict
y_pred = multinomial_nb.predict(X_val)
# evaluate
metrics_calculate('Naive Bayes - MultinomialNB', y_val, y_pred)

# %%
# get model -- BernoulliNB
bernoulli_nb = BernoulliNB()
# %time cv = cross_val_score(bernoulli_nb, X_train, y_train, cv=kfold, scoring='accuracy')
print('Naive Bayes BernoulliNB cv score: ', cv)

# %%
# fit
# %time bernoulli_nb.fit(X_train, y_train)
# predict
y_pred = bernoulli_nb.predict(X_val)
# evaluate
metrics_calculate('Naive Bayes - BernoulliNB', y_val, y_pred)

# %%
# compare boosting [P-R curve]
PR_curve_nb = plot_precision_recall_curve(gaussion_nb, X_val, y_val)
PR_curve_nb = plot_precision_recall_curve(multinomial_nb, X_val, y_val, ax=PR_curve_nb.ax_)
PR_curve_nb = plot_precision_recall_curve(bernoulli_nb, X_val, y_val, ax=PR_curve_nb.ax_)
# compare boosting [ROC curve]
ROC_curve_nb = plot_roc_curve(gaussion_nb, X_val, y_val)
ROC_curve_nb = plot_roc_curve(multinomial_nb, X_val, y_val, ax=ROC_curve_nb.ax_)
ROC_curve_nb = plot_roc_curve(bernoulli_nb, X_val, y_val, ax=ROC_curve_nb.ax_)

# %% [markdown]
# ### GridSearchCV
#
# * lightgbm
# * xgboost
# * catboost

# %%
from sklearn.model_selection import GridSearchCV

def grid_search_params(model, parameters, X_train, y_train):
    gsearch = GridSearchCV(model, param_grid=parameters, scoring='roc_auc', cv=3)
    gsearch.fit(X_train, y_train, verbose = False)
    print('Best param value is: {0}\n'.format(gsearch.best_params_))
    print('Best score is: {0}\n'.format(gsearch.best_score_))
    print(gsearch.cv_results_['mean_test_score'], '\n')
#     print(gsearch.cv_results_['params'], '\n')


# %% [markdown]
# #### Lightgbm -- GridSearchCV
#
# * tree scale [cart regression tree]
# * control fit degree
#

# %%
# %%time

# init parameters
gs_lgb = LGBMClassifier(
    objective = 'binary',
    is_unbalance = True,
    metric = 'binary_logloss,auc',
    max_depth = 6,
    num_leaves = 40,
    learning_rate = 0.1,
    feature_fraction = 0.7,
    min_child_samples=21,
    min_child_weight=0.001,
    bagging_fraction = 1,
    bagging_freq = 2,
    reg_alpha = 0.001,
    reg_lambda = 8,
    cat_smooth = 0,
    num_iterations = 200,
)

# scale of tree
scale_tree_params = {
    'max_depth': [4, 6, 8],
    'num_leaves': [20, 30, 40],
    'min_child_samples': [18, 19, 20, 21, 22],
    'min_child_weight': [0.001, 0.002],
    'feature_fraction': [0.6, 0.8, 1],
}


print('Search for best tree scale parameters')
grid_search_params(gs_lgb, scale_tree_params, X_train, y_train)

# %% [markdown]
# **get lightgbm best parameters**
#
# * 'feature_fraction': 0.8, 
# * 'min_child_samples': 19, 
# * 'min_child_weight': 0.001
# * 'max_depth': 6, 
# * 'num_leaves': 20
#

# %%
# init
gs_lgb = LGBMClassifier(
    objective = 'binary',
    is_unbalance = True,
    metric = 'binary_logloss,auc',
    max_depth = 6,
    num_leaves = 20,
    learning_rate = 0.1,
    feature_fraction = 1,
    min_child_samples=19,
    min_child_weight=0.001,
    bagging_fraction = 1,
    bagging_freq = 2,
    reg_alpha = 0.001, 
    reg_lambda = 8,
    cat_smooth = 0,
    num_iterations = 200,
)

# %% [markdown]
# #### Xgboost -- GridSearchCV

# %%
# %%time

# xgboost -- gridsearchcv
gs_xgb = XGBClassifier(
    eta= 0.3, 
    n_estimators= 500,
    gamma= 0,
    max_depth= 6, 
    min_child_weight= 1,
    colsample_bytree= 1, 
    colsample_bylevel= 1, 
    subsample= 1, 
    reg_lambda= 1, 
    reg_alpha= 0,
    seed= 33
)

# scale of tree
scale_tree_params = {
    'max_depth':[3,5,7,9],
    'min_child_weight':[1,3,5]
}

print('Search for best tree scale parameters')
grid_search_params(gs_xgb, scale_tree_params, X_train, y_train)

# %%
# control fit degree
fit_degree_params = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'min_child_weight':[6,8,10,12],
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

print('Search for best fit degree parameters')
grid_search_params(gs_xgb, fit_degree_params, X_train, y_train)

# %% [markdown]
# **get xgboost best parameters**
#
# * 'max_depth': 3
# * 'min_child_weight': 1
# * 'colsample_bytree': 0.9
# * 'min_child_weight': 6
# * 'reg_alpha': 100
# * 'subsample': 0.9

# %%
gs_xgb = XGBClassifier(
    eta= 0.3, 
    n_estimators= 500,
    gamma= 0,
    max_depth= 3, 
    min_child_weight= 6,
    colsample_bytree= 0.9, 
    colsample_bylevel= 1, 
    subsample= 0.9, 
    reg_lambda= 100, 
    reg_alpha= 0,
    seed= 33
)

# %%
# test best parameters and origin 
gs_xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],  verbose=True, early_stopping_rounds=5)
# predict
y_pred = gs_xgb.predict(X_val)
# evaluate
metrics_calculate('Boosting xgboost after grid-search-cv: ', y_val, y_pred)

# %% [markdown]
# #### catboost gridsearchcv

# %%
# catboost
gs_cb = CatBoostClassifier(
    depth = 9, 
    reg_lambda=0.1,
    learning_rate = 0.09,
    max_bin = 64,
    l2_leaf_reg = 2,
    iterations = 500,
#     random_strength = 1
)

# catboost params
params = {
    'iterations': [500, 1000, 2000],
    'depth': [i for i in range(6,10)],
    'max_bin': [None, 32, 46, 100, 254],
    'l2_leaf_reg': [None, 2,10,20,30]
}

print('Search for best catboost parameters')
grid_search_params(gs_cb, params, X_train, y_train)

# %% [markdown]
# **get catboost best parameters**
#
# * depth = 9, 
# * reg_lambda=0.1,
# * learning_rate = 0.09,
# * max_bin = 64,
# * l2_leaf_reg = 2,
# * iterations = 500,

# %%
gs_cb = CatBoostClassifier(
    depth = 9, 
    reg_lambda=0.1,
    learning_rate = 0.09,
    max_bin = 64,
    l2_leaf_reg = 2,
    iterations = 500,
#     random_strength = 1
)

# %%
# test best parameters and origin 
gs_cb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],  verbose=True, early_stopping_rounds=5)
# predict
y_pred = gs_cb.predict(X_val)
# evaluate
metrics_calculate('Boosting catboost after grid-search-cv: ', y_val, y_pred)

# %% [markdown]
# ### Soft Voting classifier
#
# * gs_lgb
# * gs_xgb 
# * gs_cb
# * 

# %%
from sklearn.ensemble import VotingClassifier

# %%
voting_clf = VotingClassifier(estimators=[
    ('gs_lgb', gs_lgb),
    ('gs_xgb', gs_xgb),
    ('gs_cb', gs_cb)
], voting='soft')

# fit
voting_clf.fit(X_train, y_train)

# predict
y_pred = gs_cb.predict(X_val)
# evaluate
metrics_calculate('Boosting catboost after grid-search-cv: ', y_val, y_pred)

# %% [markdown]
# ### Decision Tree

# %%
# # %%time

# tree_model = DecisionTreeClassifier()
# cv = cross_val_score(tree_model, X_train, y_train, cv=kfold, scoring='accuracy')
# print('Naive Bayes cv score: ', cv)

# %%
# # fit
# # %time tree_model.fit(X_train, y_train)
# # predict
# y_pred = tree_model.predict(X_val)
# # evaluate
# metrics_calculate('Decision Tree', y_val, y_pred)

# %% [markdown]
# ### LR (Linear Regression)

# %%
# # LR
# X = data['loc'].values.reshape(-1, 1)
# X_train, X_val, y_train, y_val = train_test_split(X, data['loc'].values)
# X_train.shape, X_val.shape, y_train.shape, y_val.shape

# %%
# X_train.head()

# %%
# # %%time

# model = LinearRegression()
# model.fit(X_train, y_train)

# %%
# # predict
# y_pred = model.predict(X_val)
# print('Mean Squared Error (MSE):', mean_squared_error(y_val, y_pred))  
# print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_val, y_pred)))

# %%
# X_train.head()
