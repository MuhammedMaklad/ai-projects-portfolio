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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# <a id="1"></a>
# # <div style="text-align:center; border-radius:15px 50px; padding:7px; color:white; margin:0; font-size:110%; font-family:Pacifico; background-color:#0f7a7a; overflow:hidden"><b>Preprocessing of NLP 
#
# </b></div>
#
# # Dataset link imdb 
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# %% [markdown]
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Read data</div>

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# Initialize tqdm with pandas
tqdm.pandas()

# %%
df=pd.read_csv('IMDB Dataset.csv')
df.head()

# %%
df.shape

# %%
df['review'][100].lower()

# %%
df['review']=df['review'].str.lower()
df.head()

# %%
sns.countplot(data=df,x='sentiment')

# %% [markdown]
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Remove html tags using Regular expressions</div>

# %%
import re
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub('', text)


# %%
df['review']=df['review'].progress_apply(lambda x : remove_html_tags(x))
df.head()


# %% [markdown]
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Removing URLs</div>

# %%
def remove_url(text):
    pattern=re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)

df['review']=df['review'].progress_apply(lambda x : remove_url(x))
df.head()

# %% [markdown]
#
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Handling emojis</div>

# %%
# pip install emoj
import emoji

df['review'] = df['review'].progress_apply(emoji.demojize)
df.head()


# %% [markdown]
#
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Remove digits</div>

# %%
# Define a function to remove digits
def remove_digits(text):
    # Remove digits using regex
    return re.sub(r'\d+', '', text)

df['review'] = df['review'].progress_apply(lambda x: remove_digits(x))
df.head()

# %% [markdown]
#
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Removing punchuation</div>

# %%
import string

# %%
exclude=string.punctuation
exclude


# %% [markdown]
# ##### These are the punchuations in python

# %%
def remove_punc(text):
    for char in exclude:
        text=text.replace(char,'')
    return text

df['review']=df['review'].progress_apply(lambda x : remove_punc(x))
df.head()

# %% [markdown]
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Spelling correction</div>

# %%
# #!pip install textblob
from textblob import TextBlob
def check_spelling(text):
    textblb=TextBlob(text)
    return textblb.correct().string

df['review']=df['review'].progress_apply(lambda x : check_spelling(x))
df.head()

# %% [markdown]
#
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Removing StopWords</div>
#

# %%
from nltk.corpus import stopwords

# %%
StopWords = stopwords.words("english")
StopWords


# %%
def remove_stopwords(text):
    filtered_text = ' '.join(word for word in text.split() if word.lower() not in StopWords)
    return filtered_text

df['review']=df['review'].progress_apply(lambda x : remove_stopwords(x))
df.head()

# %% [markdown]
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Stemming</div>

# %%
from nltk.stem import PorterStemmer

ps=PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])


# %%
sample='running ran runner easily fairly'
stem_words(sample)

# %%
#df['review'] = df['review'].progress_apply(stem_words)
#df.head()

# %% [markdown]
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Lemmatization</div>

# %%
from nltk.stem import WordNetLemmatizer

le=WordNetLemmatizer()
def lemm_words(text):
    return " ".join([le.lemmatize(word) for word in text.split()])


# %%
sample='running ran runner easily fairly'
lemm_words(sample)

# %%
#df['review'] = df['review'].progress_apply(lemm_words)
#df.head()

# %% [markdown]
#
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Split data</div>
#

# %%
X = df.drop('sentiment',axis =1)
y = df['sentiment']

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)


# %%
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# %% [markdown]
#
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Tokenization</div>
#

# %%
#NLTK
from nltk.tokenize import word_tokenize

# %%
X_train['Tokenization'] = X_train['review'].progress_apply(word_tokenize)
X_train.head()

# %% [markdown]
#
# <a id="1"></a>
# # <div style="text-align:center; border-radius:15px 50px; padding:7px; color:white; margin:0; font-size:110%; font-family:Pacifico; background-color:#0f7a7a; overflow:hidden"><b>Bag of Words
# </b></div>

# %% [markdown]
#
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Count vectorization and N-Gram</div>
#

# %%
from sklearn.feature_extraction.text import CountVectorizer

# %%
vectorizer = CountVectorizer(
    lowercase=True,                  # Converts all characters to lowercase before tokenization.
    tokenizer=None,                  # the default tokenizer will be used.
    stop_words=None,                 # You can provide a list of stop words or use 'english' to remove common English stop words.
    max_features=10000,              # all tokens will be included.
    ngram_range=(1, 1),              # Use unigrams (1 word)
    vocabulary=None,                 # a vocabulary will be created from the input documents.
    binary=False                     # If False, the count of occurrences of each token is recorded; if True, only presence/absence is recorded.
)

# Transform the text data into a document-term matrix
X_tr = vectorizer.fit_transform(X_train['review'])
X_te = vectorizer.transform(X_test['review'])

# %%
print(vectorizer.vocabulary_)

# %%
vocab = vectorizer.get_feature_names_out()
vocab

# %%
vocab.shape

# %%
X_train1 = pd.DataFrame(X_tr.toarray(), columns=vocab)

X_train1.head()

# %%
X_test1 = pd.DataFrame(X_te.toarray(), columns=vocab)

X_test1.head()

# %% [markdown]
#
#
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">TF-IDF (Term Frequency-Inverse Document Frequency)</div>
#
#

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
tfidf_vectorizer = TfidfVectorizer(lowercase=True,
                                    tokenizer=None,
                                    stop_words=None,
                                    ngram_range=(1, 1),
                                    max_features=None,
                                    vocabulary=None,
                                    binary=False)

# Transform the text data into a document-term matrix
X_tr = tfidf_vectorizer.fit_transform(X_train['review'])
X_te = tfidf_vectorizer.transform(X_test['review'])

# %%
vocab = tfidf_vectorizer.get_feature_names_out()
vocab

# %%
X_train2 = pd.DataFrame(X_tr.toarray(), columns=vocab)

X_train2.head()

# %%
X_test2 = pd.DataFrame(X_te.toarray(), columns=vocab)

X_test2.head()

# %% [markdown]
# <a id="1"></a>
# # <div style="text-align:center; border-radius:15px 50px; padding:7px; color:white; margin:0; font-size:110%; font-family:Pacifico; background-color:#0f7a7a; overflow:hidden"><b> Training and prediction using ML models
# </b></div>

# %% [markdown]
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Training with GaussianNB</div>

# %%
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# %%
gnb.fit(X_train1,y_train)
y_pred = gnb.predict(X_test1)
accuracy_score(y_test,y_pred)

# %%
gnb.fit(X_train2,y_train)
y_pred = gnb.predict(X_test2)
accuracy_score(y_test,y_pred)

# %% [markdown]
# <div style="padding:10px;color:white;margin:0;font-size:200%;text-align:center;display:fill;border-radius:10px;background-color:#215f95;;overflow:hidden;font-weight:501;font-family:magra">Training with Random Forest Classifier</div>

# %%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X_train1,y_train)
y_pred = rf.predict(X_test1)
accuracy_score(y_test,y_pred)

# %%
rf.fit(X_train2,y_train)
y_pred = rf.predict(X_test2)
accuracy_score(y_test,y_pred)

# %%
