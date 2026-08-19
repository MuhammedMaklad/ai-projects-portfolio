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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8687, "status": "ok", "timestamp": 1636890175823, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="j2tLXI3TzqJl" outputId="1846e756-7951-40d6-848a-edc7bee4b0c9"
pip install transformers

# %% id="LUBSfZV5P-sr"
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


# %% id="M7iJK3b6P-pp"
# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# %% colab={"base_uri": "https://localhost:8080/", "height": 145, "referenced_widgets": ["c86125fd7e9848328ceab88b5e82f6d2", "30948adc88e0428e97f78ca322dfbf55", "07fc78a7a573443396d6ce9c81f50d98", "72af2c9542eb4e6192c22264552e9a0e", "38b294c8e7cc49f5a00c40e133e38f1a", "dcab9d00150b43f7be3cb7c7d870ae79", "6f3e1a31c40d41898309b77afd073ce9", "06469f9db8b4497bac62084d1992a5e4", "913e5f45ebf348e9aafde7127c7a8671", "c09c24f8e1bb45abb88b956617016694", "29dfca96cce44fbab0539d20e957dc9c", "7102177c08bf435c933dbbc55c528b91", "08f0b7e342a3485aa5f9993c154cdc43", "641cb67ec4af4d908cd182909ddeef5d", "8394eb9d013b47d68b7a16ad30fd571a", "2ea027cfa6d040de815b331b392aab45", "43a42433ee4b4b7f97c900a57dc3a5f5", "c09fa9fe5c8445978ad86877d620049b", "ea8b702ce40a40cbb0d8553e54d84be9", "f114ccbddc494a63b2be5db9bfdc67b1", "56ef4224bc0b45cfb03bace9aaebfe68", "6478b5670ea54afcbbc17493f2104df0", "88c73d4c76fb46b8bd19a5684a55114b", "052906ed2bc8486d90d45699207acca5", "d1f25c71ba6b4206881edcfb6a056ee0", "a0da10fe81f0431a864c65087eadee09", "8968162db9534fd895dc5b62024b77ac", "b8ed7c01e0f54736a07ff9e3a1ddf65a", "3e902138693649cfb3e7643f3a65f5cd", "bf90fa65cefa48d2b415d572e5154a53", "75193444b95f4568903d02ae91de5d4e", "bdf67595e4e7468d89b83839d00e364d", "b346247cd094482e8b691bbbe2fda6a2", "81e4f7852fb54f4eba8574ec348253c0", "57dcf80ddc7841c79abd3a268e649e4d", "4b9b570ded1e4bd4b39b5b77fc51db23", "89e96fc456764c7e92ddba68076e88ac", "7bcd07150d06469a9588b61f630ac8df", "82071432ed264c0fa0e3d4c0420fc256", "9fe72bac3e48437da51173bd4eb16ef0", "3162d0b1d04c49fc90f3ec8663fd296e", "c47d625d1cbf4006b06c29f5314a631d", "86774bfd32404b7eb5f44f16cb65a47c", "6ce29f7da3c74856bdc6fcc65422fd54"]} executionInfo={"elapsed": 2216, "status": "ok", "timestamp": 1636890208042, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="DK79KuM0Qec0" outputId="96c42dad-e6f6-4326-c77b-127540088900"
# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary
task='emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)



# %% id="iz-BaW4WQg2O"
# download label mapping
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 282, "status": "ok", "timestamp": 1636890222773, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="cYgzhGeZQkD-" outputId="9cb2b9cc-5237-461b-be42-baa76c3c9aa6"
labels

# %% colab={"base_uri": "https://localhost:8080/", "height": 48, "referenced_widgets": ["3aed5459272145deb4a15c32f89b59a2", "9cae19141cef472db9b78b545a50c801", "468c8f016fee40038852db79fd948fee", "34153fb9e2db4636aa7495a2c6f22334", "76d3d4d39a1d43ee95052ea817fa06b3", "ca7bd808e3354ae0a4eff7c9902749d0", "6aaef6fd502044b3b61aadd50cabe46c", "dd0680981b644c3a9fc577d46ef4a7ed", "0f4072bf774d4f8ab0f560c24e537829", "69fe3694dfe64e3396f06bfc7289142f", "431d27410b0e4a0b96eda93679941851"]} executionInfo={"elapsed": 16587, "status": "ok", "timestamp": 1636890281220, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="inNQQJpIQlGE" outputId="43c09cc6-2d87-4bb0-ac37-0ca57640d1d6"
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 286, "status": "ok", "timestamp": 1636890282784, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="zL76r00tQ0J_" outputId="738f322b-c196-4ec4-ced0-701c03216c1b"
text = "Celebrating my promotion 😎"
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1636890282785, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="gakbBdpxQ3eO" outputId="b4a2b5a9-b5d1-4adb-e4b7-031024bd9e84"
ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")


# %% id="i4tOJuVJQ4HO"
def EmojiSA(text) :
  text = preprocess(text)
  encoded_input = tokenizer(text, return_tensors='pt')
  output = model(**encoded_input)
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)
  ranking = np.argsort(scores)
  ranking = ranking[::-1]
  for i in range(scores.shape[0]):
      l = labels[ranking[i]]
      s = scores[ranking[i]]
      print(f"{i+1}) {l} {np.round(float(s), 4)}")


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 241, "status": "ok", "timestamp": 1636890410693, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="Cu3-2am_RKr_" outputId="647c0a59-c7ba-4855-8551-137ea2652330"
EmojiSA('I`m sad  😖  😖  😖 ')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 265, "status": "ok", "timestamp": 1636890407546, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="dP81ocf4RTJf" outputId="dce66b8d-23dc-4cdb-87d9-1927ae887a81"
EmojiSA('I`m happy  😖  😖  😖 ')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 259, "status": "ok", "timestamp": 1636890427866, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="fHa1ad50RW_r" outputId="cbbb8149-b751-4dfb-ad98-7800d7ab2a86"
EmojiSA('I got my result  😖  😖  😖 ')

 # %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 321, "status": "ok", "timestamp": 1636890452819, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="5sHbaYC5Rb89" outputId="d3f54e81-13a2-477e-dec5-46eea6f39285"
 EmojiSA('I got my result  🥳🥳🥳')

 # %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 264, "status": "ok", "timestamp": 1636890476462, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="akGcTd3aRiBl" outputId="1b988c5a-d76b-4ab1-f4fa-183716115186"
 EmojiSA('I got my result  🤬🤬🤬')

 # %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 312, "status": "ok", "timestamp": 1636890503248, "user": {"displayName": "ahmed samy", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "08145535144646204615"}, "user_tz": 0} id="_iQ3S1J1Rnz9" outputId="f5d1e923-c6e9-492b-b977-e9520a583a95"
 EmojiSA('I got my result   🤪  🤪  🤪 ')
