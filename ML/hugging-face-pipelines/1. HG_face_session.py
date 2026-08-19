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

# %% [markdown] id="KpFLbCN2VfQ0"
# 1- Text classification is a common NLP task that assigns a label or class to text.
#
# 2- Some of the largest companies run text classification in production for a wide range of practical applications.
#
# 3- One of the most popular forms of text classification is sentiment analysis, which assigns a label like 🙂 positive, 🙁 negative, or 😐 neutral to a sequence of text.
#
# This guide will show you how to:
#
#   1- Finetune DistilBERT on the IMDb dataset to determine whether a movie review is positive or negative.
#
#   2- Use your finetuned model for inference.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4342, "status": "ok", "timestamp": 1727455584130, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="nByOP05XVTqN" outputId="d5d4547b-116d-4527-c721-8f7c156edf41"
pip install transformers datasets evaluate accelerate

# %% [markdown] id="mK7XoxnlVjAd"
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 465, "status": "ok", "timestamp": 1727455584592, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="9h0X4X3aVjLW" outputId="20518f33-e1dd-413f-b3da-efe9fe1400fd"
# Use a pipeline as a high-level helper
from transformers import pipeline

# This model does not appear to support text classification
# pipe = pipeline("text-classification", model="Qwen/Qwen2.5-Math-RM-72B", trust_remote_code=True)

# Try this model instead
pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
pipe("A soccer game with multiple males playing. Some men are playing a sport.")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1482, "status": "ok", "timestamp": 1727455586070, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="W5UwNh8UVYzU" outputId="76d28472-af9b-4759-af74-48fbccd63ba0"
from transformers import pipeline

classifier = pipeline("text-classification", model = "roberta-large-mnli")
classifier("A soccer game with multiple males playing. Some men are playing a sport.")
## [{'label': 'ENTAILMENT', 'score': 0.98}]

# %% executionInfo={"elapsed": 6434, "status": "ok", "timestamp": 1727455592501, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="idGXhDPbVY29"
from datasets import load_dataset

imdb = load_dataset("imdb")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1727455592502, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="7chb-eatVY6f" outputId="c393c3f1-a17c-46f9-d1f8-5bf094e98e87"
imdb["test"][0]

# %% [markdown] id="HoXqheUrWPXt"
# There are two fields in this dataset:
#
#     text: the movie review text.
#
#     label: a value that is either 0 for a negative review or 1 for a positive review.

# %% executionInfo={"elapsed": 438, "status": "ok", "timestamp": 1727455592934, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="JZGKP_E0VY-L"
from transformers import AutoTokenizer  # for personal using , finetuning , custom task
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased") #200


# %% executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1727455592934, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="SCid_tfmWVi9"
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)  # fixed size Example 1024


# %% colab={"base_uri": "https://localhost:8080/", "height": 113} executionInfo={"elapsed": 133996, "status": "ok", "timestamp": 1727455726921, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="uC-n_0laWVlU" outputId="956a9727-e366-4081-937d-dd4b2008ac3e"
tokenized_imdb = imdb.map(preprocess_function, batched=True)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1727455726921, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="rr823gToiDzc" outputId="30e4bee3-ba7f-4cde-e7b6-eca0238af250"
tokenized_imdb["train"][0]   #id for every word

# %% executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1727455726921, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="DtNslMSoWVo5"
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% colab={"base_uri": "https://localhost:8080/", "height": 49} executionInfo={"elapsed": 2438, "status": "ok", "timestamp": 1727455729346, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4LMb9jRjWo9X" outputId="fb493cf5-bc0d-4a4c-f4b7-5e8d552ae96c"
import evaluate

accuracy = evaluate.load("accuracy")

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1727455729346, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="I-N1SR6_Wo_8"
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1727455729346, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="x5wkCC5lWpDa"
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# %% colab={"base_uri": "https://localhost:8080/", "height": 104} executionInfo={"elapsed": 4008, "status": "ok", "timestamp": 1727455733349, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="sLxAg1UAW1Bd" outputId="c356037f-3279-4d94-9858-622d5ac00fc5"
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 145} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1727455734965, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Hj3F-1sqXLQJ" outputId="64132b69-d069-4d29-e0da-cee548e500f1"
from huggingface_hub import notebook_login

notebook_login()

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/", "height": 75} id="JBcDJRsRW8Q7"
training_args = TrainingArguments(
    output_dir="SentimentAnalysis",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# %% colab={"base_uri": "https://localhost:8080/", "height": 141} executionInfo={"elapsed": 1904, "status": "error", "timestamp": 1727341360524, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="6yGM9hoKW8Uc" outputId="7470e9ca-5513-4e37-e3ed-d717e0a6376b"
trainer.push_to_hub()

# %% colab={"base_uri": "https://localhost:8080/", "height": 350} executionInfo={"elapsed": 38885, "status": "ok", "timestamp": 1727341399400, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="w2JRPen8ZLRN" outputId="cfae86d5-6697-4096-bfee-08cc0141a6fa"
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="Yousry609/SentimentAnalysis")
classifier(text)
