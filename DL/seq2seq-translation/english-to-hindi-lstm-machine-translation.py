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

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import string
from string import digits
import matplotlib.pyplot as plt
# %matplotlib inline
import re

import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)

# Any results you write to the current directory are saved as output.

# %% _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"
lines=pd.read_csv("../input/Hindi_English_Truncated_Corpus.csv",encoding='utf-8')

# %%
lines['source'].value_counts()

# %%
lines=lines[lines['source']=='ted']

# %%
lines.head(20)

# %%
pd.isnull(lines).sum()

# %%
lines=lines[~pd.isnull(lines['english_sentence'])]

# %%
lines.drop_duplicates(inplace=True)

# %% [markdown]
# * ### Let us pick any 25000 rows from the dataset.

# %%
lines=lines.sample(n=25000,random_state=42)
lines.shape

# %%
# Lowercase all characters
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.lower())
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.lower())

# %%
# Remove quotes
lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub("'", '', x))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub("'", '', x))

# %%
exclude = set(string.punctuation) # Set of all special characters
# Remove all the special characters
lines['english_sentence']=lines['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

# %%
# Remove all numbers from text
remove_digits = str.maketrans('', '', digits)
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.translate(remove_digits))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.translate(remove_digits))

lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

# Remove extra spaces
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.strip())
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.strip())
lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub(" +", " ", x))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x))


# %%
# Add start and end tokens to target sequences
lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x : 'START_ '+ x + ' _END')

# %%
lines.head()

# %%
### Get English and Hindi Vocabulary
all_eng_words=set()
for eng in lines['english_sentence']:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

all_hindi_words=set()
for hin in lines['hindi_sentence']:
    for word in hin.split():
        if word not in all_hindi_words:
            all_hindi_words.add(word)

# %%
len(all_eng_words)

# %%
len(all_hindi_words)

# %%
lines['length_eng_sentence']=lines['english_sentence'].apply(lambda x:len(x.split(" ")))
lines['length_hin_sentence']=lines['hindi_sentence'].apply(lambda x:len(x.split(" ")))

# %%
lines.head()

# %%
lines[lines['length_eng_sentence']>30].shape

# %%
lines=lines[lines['length_eng_sentence']<=20]
lines=lines[lines['length_hin_sentence']<=20]

# %%
lines.shape

# %%
print("maximum length of Hindi Sentence ",max(lines['length_hin_sentence']))
print("maximum length of English Sentence ",max(lines['length_eng_sentence']))

# %%
max_length_src=max(lines['length_hin_sentence'])
max_length_tar=max(lines['length_eng_sentence'])

# %%
input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_hindi_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_hindi_words)
num_encoder_tokens, num_decoder_tokens

# %%
num_decoder_tokens += 1 #for zero padding


# %%
input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

# %%
reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

# %%
lines = shuffle(lines)
lines.head(10)

# %% [markdown]
# ### Split the data into train and test

# %%
X, y = lines['english_sentence'], lines['hindi_sentence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)
X_train.shape, X_test.shape

# %% [markdown]
# ### Let us save this data

# %%
X_train.to_pickle('X_train.pkl')
X_test.to_pickle('X_test.pkl')


# %%
def generate_batch(X = X_train, y = y_train, batch_size = 128):
    ''' Generate a batch of data '''
    while True:  # Infinite loop to keep generating batches for training
        for j in range(0, len(X), batch_size):  # Loop through the dataset in steps of batch_size
            # Initialize empty arrays for encoder input, decoder input, and decoder target data
            encoder_input_data = np.zeros((batch_size, max_length_src), dtype='float32')  # Encoder input sequences
            decoder_input_data = np.zeros((batch_size, max_length_tar), dtype='float32')  # Decoder input sequences
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens), dtype='float32')  # One-hot encoded target sequences

            # Loop through each input-output pair in the current batch
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                # Process the input text (source sequence) for the encoder
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = input_token_index[word]  # Map word to token index for encoder input

                # Process the target text (target sequence) for the decoder
                for t, word in enumerate(target_text.split()):
                    if t < len(target_text.split()) - 1:  # Exclude the last word from decoder input
                        decoder_input_data[i, t] = target_token_index[word]  # Map word to token index for decoder input
                    if t > 0:  # Skip the first word in the target sequence for decoder target (it's the "START_" token)
                        # One-hot encode the target sequence for the decoder output (excluding "START_" token)
                        decoder_target_data[i, t - 1, target_token_index[word]] = 1.

            # Yield the batch, where the encoder input data, decoder input data are the inputs
            # and decoder target data is the output for training the model
            yield([encoder_input_data, decoder_input_data], decoder_target_data)



# %% [markdown]
# ### Encoder-Decoder Architecture

# %%
latent_dim=300

# %% [markdown]
# ![image.png](attachment:d364e84c-39b5-4c17-90ef-7f6696d4eba8.png)

# %%
# Encoder
encoder_inputs = Input(shape=(None,))  # Input layer, where shape=(None,) indicates variable length sequences
enc_emb = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)  # Embedding layer for input sequences
encoder_lstm = LSTM(latent_dim, return_state=True)  # LSTM layer with latent_dim as the output size
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)  # LSTM outputs and states (h and c)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]  # Encoder's final states (used for decoder initialization)


# %%
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))  # Input layer for the decoder
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)  # Embedding layer for decoder inputs
dec_emb = dec_emb_layer(decoder_inputs)  # Apply embedding layer to decoder inputs

# Set up the LSTM layer for the decoder
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)  
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

# Dense layer to output probabilities over the decoder vocabulary (softmax)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model: encoder_inputs and decoder_inputs will output decoder_outputs
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# %%
model.compile(optimizer='adam', loss='categorical_crossentropy')

# %%
model.summary()

# %%
train_samples = len(X_train)
val_samples = len(X_test)
batch_size = 128
epochs = 100

# %%
history = model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                    steps_per_epoch = train_samples//batch_size,
                    epochs=epochs,
                    validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                    validation_steps = val_samples//batch_size)



# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()

# %%
model.save_weights('nmt_weights.h5')

# %%
# Encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)


# %%
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+ sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


# %%
train_gen = generate_batch(X_train, y_train, batch_size = 1)
k=-1


# %%
k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input English sentence:', X_train[k:k+1].values[0])
print('Actual Hindi Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation:', decoded_sentence[:-4])

# %%
k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input English sentence:', X_train[k:k+1].values[0])
print('Actual Hindi Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation:', decoded_sentence[:-4])

# %%
k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input English sentence:', X_train[k:k+1].values[0])
print('Actual Hindi Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation:', decoded_sentence[:-4])

# %%
k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input English sentence:', X_train[k:k+1].values[0])
print('Actual Hindi Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation:', decoded_sentence[:-4])

# %%
k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input English sentence:', X_train[k:k+1].values[0])
print('Actual Hindi Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation:', decoded_sentence[:-4])

# %%
