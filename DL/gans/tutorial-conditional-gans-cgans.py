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
# # Importing required libraries

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
import numpy as np
import numpy.random as R
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout, MaxPooling2D, Embedding, Concatenate
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from PIL import Image
import tensorflow as tf

# %% [markdown]
# # Preparing a real dataset

# %%
files = glob.glob('/kaggle/input/footwear/Footwear/*')
files[:5]

# %%
#windows path
labels = list(set([x.split('/')[-1] for x in files]))
label_dict = {y:x for x,y in enumerate(labels)}

# %%
labels

# %%
label_dict


# %% [markdown]
# ## Load Dataset

# %%
def load_real_samples():
    x_train,y_train = [],[]
    for x,y in label_dict.items():
        files = glob.glob('/kaggle/input/footwear/Footwear/{}/*'.format(x))
        for file in files:
            x_train.append(np.asarray(Image.open(file).resize((28,28)).convert('L')))
        y_train.extend([y for value in range(len(files))])
    x_train = np.array(x_train).astype('float32').reshape(-1,28,28,1)
    return [x_train,np.array(y_train)]

dataset = load_real_samples()

# %%
dataset[0].shape[0]


# %% [markdown]
# ## Generate real Samples data

# %%
def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    
    #generating n random samples
    ix = R.randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    
    #Observe how class-labels alongside binary label(1) is return. 
    y = np.ones((n_samples, 1))
    return [X, labels], y


# %% [markdown]
# # Define Generator Network

# %%
latent_dim = 100

def build_generator(latent_dim, n_classes=3):
    in_label = Input(shape=(1,))

    #Class-Label embedding
    label = Embedding(n_classes, 50)(in_label)
    n_nodes = 7 * 7

    #Expanding class-label embedding
    label = Dense(n_nodes)(label)

    #Converting flat array as 2d image 
    label = Reshape((7,7, 1))(label)

    in_latent = Input(shape=(latent_dim,))
    n_nodes = 128 * 7 * 7

    #expanding random noise vector and converting in 7x7x128 image
    gen = Dense(n_nodes)(in_latent)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)

    #Adding class-label 2d image to this random noise image
    merge = Concatenate()([gen, label])

    #Creating features in image by upsampling 
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    #Converting final image to a 1 channel image
    out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    model = Model([in_latent, in_label], out_layer)
    return model

generator = build_generator(latent_dim)
generator.summary()


# %% [markdown]
# # Generating a latent vector to generate fake images

# %%
def generate_latent_vector(latent_dim, n_samples, n_classes=3):
    x_input = R.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = R.randint(0, n_classes, n_samples)
    return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
    z_input, labels_input = generate_latent_vector(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y


# %% [markdown]
# # Define Discriminator Network

# %%
def build_discriminator(in_shape=(28,28,1), n_classes=3):
    in_label = Input(shape=(1,))
    label = Embedding(n_classes, 50)(in_label)
    n_nodes = in_shape[0] * in_shape[1]
    label = Dense(n_nodes)(label)
    label = Reshape((in_shape[0], in_shape[1], 1))(label)
    in_image = Input(shape=in_shape)
    merge = Concatenate()([in_image, label])
    disc = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
    disc = LeakyReLU(alpha=0.2)(disc)
    disc = Conv2D(128, (3,3), strides=(2,2), padding='same')(disc)
    disc = LeakyReLU(alpha=0.2)(disc)
    disc = Flatten()(disc)
    disc = Dropout(0.4)(disc)
    out_layer = Dense(1, activation='sigmoid')(disc)
    model = Model([in_image, in_label], out_layer)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

discriminator = build_discriminator()
discriminator.summary()


# %% [markdown]
# # Build Condition GANs

# %%
def build_cgan(g_model, d_model):
    d_model.trainable = False
    gen_noise, gen_label = g_model.input
    gen_output = g_model.output
    gan_output = d_model([gen_output, gen_label])
    model = Model([gen_noise, gen_label], gan_output)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

cgan = build_cgan(generator, discriminator)
cgan.summary()

# %% [markdown]
# # train CGANs

# %%
batch_size = 64
epochs = 10
batch_per_epo = int(dataset[0].shape[0] / batch_size)
half_batch = int(batch_size / 2)
for i in tqdm(range(epochs)):
    for j in range(batch_per_epo):
        #generate real sample
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        #train discriminator on real dataset
        d_loss1, _ = discriminator.train_on_batch([X_real, labels_real], y_real)
        #generate fake sample
        [X_fake, labels], y_fake = generate_fake_samples(generator, latent_dim, half_batch)
        #train discriminator on fake dataset
        d_loss2, _ = discriminator.train_on_batch([X_fake, labels], y_fake)
        #Training CGAN
        [z_input, labels_input] = generate_latent_vector(latent_dim, batch_size)
        y_gan = np.ones((batch_size, 1))
        g_loss = cgan.train_on_batch([z_input, labels_input], y_gan)
    print('>Loss Discriminator: {}, {} , Generator: {}'.format(d_loss1,d_loss2,g_loss))
 

# %% [markdown]
# # Generate Samples

# %%
def generated_plot(examples, n):
     fig, ax = plt.subplots(1,n,figsize=(12,12))
     for index,x in enumerate(examples):
        ax[index].imshow(x,cmap='gray')
        ax[index].axis('off')
     plt.show()
     
#latent vector generator with specific class-label and not random class
def generate_latent_vector_class(latent_dim, n_samples, class_label):
    x_input = R.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = np.array([class_label for _ in range(n_samples)])
    return [z_input, labels]

# code to generat images for class=0, similarly repeat for other 2 classes
latent_vectors, labels = generate_latent_vector_class(100,10,0)
generated  = generator.predict([latent_vectors, labels])
generated = (generated + 1) / 2.0
generated_plot(generated, 10)

# %%

# %%
