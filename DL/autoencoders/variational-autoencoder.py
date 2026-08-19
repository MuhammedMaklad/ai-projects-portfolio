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

# %% id="uStVeFqnkhkH"
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# %% id="JWEHqWqboJpG"
def load_dataset():
  path = '/kaggle/input/animefacedataset'
  print("Path to dataset files:", path)
  imgsz = (64, 64)
  image_folder = os.path.join(path, "images")
  images = []
  for filename in tqdm(os.listdir(image_folder)):
    file_path = os.path.join(image_folder, filename)
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=imgsz)
    images.append(img)
  images = np.array(images)
  return images

def visualize_random_images(images):
  # Select 10 random images
  random_indices = random.sample(range(images.shape[0]), 10)
  selected_images = images[random_indices]

  # Create a 2x5 grid
  plt.figure(figsize=(15, 6))
  for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(selected_images[i])
    plt.axis('off')
  plt.tight_layout()
  plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} id="8aLupZsdsBtW" outputId="dc85b06c-348f-41b5-c956-3ba32454f0d2"
dset_anime_faces = load_dataset()

# %% colab={"base_uri": "https://localhost:8080/", "height": 266} id="XkHjbQpxsFwm" outputId="25be2462-adc5-4b00-8bb8-f4db6bc29393"
visualize_random_images(dset_anime_faces)


# %% id="eC0jOZnLyOsz"
def processing(images):
  train_images, valid_images = train_test_split(images, test_size=0.1, random_state=42)
  train_images = train_images.astype("float32") / 255.0
  valid_images = valid_images.astype("float32") / 255.0
  return train_images, valid_images


# %% id="yxWzUqK9yqmt"
x_train, valid = processing(dset_anime_faces)


# %% id="6bzYVka3Fc_-"
def create_vae(latent_dims=128):
    # Encoder
    encoder_in = tf.keras.Input(shape=(64, 64, 3))  # Input for the image

    encoder_l1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')(encoder_in)
    encoder_l1 = tf.keras.layers.BatchNormalization()(encoder_l1)
    encoder_l1 = tf.keras.layers.LeakyReLU(0.2)(encoder_l1)

    encoder_l1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(encoder_l1)
    encoder_l1 = tf.keras.layers.BatchNormalization()(encoder_l1)
    encoder_l1 = tf.keras.layers.LeakyReLU(0.2)(encoder_l1)

    encoder_l2 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(encoder_l1)
    encoder_l2 = tf.keras.layers.BatchNormalization()(encoder_l2)
    encoder_l2 = tf.keras.layers.LeakyReLU(0.2)(encoder_l2)

    encoder_l3 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(encoder_l2)
    encoder_l3 = tf.keras.layers.BatchNormalization()(encoder_l3)
    encoder_l3 = tf.keras.layers.LeakyReLU(0.2)(encoder_l3)

    encoder_l4 = tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding='same')(encoder_l3)
    encoder_l4 = tf.keras.layers.BatchNormalization()(encoder_l4)
    encoder_l4 = tf.keras.layers.LeakyReLU(0.2)(encoder_l4)

    flatten = tf.keras.layers.Flatten()(encoder_l4)

    encoder_dense = tf.keras.layers.Dense(1024)(flatten)
    encoder_dense = tf.keras.layers.BatchNormalization()(encoder_dense)
    encoder_out = tf.keras.layers.LeakyReLU(0.2)(encoder_dense)

    # Sampling layer
    class SamplingLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            mu, log_var = inputs
            batch = tf.shape(mu)[0]
            dim = tf.shape(mu)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return mu + tf.exp(0.5 * log_var) * epsilon

    # Mu and log variance layers
    mu = tf.keras.layers.Dense(latent_dims, name='mu')(encoder_out)
    log_var = tf.keras.layers.Dense(latent_dims, name='log_var')(encoder_out)

    # Custom loss layers
    class KLDivergenceLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, inputs):
            mu, log_var = inputs
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
            self.add_loss(kl_loss)
            return inputs

    class ReconstructionLossLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, inputs):
            original, reconstructed = inputs
            rec_loss = tf.reduce_mean(tf.square(original - reconstructed))
            self.add_loss(rec_loss)
            return inputs

    # Sampling
    z = SamplingLayer()([mu, log_var])

    # Decoder
    decoder_input = tf.keras.layers.Input(shape=(latent_dims,))
    x = tf.keras.layers.Dense(1024)(decoder_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Dense(8192)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Reshape((4, 4, 512))(x)

    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    decoder_output = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=5, strides=1, padding='same', activation='sigmoid')(x)

    # Decoder model
    decoder = tf.keras.Model(decoder_input, decoder_output, name='decoder')
    reconstructed = decoder(z)

    # Apply custom loss layers
    kl_layer = KLDivergenceLayer()([mu, log_var])
    rec_layer = ReconstructionLossLayer()([encoder_in, reconstructed])

    # VAE Model
    vae = tf.keras.Model(encoder_in, reconstructed, name='vae')
    vae.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

    return vae


# %% colab={"base_uri": "https://localhost:8080/"} id="hllXsoNly_xR" outputId="5722dd80-5d61-4fe5-db4e-3d5bf8bc2fbf"
print("Variational Autoencoder")
vae = create_vae()
vae.fit(x_train, x_train, epochs = 30, batch_size = 64)


# %% id="dl6PUSWlMpaW"
def predict_images(model, x_input, num_images=10):
    # Select a random subset of images
    indices = np.random.choice(len(x_input), num_images, replace=False)
    x_selected = x_input[indices]

    # Predict reconstructed images
    reconstructed = model.predict(x_selected)

    # Plot original and reconstructed images side by side
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(x_selected[i])
        plt.title("Original")
        plt.axis("off")

        # Reconstructed image
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstructed[i])
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 422} id="i3p05Q0IOG5C" outputId="480e4a36-b6b9-4d2b-b662-dc1348f0ce3d"
predict_images(vae, valid, num_images=10)


# %%
def generate_from_noise_with_image(decoder, num_images=10, latent_dim=128):
    noises = []
    for _ in range(num_images):
        noise_type = np.random.choice(['normal', 'uniform', 'exponential'])
        
        if noise_type == 'normal':
            mean = np.random.uniform(-1, 1)
            std = np.random.uniform(0.5, 2)
            noise = np.random.normal(loc=mean, scale=std, size=(latent_dim,))
        
        elif noise_type == 'uniform':
            low = np.random.uniform(-2, 0)
            high = np.random.uniform(0, 2)
            noise = np.random.uniform(low=low, high=high, size=(latent_dim,))
        
        else:  # exponential
            scale = np.random.uniform(0.5, 2)
            noise = np.random.exponential(scale=scale, size=(latent_dim,))
        
        noises.append(noise)
    
    random_noise = np.array(noises)
    generated_images = decoder.predict(random_noise)
    
    # Visualisation
    plt.figure(figsize=(20, 10))
    
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        side_length = int(np.ceil(np.sqrt(latent_dim)))
        noise_image = np.zeros((side_length, side_length))
        noise_image.flat[:latent_dim] = random_noise[i]
        
        plt.imshow(noise_image, cmap='viridis')
        plt.title(f"Noise {i+1}")
        plt.axis('off')
        
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(generated_images[i])
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 567} id="wBtSf2tROO8M" outputId="acc71e48-d129-484e-998a-667c20c244c4"
vae_decoder = vae.get_layer("decoder") 
vae_decoder.summary()

# %%
generate_from_noise_with_image(vae_decoder, num_images=10, latent_dim=128)


# %%
def plot_images(rows, cols, images, title):
    grid = np.zeros(shape=(rows*64, cols*64, 3))
    for row in range(rows):
        for col in range(cols):
            grid[row*64:(row+1)*64, col*64:(col+1)*64, :] = images[row*cols + col]

    plt.figure(figsize=(20,20))       
    plt.imshow(grid)
    plt.title(title)
    plt.show()


# %%
predictions = valid[:100]
plot_images(10,10,predictions,"ORIGINAL FACES")

# %%
predictions  = vae.predict(valid[:100])
plot_images(10,10,predictions, "RECONSTRUCTED FACES")

# %%
