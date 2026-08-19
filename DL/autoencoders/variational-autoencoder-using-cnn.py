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

# %% id="intensive-church" papermill={"duration": 14.517669, "end_time": "2024-12-10T18:34:33.437277", "exception": false, "start_time": "2024-12-10T18:34:18.919608", "status": "completed"}
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import numpy as np

# %% [markdown] papermill={"duration": 0.005503, "end_time": "2024-12-10T18:34:33.447817", "exception": false, "start_time": "2024-12-10T18:34:33.442314", "status": "completed"}
# ![image.png](attachment:a5f488c3-3d9e-45f1-a94c-15560af47900.png)

# %% id="banned-mexican" outputId="34d857c2-22e8-4f61-d110-c29b1d135165" papermill={"duration": 0.466741, "end_time": "2024-12-10T18:34:33.918854", "exception": false, "start_time": "2024-12-10T18:34:33.452113", "status": "completed"}
image_dir = "/kaggle/input/animefacedataset/images/"
images = [os.path.join(image_dir, image) for image in os.listdir(image_dir)]
images[:2]

# %% papermill={"duration": 1.027164, "end_time": "2024-12-10T18:34:34.951123", "exception": false, "start_time": "2024-12-10T18:34:33.923959", "status": "completed"}
from sklearn.model_selection import train_test_split
X_train,X_test = train_test_split(images,test_size = 0.2,shuffle=True)

# %% id="alpha-sunglasses" papermill={"duration": 0.013372, "end_time": "2024-12-10T18:34:34.969232", "exception": false, "start_time": "2024-12-10T18:34:34.955860", "status": "completed"}
# preprocess
image_size = 64

def preprocess(image):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image = image / 255.0
    image = tf.reshape(image, shape = (image_size, image_size, 3,))
    return image


# %% id="greek-usage" papermill={"duration": 1.112503, "end_time": "2024-12-10T18:34:36.086428", "exception": false, "start_time": "2024-12-10T18:34:34.973925", "status": "completed"}
batch_size = 128

training_dataset = tf.data.Dataset.from_tensor_slices((X_train))
training_dataset = training_dataset.map(preprocess)
training_dataset = training_dataset.shuffle(1000).batch(batch_size)

# %% id="resident-defeat" outputId="d214eb75-e187-4e6b-e4a7-df7fb2106ea2" papermill={"duration": 0.012684, "end_time": "2024-12-10T18:34:36.103924", "exception": false, "start_time": "2024-12-10T18:34:36.091240", "status": "completed"}
len(training_dataset)

# %% id="experimental-departure" outputId="a6dd9db9-f27f-4fa0-8626-e839f7b912be" papermill={"duration": 5.033468, "end_time": "2024-12-10T18:34:41.142107", "exception": false, "start_time": "2024-12-10T18:34:36.108639", "status": "completed"}
# visualize some of them
fig, axes = plt.subplots(5,5, figsize = (14,14))
sample = training_dataset.unbatch().take(25)
sample = [image for image in sample]

idx = 0
for row in range(5):
    for column in range(5):
        axes[row, column].imshow(sample[idx])
        idx+=1

# %% papermill={"duration": 0.075123, "end_time": "2024-12-10T18:34:41.240919", "exception": false, "start_time": "2024-12-10T18:34:41.165796", "status": "completed"}
batch_size = 128

testing_dataset = tf.data.Dataset.from_tensor_slices((X_test))
testing_dataset = testing_dataset.map(preprocess)
testing_dataset = testing_dataset.shuffle(1000).batch(batch_size)

# %% papermill={"duration": 0.033009, "end_time": "2024-12-10T18:34:41.299382", "exception": false, "start_time": "2024-12-10T18:34:41.266373", "status": "completed"}
len(testing_dataset)

# %% papermill={"duration": 4.89963, "end_time": "2024-12-10T18:34:46.223965", "exception": false, "start_time": "2024-12-10T18:34:41.324335", "status": "completed"}
# visualize some of them
fig, axes = plt.subplots(5,5, figsize = (14,14))
sample = testing_dataset.unbatch().take(25)
sample = [image for image in sample]

idx = 0
for row in range(5):
    for column in range(5):
        axes[row, column].imshow(sample[idx])
        idx+=1

# %% id="reported-investor" papermill={"duration": 0.056199, "end_time": "2024-12-10T18:34:46.333261", "exception": false, "start_time": "2024-12-10T18:34:46.277062", "status": "completed"}
# build the model

latent_dim = 512

# %% id="employed-luther" papermill={"duration": 0.055444, "end_time": "2024-12-10T18:34:46.434972", "exception": false, "start_time": "2024-12-10T18:34:46.379528", "status": "completed"}
from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, Conv2DTranspose, Input, Flatten, BatchNormalization, Lambda, Reshape, Activation
from keras.layers import LeakyReLU
from keras.activations import selu
from keras.layers import Multiply, Add
from keras.optimizers import Adam

from keras import backend as K


# %% id="irish-bernard" papermill={"duration": 0.280174, "end_time": "2024-12-10T18:34:46.759365", "exception": false, "start_time": "2024-12-10T18:34:46.479191", "status": "completed"}
K.clear_session()

# %% [markdown]
# ![image.png](attachment:9338528f-35a3-4c8e-b958-3d5669286825.png)

# %% id="appreciated-discovery" outputId="6256a4ed-e516-460c-da06-709020316585" papermill={"duration": 0.294663, "end_time": "2024-12-10T18:34:47.102349", "exception": false, "start_time": "2024-12-10T18:34:46.807686", "status": "completed"}
from tensorflow.keras.layers import Lambda
import tensorflow as tf

# Encoder model
encoder_input = Input(shape=(64, 64, 3))

x = Conv2D(32, kernel_size=5, activation=LeakyReLU(0.02), strides=1, padding='same')(encoder_input)
x = BatchNormalization()(x)

filter_size = [64, 128, 256, 512]
for i in filter_size:
    x = Conv2D(i, kernel_size=5, activation=LeakyReLU(0.02), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(1024, activation=selu)(x)
encoder_output = BatchNormalization()(x)

# Sampling layer
mu = Dense(latent_dim)(encoder_output)
log_var = Dense(latent_dim)(encoder_output)

# Use a Lambda layer to apply TensorFlow operations
def sampling_func(inputs):
    mu, log_var = inputs
    epsilon = tf.random.normal(shape=tf.shape(mu))  # Sample epsilon
    sigma = tf.exp(0.5 * log_var)  # Compute sigma
    z = mu + sigma * epsilon  # Compute z
    return z

z = Lambda(sampling_func)([mu, log_var])

encoder = Model(encoder_input, outputs=[mu, log_var, z], name='encoder')
encoder.summary()


# %% id="beneficial-darkness" outputId="7ae1e781-2c87-423c-b9c2-1f6ccb7d4500" papermill={"duration": 0.233381, "end_time": "2024-12-10T18:34:47.388555", "exception": false, "start_time": "2024-12-10T18:34:47.155174", "status": "completed"}
# build the decoder

decoder = Sequential()
decoder.add(Dense(1024, activation = selu, input_shape = (latent_dim, )))
decoder.add(BatchNormalization())

decoder.add(Dense(8192, activation = selu))
decoder.add(Reshape((4,4,512)))

decoder.add(Conv2DTranspose(256, (5,5), activation = LeakyReLU(0.02), strides = 2, padding = 'same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(128, (5,5), activation = LeakyReLU(0.02), strides = 2, padding = 'same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(64, (5,5), activation = LeakyReLU(0.02), strides = 2, padding = 'same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(32, (5,5), activation = LeakyReLU(0.02), strides = 2, padding = 'same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(3, (5,5), activation = "sigmoid", strides = 1, padding = 'same'))
decoder.add(BatchNormalization())

decoder.summary()

# %% [markdown]
# ![image.png](attachment:e921eb6b-39ba-4a27-816d-39e69a5f0342.png)

# %% id="swiss-popularity" papermill={"duration": 0.062088, "end_time": "2024-12-10T18:34:47.503202", "exception": false, "start_time": "2024-12-10T18:34:47.441114", "status": "completed"}
import tensorflow as tf

# Reconstruction loss function (MSE)
def reconstruction_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# KL divergence loss function
def kl_loss(mu, log_var):
    # Use the formula for KL divergence between the learned distribution and a standard normal
    kl_loss_value = -0.5 * tf.reduce_mean(
        1 + log_var - tf.square(mu) - tf.exp(log_var)
    )
    return kl_loss_value

# Total VAE loss = reconstruction loss + KL divergence loss
def vae_loss(y_true, y_pred, mu, log_var):
    # Reconstruction loss (MSE)
    recon_loss = reconstruction_loss(y_true, y_pred)
    
    # KL divergence loss with a scaling factor
    kl_loss_value = kl_loss(mu, log_var)
    
    # Combine the losses (scaled KL loss by 1/(image_width * image_height) or any suitable factor)
    total_loss = recon_loss + (1 / (64*64)) * kl_loss_value  # The factor 64*64 is used for scaling KL divergence
    
    return total_loss



# %% id="valid-fisher" papermill={"duration": 0.058619, "end_time": "2024-12-10T18:34:47.612156", "exception": false, "start_time": "2024-12-10T18:34:47.553537", "status": "completed"}
import matplotlib.pyplot as plt

def save_images(decoder, epoch, step, random_vector):
    generated_images = decoder(random_vector)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(generated_images[i].numpy().reshape(64, 64, 3))
        ax.axis('off')

    os.makedirs('output',exist_ok=True)
    plt.savefig(f"output/reconstructed_images_epoch_{epoch}_step_{step}.png")
    plt.close()



# %% id="0wryyAYN-pzQ" outputId="030c8196-c247-4f5c-8065-b6cb2e1063ee" papermill={"duration": 2.214644, "end_time": "2024-12-10T18:34:49.875303", "exception": false, "start_time": "2024-12-10T18:34:47.660659", "status": "completed"}

from keras.optimizers import Adam

random_vector = tf.random.normal(shape = (25, latent_dim,))
save_images(decoder, 0, 0, random_vector)

# %% papermill={"duration": 0.060245, "end_time": "2024-12-10T18:34:49.984042", "exception": false, "start_time": "2024-12-10T18:34:49.923797", "status": "completed"}
# conbine encoder and decoder
mu, log_var, z = encoder(encoder_input)
reconstructed = decoder(z)
model = Model(encoder_input, reconstructed, name ="vae")

# %% id="cutting-highway" outputId="f521a177-c6f5-47e4-da49-d85fe4d8787d" papermill={"duration": 6121.340778, "end_time": "2024-12-10T20:16:51.373554", "exception": false, "start_time": "2024-12-10T18:34:50.032776", "status": "completed"}
import tensorflow as tf

# Assuming the encoder, decoder, and training_dataset/testing_dataset are defined
optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)
epochs = 50

# Initialize dictionaries to store loss values
loss_history = {
    "train_total_loss": [],
    "train_recon_loss": [],
    "train_kl_loss": [],
    "test_total_loss": [],
    "test_recon_loss": [],
    "test_kl_loss": []
}

random_vector = tf.random.normal(shape = (25, latent_dim,))

for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    
    # Training Phase
    train_total_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    train_steps = 0
    
    for step, training_batch in enumerate(training_dataset):
        with tf.GradientTape() as tape:
            # Get encoder output (mu, log_var, z) and decoder output
            mu, log_var, z = encoder(training_batch)
            reconstructed = decoder(z)
            
            # Compute total VAE loss
            loss = vae_loss(training_batch, reconstructed, mu, log_var)
            
            # Compute individual losses
            recon_loss = reconstruction_loss(training_batch, reconstructed)
            kl_loss_value = kl_loss(mu, log_var)

        # Compute gradients and apply them
        grads = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))
        if step % 10 == 0:
            save_images(decoder, epoch, step, random_vector)
            
        # Update loss accumulators
        train_total_loss += loss.numpy()
        train_recon_loss += recon_loss.numpy()
        train_kl_loss += kl_loss_value.numpy()
        train_steps += 1

        # Display losses periodically
        if step % 50 == 0:
            print(
                f"Epoch: {epoch} - Step: {step} - Total Loss: {loss.numpy():.4f} - "
                f"Reconstruction Loss: {recon_loss.numpy():.4f} - KL Loss: {kl_loss_value.numpy():.4f}"
            )

    # Store average training losses
    loss_history["train_total_loss"].append(train_total_loss / train_steps)
    loss_history["train_recon_loss"].append(train_recon_loss / train_steps)
    loss_history["train_kl_loss"].append(train_kl_loss / train_steps)

    # Testing Phase
    total_test_loss = 0
    total_test_recon_loss = 0
    total_test_kl_loss = 0
    total_test_steps = 0
    
    for step, testing_batch in enumerate(testing_dataset):
        # Get encoder output (mu, log_var, z) and decoder output
        mu, log_var, z = encoder(testing_batch)
        reconstructed = decoder(z)
        
        # Compute losses
        test_loss = vae_loss(testing_batch, reconstructed, mu, log_var)
        recon_loss = reconstruction_loss(testing_batch, reconstructed)
        kl_loss_value = kl_loss(mu, log_var)
        
        # Update test loss accumulators
        total_test_loss += test_loss.numpy()
        total_test_recon_loss += recon_loss.numpy()
        total_test_kl_loss += kl_loss_value.numpy()
        total_test_steps += 1

        
    # Store average testing losses
    loss_history["test_total_loss"].append(total_test_loss / total_test_steps)
    loss_history["test_recon_loss"].append(total_test_recon_loss / total_test_steps)
    loss_history["test_kl_loss"].append(total_test_kl_loss / total_test_steps)

    # Display average test losses
    print(
        f"Epoch: {epoch} - Average Test Loss: {loss_history['test_total_loss'][-1]:.4f} - "
        f"Reconstruction Loss: {loss_history['test_recon_loss'][-1]:.4f} - "
        f"KL Loss: {loss_history['test_kl_loss'][-1]:.4f}"
    )

# %% papermill={"duration": 0.373597, "end_time": "2024-12-10T20:16:51.819534", "exception": false, "start_time": "2024-12-10T20:16:51.445937", "status": "completed"}
# Plot the total loss
plt.figure(figsize=(12, 8))
plt.plot(loss_history["train_total_loss"], label="Training Total Loss")
plt.plot(loss_history["test_total_loss"], label="Testing Total Loss")
plt.title("Total Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# %% papermill={"duration": 0.389776, "end_time": "2024-12-10T20:16:52.287572", "exception": false, "start_time": "2024-12-10T20:16:51.897796", "status": "completed"}
# Plot the reconstruction loss
plt.figure(figsize=(12, 8))
plt.plot(loss_history["train_recon_loss"], label="Training Reconstruction Loss")
plt.plot(loss_history["test_recon_loss"], label="Testing Reconstruction Loss")
plt.title("Reconstruction Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()



# %% papermill={"duration": 0.378945, "end_time": "2024-12-10T20:16:52.735323", "exception": false, "start_time": "2024-12-10T20:16:52.356378", "status": "completed"}
# Plot the KL divergence loss
plt.figure(figsize=(12, 8))
plt.plot(loss_history["train_kl_loss"], label="Training KL Loss")
plt.plot(loss_history["test_kl_loss"], label="Testing KL Loss")
plt.title("KL Divergence Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()


# %% id="brutal-apple" papermill={"duration": 0.900246, "end_time": "2024-12-10T20:16:53.713150", "exception": true, "start_time": "2024-12-10T20:16:52.812904", "status": "failed"}
import cv2
import os
import imageio
from IPython.display import display, Image

# Set the path to the directory containing the images
output_dir = "output/"

# Get a list of all files in the output directory
files = os.listdir(output_dir)

# Sort the files to ensure they are in the correct order
files.sort()

# Initialize a list to store the images
images = []

# Loop through the files and read the images using cv2
for file in files:
    # Construct full file path
    file_path = os.path.join(output_dir, file)

    # Check if the file is an image (for example, with .jpg or .png extension)
    if file.endswith('.jpg') or file.endswith('.png'):
        # Read the image
        img = cv2.imread(file_path)

        # Convert the image from BGR (OpenCV format) to RGB (for imageio)45EWQ2aq
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Append the image to the list
        images.append(img_rgb)

# Define the output gif path
output_gif_path = "output/animation.gif"

# Create and save the gif
imageio.mimsave(output_gif_path, images, duration=0.1)  # Adjust duration for the speed of the gif

# Display the GIF
display(Image(filename=output_gif_path))

print(f"GIF saved and displayed at: {output_gif_path}")


# %% id="controlling-replacement" papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"}
