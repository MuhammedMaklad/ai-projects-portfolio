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
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import mean_squared_error
from math import sqrt

# %%
# Example Dataset (e.g., MNIST Digits)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and Flatten the Data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((x_train.shape[0], -1))  # Flatten
x_test = x_test.reshape((x_test.shape[0], -1))     # Flatten

# Add Noise to the Input Data
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# %%
import matplotlib.pyplot as plt

# Function to display original and noisy images
def display_input_vs_noisy(original, noisy, n=10):
    plt.figure(figsize=(18, 6))
    for i in range(n):
        # Original Images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == n // 2:
            ax.set_title("Original Images")
        
        # Noisy Images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(noisy[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == n // 2:
            ax.set_title("Noisy Images")
    plt.show()

# Display original and noisy images
display_input_vs_noisy(x_test, x_test_noisy, n=10)


# %% [markdown]
# ![image.png](attachment:4e03e8e1-51fd-451d-b866-5e1cdaed746a.png)

# %%
# Define the Autoencoder Architecture with More Layers
def build_autoencoder(input_dim, encoding_dim):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(input_layer)
    x = layers.Dense(256, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)  # Bottleneck layer
    
    # Decoder
    x = layers.Dense(256, activation='relu')(encoded)
    x = layers.Dense(512, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='sigmoid')(x)
    
    # Autoencoder (Combining Encoder and Decoder)
    autoencoder = models.Model(input_layer, decoded)
    
    # Separate Encoder Model
    encoder = models.Model(input_layer, encoded)
    
    # Decoder Input
    encoded_input = layers.Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-3:]  # Last three layers for decoding
    x = decoder_layer[0](encoded_input)
    x = decoder_layer[1](x)
    decoded_output = decoder_layer[2](x)
    decoder = models.Model(encoded_input, decoded_output)
    
    return autoencoder, encoder, decoder


# %%
# Parameters
input_dim = 784  # Example: MNIST flattened (28x28 images)
encoding_dim = 64  # Size of the bottleneck layer

# Build the Models
autoencoder, encoder, decoder = build_autoencoder(input_dim, encoding_dim)

# Compile the Autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Display the Model Summary
autoencoder.summary()


# %%
# Plot the Model Architecture
plot_model(autoencoder, to_file='autoencoder_model.png', show_shapes=True, dpi=100)


# %%
# Train the Autoencoder
history = autoencoder.fit(x_train_noisy, x_train, epochs=20, batch_size=256, shuffle=True, validation_data=(x_test_noisy, x_test))


# %%
# Plot Training and Validation Loss
def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# Call the function to plot the history
plot_training_history(history)


# %%
# Encode and Decode the Test Data
encoded_imgs = encoder.predict(x_test_noisy)
decoded_imgs = decoder.predict(encoded_imgs)


# %%
# Visualize Original, Noisy, and Reconstructed Images
def display_images(original, noisy, reconstructed, n=10):
    plt.figure(figsize=(18, 6))
    for i in range(n):
        # Original Images
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == n // 2:
            ax.set_title("Original Images")
        
        # Noisy Images
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == n // 2:
            ax.set_title("Noisy Images")
        
        # Reconstructed Images
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == n // 2:
            ax.set_title("Reconstructed Images")
    plt.show()
display_images(x_test, x_test_noisy, decoded_imgs)

# %%

# %% [markdown]
# ![image.png](attachment:1797e8b8-8ce4-457a-aad5-3fc7a02be71f.png)

# %%
# how to calculate the PSNR for our data using peak_signal_noise_ratio
psnr_values = []
for i in range(len(x_test)):
  psnr = peak_signal_noise_ratio(x_test[i], decoded_imgs[i])
  psnr_values.append(psnr)

# Calculate the average PSNR
average_psnr = np.mean(psnr_values)
print("Average PSNR:", average_psnr)


# %%
# average of the RMSE and also average MSE
rmse_values = []
mse_values = []

for i in range(len(x_test)):
  mse = mean_squared_error(x_test[i].flatten(), decoded_imgs[i].flatten())
  rmse = sqrt(mse)
  rmse_values.append(rmse)
  mse_values.append(mse)

average_rmse = np.mean(rmse_values)
average_mse = np.mean(mse_values)

print("Average RMSE:", average_rmse)
print("Average MSE:", average_mse)


# %% [markdown]
# ![image.png](attachment:34912859-8b3d-46e2-82a3-fdcf68618268.png)
