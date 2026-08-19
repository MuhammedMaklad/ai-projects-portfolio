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

# %%
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from skimage.metrics import peak_signal_noise_ratio


# %%
# Load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize the data to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# %%
def add_noise(images, noise_factor=0.2):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)  # Ensure pixel values are between 0 and 1
    return noisy_images

# Add noise to the training and test images
x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

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
# ![image.png](attachment:a16f6a4a-b60f-4422-8d09-75d4e6215e3f.png)

# %%
# Define the Autoencoder Model
input_layer = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # Use sigmoid for output to ensure [0, 1] range

# Build the model
autoencoder = models.Model(input_layer, decoded)

# %%
# Example of creating the VAE model# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()


# %%
# Train the model
history = autoencoder.fit(x_train_noisy, x_train, epochs=50, batch_size=128, validation_data=(x_test_noisy, x_test))


# %%
# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Visualize a few original, noisy, and reconstructed images
decoded_imgs = autoencoder.predict(x_test_noisy)


# %%
# Function to display the original, noisy, and reconstructed images
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
        ax = plt.subplot(3, n, i + 1 + n)  # Adjust the index for noisy images
        plt.imshow(noisy[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == n // 2:
            ax.set_title("Noisy Images")

        # Reconstructed Images
        ax = plt.subplot(3, n, i + 1 + 2 * n)  # Adjust the index for reconstructed images
        plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == n // 2:
            ax.set_title("Reconstructed Images")
    
    plt.show()

# Display the original, noisy, and reconstructed images
display_images(x_test, x_test_noisy, decoded_imgs, n=10)


# %% [markdown]
# ![image.png](attachment:a5e70d82-6f7a-4d97-ba53-dfa273926113.png)

# %%
# Initialize lists to store the values
psnr_values = []
rmse_values = []
mse_values = []

# Iterate over the test dataset
for i in range(len(x_test)):
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(x_test[i].reshape(28, 28), decoded_imgs[i].reshape(28, 28))
    psnr_values.append(psnr)

    # Calculate MSE (Mean Squared Error)
    mse = mean_squared_error(x_test[i].flatten(), decoded_imgs[i].flatten())
    mse_values.append(mse)

    # Calculate RMSE (Root Mean Squared Error)
    rmse = sqrt(mse)
    rmse_values.append(rmse)

# Calculate the average PSNR
average_psnr = np.mean(psnr_values)
print("Average PSNR:", average_psnr)

# Calculate the average RMSE and MSE
average_rmse = np.mean(rmse_values)
average_mse = np.mean(mse_values)

print("Average RMSE:", average_rmse)
print("Average MSE:", average_mse)


# %% [markdown]
# ![image.png](attachment:0e5ac078-e2b6-4a5b-b912-411aefe3affd.png)

# %%
