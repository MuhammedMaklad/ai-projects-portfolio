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

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input,Concatenate,Dense,Reshape,Flatten,Conv2D,Conv2DTranspose,LeakyReLU,Dropout,Embedding

from matplotlib import pyplot as plt
import os
import glob as gb
import cv2
import numpy as np

# %% [markdown]
# # Load dataset

# %%
#loading data and resize it and collect it in one folder
trainpath = '/kaggle/input/footwear/Footwear/'
new_size=32    
X_train = []
y_train = []
for folder in  os.listdir(trainpath ) : 
    print( 'folder name is : ', folder)
    files = gb.glob(pathname= str( trainpath  + folder + '/*.jpg'))
    print( 'numbers of images in folder are : ', len(files))
    print(' start reading images')
    for file in files: 
        image_class = {'Boot': 0, 'Sandal': 1,'Shoe':2}
        orignal_image = cv2.imread(file)
        image = cv2.cvtColor(orignal_image, cv2.COLOR_BGR2GRAY) 
        resized_image = cv2.resize(image , (new_size,new_size))
        resized_image = resized_image /255.0 #Generator uses sigmoid activation so rescale 
        X_train.append(resized_image)
        y_train.append(image_class[folder])
    print('image reading ...finished')
print('--------------------------------------------------')        
#check items in X_test
print("items in X_train is:       ",len(X_train) , " items") 
print("items in y_train is:       ",len(y_train) , " items") 

# %% [markdown]
# ## Show samples

# %%
#showing training images with labels
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train ),16))) : 
    plt.subplot(4,4,n+1)
    plt.imshow(X_train [i],cmap='gray')   
    plt.axis('off')
    classes = {'Boot': 0, 'Sandal': 1,'Shoe':2}
    def ImageClass(n):
        for x , y in classes.items():
            if n == y :
                return x
    plt.title(ImageClass(y_train[i]))

# %%
#converting all TRAIN data to array
X_train = np.array(X_train)
y_train = np.array(y_train)
print("X_train shape  :" ,X_train.shape)
print("y_train shape :", y_train.shape)


# %% [markdown]
# # Define Discriminator

# %%
def define_discriminator(image_shape=(32, 32, 1), n_classes=3):
    """
    Define a conditional discriminator for a GAN.

    Args:
    - image_shape (tuple): Shape of the input image (height, width, channels).
    - n_classes (int): Number of classes for conditional input.

    Returns:
    - model (tf.keras.Model): Compiled discriminator model.
    """

    # Label input
    label_input = Input(shape=(1,), name='label_input')
    # Embedding layer for the label
    label_embedding = Embedding(input_dim=n_classes, output_dim=50, name='label_embedding')(label_input)
    # Dense layer to scale the embedding to match image dimensions
    nodes = image_shape[0] * image_shape[1]  # Example: 32 * 32 = 1024
    label_dense = Dense(nodes, name='label_dense')(label_embedding)
    # Reshape to match the image's spatial dimensions
    label_reshape = Reshape((image_shape[0], image_shape[1], 1), name='label_reshape')(label_dense)

    # Image input
    image_input = Input(shape=image_shape, name='image_input')
    # Concatenate the image and label as channels
    merged_input = Concatenate(name='concat')([image_input, label_reshape])

    # Convolutional layers for feature extraction
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv1')(merged_input)
    x = LeakyReLU(alpha=0.2, name='lrelu1')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv2')(x)
    x = LeakyReLU(alpha=0.2, name='lrelu2')(x)

    # Flatten the feature maps and apply dropout
    x = Flatten(name='flatten')(x)
    x = Dropout(0.4, name='dropout')(x)

    # Output layer: Binary classification (real/fake)
    output = Dense(1, activation='sigmoid', name='output')(x)

    # Define the model
    model = Model([image_input, label_input], output, name='discriminator')

    # Compile the model
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# %%
test_discr = define_discriminator()
print(test_discr.summary())


# %% [markdown]
# # Define Generator

# %%
def define_generator(latent_dim, n_classes=3):
    """
    Define a conditional generator for a GAN.

    Args:
    - latent_dim (int): Dimensionality of the latent space.
    - n_classes (int): Number of classes for conditional input.

    Returns:
    - model (tf.keras.Model): Generator model.
    """

    # Label input
    label_input = Input(shape=(1,), name='label_input')  # Shape: (batch_size, 1)
    # Embedding layer for the label
    label_embedding = Embedding(input_dim=n_classes, output_dim=50, name='label_embedding')(label_input)  # Shape: (batch_size, 1, 50)
    # Dense layer to scale the embedding to match image dimensions
    label_dense = Dense(8 * 8, name='label_dense')(label_embedding)  # Shape: (batch_size, 1, 64)
    # Reshape to match the spatial dimensions
    label_reshape = Reshape((8, 8, 1), name='label_reshape')(label_dense)  # Shape: (batch_size, 8, 8, 1)

    # Latent space input
    latent_input = Input(shape=(latent_dim,), name='latent_input')  # Shape: (batch_size, latent_dim)
    # Foundation for the 8x8 image
    n_nodes = 128 * 8 * 8
    latent_dense = Dense(n_nodes, name='latent_dense')(latent_input)  # Shape: (batch_size, 8192)
    latent_reshape = LeakyReLU(alpha=0.2, name='latent_lrelu')(latent_dense)
    latent_reshape = Reshape((8, 8, 128), name='latent_reshape')(latent_reshape)  # Shape: (batch_size, 8, 8, 128)

    # Merge label embedding and latent input
    merged_input = Concatenate(name='concat')([latent_reshape, label_reshape])  # Shape: (batch_size, 8, 8, 129)

    # Upsampling to 16x16
    x = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', name='upsample_16')(merged_input)  # Shape: (batch_size, 16, 16, 128)
    x = LeakyReLU(alpha=0.2, name='lrelu_16')(x)

    # Upsampling to 32x32
    x = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', name='upsample_32')(x)  # Shape: (batch_size, 32, 32, 128)
    x = LeakyReLU(alpha=0.2, name='lrelu_32')(x)

    # Output layer
    output = Conv2D(1, kernel_size=(8, 8), activation='sigmoid', padding='same', name='output_layer')(x)  # Shape: (batch_size, 32, 32, 1)

    # Define the model
    model = Model([latent_input, label_input], output, name='generator')

    return model


# %%
latent_dim = 100
generator = define_generator(latent_dim=latent_dim, n_classes=3)
generator.summary()


# %% [markdown]
# # Define Conditional GANs

# %%
def define_gan(g_model, d_model):
    d_model.trainable = False  #Discriminator is trained separately. So set to not trainable.

    ## connect generator and discriminator...
    # first, get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input  #Latent vector size and label size
    # get image output from the generator model
    gen_output = g_model.output  #32x32x1

    # generator image output and corresponding input label are inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# %% [markdown]
# ## Generate Real Samples

# %%
def generate_real_samples(images, labels, n_samples):
    """
    Generate a batch of real samples for training the discriminator.

    Args:
    - images (numpy.ndarray): Array of images (e.g., shape: (num_samples, height, width, channels)).
    - labels (numpy.ndarray): Array of corresponding labels (e.g., shape: (num_samples,)).
    - n_samples (int): Number of samples to generate.

    Returns:
    - X (list): A list containing the selected images and their corresponding labels.
    - y (numpy.ndarray): Array of class labels for the discriminator, set to 1 (real samples).
    """

    # Randomly select indices for the batch
    indices = np.random.randint(0, images.shape[0], n_samples)

    # Select images and labels based on the indices
    selected_images = images[indices]
    selected_labels = labels[indices]

    # Generate target labels for the discriminator, indicating real samples (label=1)
    y = np.ones((n_samples, 1))

    return [selected_images, selected_labels], y


# %%
generate_real_samples(X_train,y_train,1)

# %%
ix = np.random.randint(0, 100, 4)
print(ix)

# %%
x_input = np.random.randn(100 * 64)
print(x_input.shape)
z_input = x_input.reshape(64, 100)
z_input.shape


# %% [markdown]
# # Generate Fake Samples using Latent Points

# %%
def generate_latent_points(latent_dim, n_samples, n_classes=3):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples) #6400
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim) # 64,100
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# %%
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))  #Label=0 indicating they are fake
    return [images, labels_input], y


# %% [markdown]
# # Train CGANs

# %%
def train(g_model, d_model, gan_model, X, y, latent_dim, n_epochs=100, n_batch=64):
    """
    Train the conditional GAN model.

    Args:
    - g_model: Generator model.
    - d_model: Discriminator model.
    - gan_model: Combined GAN model.
    - X: Training images.
    - y: Training labels.
    - latent_dim: Size of the latent space.
    - n_epochs: Number of training epochs.
    - n_batch: Batch size.

    Saves:
    - The trained generator model as 'conditional_generator11.h5'.
    """
    # Calculate the number of batches per epoch and half-batch size
    bat_per_epo = int(X.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    # Manually enumerate epochs
    for epoch in range(n_epochs):
        for batch in range(bat_per_epo):
            # Train the discriminator
            # Select a half batch of real samples
            [X_real, labels_real], y_real = generate_real_samples(X, y, half_batch)
            d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)

            # Generate a half batch of fake samples
            [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss_fake, _ = d_model.train_on_batch([X_fake, labels_fake], y_fake)

            # Prepare the latent points and corresponding labels
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)

            # Create inverted labels (all ones) for the generator
            y_gan = np.ones((n_batch, 1))

            # Train the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)

            # Handle generator loss if it is a list of values
            if isinstance(g_loss, list):
                g_loss = np.mean(g_loss)

            # Print training progress for the current batch
            print(f'Epoch>{epoch+1}, Batch>{batch+1}/{bat_per_epo}, '
                  f'd_real={d_loss_real:.3f}, d_fake={d_loss_fake:.3f}, g_loss={g_loss:.3f}')

    # Save the trained generator model
    g_model.save('conditional_generator11.h5')
    print("Generator model saved as 'conditional_generator11.h5'")


# %%
#Train the GAN

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)

# train model
train(g_model, d_model, gan_model, X_train,y_train, latent_dim, n_epochs=100, n_batch=256)

# %% [markdown]
# # Load model

# %%
# load model
from tensorflow.keras.models import load_model

model = load_model('conditional_generator11.h5')


# %% [markdown]
# # Generate Images

# %%
def generate_latent_points_new(latent_dim, n_samples):
    """
    Generate points in the latent space and corresponding random labels.

    Args:
    - latent_dim: Size of the latent space.
    - n_samples: Number of samples to generate.

    Returns:
    - Latent points and corresponding random labels.
    """
    # Generate random points in the latent space
    latent_points = np.random.randn(latent_dim * n_samples)
    latent_points = latent_points.reshape(n_samples, latent_dim)

    # Generate random labels (3 classes: 0, 1, 2)
    labels = np.random.randint(0, 3, n_samples)

    return latent_points, labels

def show_plot(examples, n):
    """
    Plot generated images in an n x n grid.

    Args:
    - examples: Generated images.
    - n: Number of rows and columns for the grid.
    """
    plt.figure(figsize=(10,10))
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :], cmap='gray')
    plt.show()

# Generate latent points and labels
latent_points, labels = generate_latent_points_new(100, 100)

# Specify labels: generate 100 sets of labels, each going from 0 to 2 (3 classes)
# This line is redundant since we already generate random labels in `generate_latent_points`
# labels = np.asarray([x for _ in range(10) for x in range(10)])

# Generate images using the generator model
X = g_model.predict([latent_points, labels]) # or if load model put:  X = model.predict([latent_points, labels]) 


# Scale images from [0, 1] to [0, 255]
X = (X * 255).astype(np.uint8)

# Plot the result (10x10 grid, all images in a column should belong to the same class)
show_plot(X, 10)


# %%

# %%

# %%

# %%

# %%
