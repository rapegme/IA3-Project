# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:20:28 2021

@author: Rafa
"""

from IPython import display

import glob

import imageio
import PIL

import os
import time
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection  import train_test_split
from keras.callbacks import ModelCheckpoint

#Console message of Tensorflow
# =============================================================================
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# =============================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#Image path
path_anime = r"C:\Users\Rafa\Documents\Pruebas-IA\fotos dibujos\data"
dir_anime = pathlib.Path(path_anime)

list_img_anime = [str(r"C:\Users\Rafa\Documents\Pruebas-IA\fotos dibujos\data\Imagenes" + "\\" + fn) for fn in os.listdir(r"C:\Users\Rafa\Documents\Pruebas-IA\fotos dibujos\data\Imagenes")]

print(list_img_anime[-1])

train_list, test_list = train_test_split(list_img_anime, test_size=0.2, random_state=1)

#Training input
  
epochs = 50

num_examples_to_generate = 16
batch_size = 64
lr = 0.001

latent_dim = 2

train_size = len(train_list)
test_size =  len(test_list)


# =============================================================================
# train_dt = tf.data.Dataset.from_tensor_slices(train_list)
# 
# test_dt = tf.data.Dataset.from_tensor_slices(test_list)
# =============================================================================

#Create an ImageDataGenerator to separate the imagenes from the folder
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Divide the dataset for training/test using Flow_From_Directory
train_dt = datagen.flow_from_directory(dir_anime,
                                             target_size=(64,64),
                                             batch_size=batch_size,
                                             seed = 1124,
                                             color_mode ="rgb",
                                             subset="training")

test_dt = datagen.flow_from_directory(dir_anime,
                                             target_size=(64,64),
                                             batch_size=batch_size,
                                             seed = 1124,
                                             color_mode ="rgb",
                                             subset="validation"
    )


sample_dt = tf.keras.preprocessing.image_dataset_from_directory(dir_anime,
                                                              label_mode=None,
                                                              validation_split=0.2,
                                                              batch_size=batch_size,
                                                              image_size=(64,64),
                                                              color_mode="rgb",
                                                              seed=1124,
                                                              subset="validation")

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in sample_dt.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]


"""
## Create a sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
"""
## Build the encoder
"""



encoder_inputs = keras.Input(shape=(64, 64, 3))
x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(), strides=1, padding="same")(encoder_inputs) #Out -> (64,64,32)
x = layers.MaxPool2D(pool_size=2,padding="same")(x) #Out -> (32,32,32)
x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(), strides=1, padding="same")(x) #Out -> (32,32,64)
x = layers.MaxPool2D(pool_size=2,padding="same")(x) #Out -> (16,16,64)
x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(), strides=1, padding="same")(x) #Out -> (16,16,128)
x = layers.MaxPool2D(pool_size=2,padding="same")(x) #Out -> (8,8,128)
x = layers.Flatten()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512,  activation=layers.LeakyReLU())(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


"""
## Build the decoder
"""
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 128,  activation=layers.LeakyReLU())(latent_inputs)
x = layers.Reshape((8, 8, 128))(x)
x = layers.UpSampling2D(size=2)(x)
x = layers.Conv2DTranspose(128, 3, activation=layers.LeakyReLU(), strides=1, padding="same")(x)
x = layers.UpSampling2D(size=2)(x)
x = layers.Conv2DTranspose(64, 3, activation=layers.LeakyReLU(), strides=1, padding="same")(x)
x = layers.UpSampling2D(size=2)(x)
x = layers.Conv2DTranspose(32, 3, activation=layers.LeakyReLU(), strides=1, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


"""
## Build the VAE
"""
class VAE(keras.Model):
    def __init__(self, encoder, decoder, latent_dim, sample_dt=None, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.sample_dt = sample_dt
        self.random_sample = tf.random.normal(shape=(1, self.latent_dim))
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            if isinstance(data, tuple):
                data=data[0]
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

    def test_step(self, data):
      if isinstance(data, tuple):
        data = data[0]

      z_mean, z_log_var, z = self.encoder(data)
      reconstruction = self.decoder(z)
      reconstruction_loss = tf.reduce_mean(
          keras.losses.binary_crossentropy(data, reconstruction)
      )
      reconstruction_loss *= 64 * 64 * 3
      kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
      kl_loss = tf.reduce_mean(kl_loss)
      kl_loss *= -0.5
      total_loss = reconstruction_loss + kl_loss
      return {
          "loss": total_loss,
          "reconstruction_loss": reconstruction_loss,
          "kl_loss": kl_loss,
      }

    def call(self, inputs):
      z_mean, z_log_var, z = encoder(inputs)
      reconstruction = decoder(z)
      reconstruction_loss = tf.reduce_mean(
          keras.losses.binary_crossentropy(inputs, reconstruction)
      )
      # reconstruction_loss *= 64* 64 * 3
      reconstruction_loss *= 64* 64
      kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
      kl_loss = tf.reduce_mean(kl_loss)
      kl_loss *= -0.5
      total_loss = reconstruction_loss + kl_loss
      self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
      self.add_metric(total_loss, name='total_loss', aggregation='mean')
      self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
      return reconstruction

  
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
  
    def encode(self, x):
        # mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        mean, logvar, _ = self.encoder(x)
        return mean, logvar
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
          probs = tf.sigmoid(logits)
          return probs
        return logits
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    #Hay que revisarlo para que plotee mejor
    # def generate_and_save_images(self, epoch):
    #     if self.sample_dt != None:
    #       test_sample = self.sample_dt 
    #       mean, logvar = self.encode(test_sample)
    #       z = self.reparameterize(mean, logvar)
    #       predictions = self.sample(z)
    #       fig = plt.figure(figsize=(64, 64))
        
    #       for i in range(predictions.shape[0]):
    #         plt.subplot(16, 16, i + 1)
    #         plt.imshow(predictions[i, :, :, :], cmap="gray")
    #         plt.axis('off')
        
    #       # tight_layout minimizes the overlap between 2 sub-plots
    #       plt.savefig('./Img_Epoch/image_at_epoch_{:04d}.png'.format(epoch))
    #       plt.show()
    #     else:
    #         predictions = self.sample()
    #         fig = plt.figure(figsize=(64, 64))
        
    #         for i in range(predictions.shape[0]):
    #             plt.subplot(16, 16, i + 1)
    #             plt.imshow(predictions[i, :, :, :], cmap="gray")
    #             plt.axis('off')
        
    #         # tight_layout minimizes the overlap between 2 sub-plots
    #         plt.savefig('./Img_Epoch/image_at_epoch_{:04d}.png'.format(epoch))
    #         plt.show()
          
    def generate_and_save_images_latent_dim(self, epoch):
        n=5
        figsize = 16
        digit_size = 64
        scale = 1.0
        figure = np.zeros((digit_size * n, digit_size * n, 3))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]
    
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_x = np.linspace(-xi, xi, int(self.latent_dim/2))
                z_y = np.linspace(-yi, yi, int(self.latent_dim/2))
                z = np.concatenate([z_x, z_y])
                z_sample = np.array([z])
                x_decoded = vae.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size,3)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit
    
        plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.imshow(figure)
        plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('./Img_Epoch/image_at_epoch_{:02d}.png'.format(epoch))
        plt.show()
    
    
# =============================================================================
# Create a callback to modify the learning rate during training
# =============================================================================
    
def Learning_Rate_Scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

#Create a callback to create a plot and save it.
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, numpy_logs):
        # print("Epoch " + str(epoch))
        self.model.generate_and_save_images_latent_dim(epoch)
        
custom_callback = CustomCallback()  

lr_callback = tf.keras.callbacks.LearningRateScheduler(Learning_Rate_Scheduler)

#Create a save of the best model each X epochs      
checkpoint =  ModelCheckpoint("./Best_Model/best_model.h5", monitor="loss", verbose=1, save_best_only=True, save_freq="epoch")

vae = VAE(encoder, decoder, latent_dim, test_sample)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=0.87, beta_2=0.97))


STEP_SIZE_TRAIN = train_dt.n//train_dt.batch_size
STEP_SIZE_VALID = test_dt.n//test_dt.batch_size

# To train ↓↓↓ use code ↓↓↓ or uncomment

# vae.fit(train_dt, epochs=epochs, validation_data=test_dt, callbacks=[lr_callback, checkpoint, custom_callback])

def plot_latent_space(vae, n=30, figsize=16):
    # display a n*n 2D manifold of digits
    digit_size = 64
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size,3)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    plt.show()


# plot_latent_space(vae)