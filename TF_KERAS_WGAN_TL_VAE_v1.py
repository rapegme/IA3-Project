# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:57:35 2021

@author: Rafa
"""

from IPython import display

import glob

import imageio
import PIL

import os
import time
import pathlib
import tqdm
import functools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from sklearn.model_selection  import train_test_split
from keras.callbacks import ModelCheckpoint

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import keras.backend as K

#Training input
EPOCHS = 500
BATCH_SIZE = 128
noise_dim = 128

# Console message of Tensorflow
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

train_size = len(train_list)
test_size =  len(test_list)


train_dt = tf.data.Dataset.from_tensor_slices(train_list)

test_dt = tf.data.Dataset.from_tensor_slices(test_list)

def preprocess(fn):
    img = tf.io.read_file(fn)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.cast(img, dtype=tf.float32)
    img = (img - 127.5) / 127.5
    img = tf.image.resize(img, (64, 64))
    return img

train_dt = train_dt.map(preprocess, num_parallel_calls=-1).cache()

train_dt = train_dt.shuffle(train_size).batch(BATCH_SIZE, drop_remainder=True)

train_dt.prefetch(32)

#Create an ImageDataGenerator to separate the imagenes from the folder
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1./127.5)-1, validation_split=0.2)

class WeightClipping(keras.constraints.Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value
    def __call__(self, weights):
        return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

class ConvLayer(Layer):
    def __init__(self, nf, ks, strides=2, padding='same', constraint=None, **kwargs):
        super().__init__(**kwargs)
        
        self.conv = Conv2D(nf, ks, strides=strides, padding=padding,
                           kernel_initializer='he_normal', kernel_constraint=constraint, use_bias=False)

        self.norm = LayerNormalization()
        
        self.act = LeakyReLU(0.2)
        
    def call(self, X):
        X = self.act(self.conv(X))
        return self.norm(X)
    
def conv_layer(nf, ks, strides=2, padding='same'):
    
    conv = Conv2D(nf, ks, strides=strides, padding=padding, use_bias=False)
    bn = BatchNormalization()
    act = LeakyReLU(0.2)
    return keras.Sequential([conv, act, bn])

#Create a function to create a model for the Discriminator GAN
def critic(input_shape=(64, 64, 3), dim=64, n_downsamplings=4):
    h = inputs = keras.Input(shape=input_shape)
    # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
    h = ConvLayer(dim, 4, strides=2, padding='same')(h)
    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), dim * 8)
        h = ConvLayer(d, 4, strides=2, padding='same')(h)

    h = keras.layers.Conv2D(1, 4, strides=1, padding='valid', kernel_initializer='he_normal')(h)
    h = Flatten()(h)
    return keras.Model(inputs=inputs, outputs=h, name="Discriminator")

d = critic()
d.summary()

class UpsampleBlock(Layer):
    def __init__(self, nf, ks, strides=2, padding='same', constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.conv_transpose = Conv2DTranspose(nf, ks, strides=strides, padding=padding, 
                                              kernel_initializer='he_normal', kernel_constraint=constraint)
        
        self.norm = LayerNormalization()
        self.act = ReLU()
        
    def call(self, X):
        X = self.act(self.conv_transpose(X))
        return self.norm(X)
    
def deconv_layer( nf, ks, strides=2, padding='same'):
    conv_transpose = Conv2DTranspose(nf, ks, strides=strides, padding=padding)
    bn = BatchNormalization()
    act = ReLU()
    return keras.Sequential([conv_transpose, act, bn])

def generator(input_shape=(128,), output_channels=3, dim=64, n_upsamplings=4):
    
    x = inputs = keras.Input(shape=input_shape)
    x = layers.Dense(8 * 8 * 128,  activation=layers.LeakyReLU())(x)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation=layers.LeakyReLU(), strides=1, padding="same")(x)
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation=layers.LeakyReLU(), strides=1, padding="same")(x)
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Conv2DTranspose(32, 3, activation=layers.LeakyReLU(), strides=1, padding="same")(x)
    h = layers.Conv2DTranspose(3, 3, activation="tanh", padding="same")(x)

    return keras.Model(inputs=inputs, outputs=h, name="Generator")

g = generator()

g.load_weights("./Pesos/pesos_decoder")
g.summary()

def d_loss(real, fake):
    real_loss = -tf.reduce_mean(real)
    fake_loss = tf.reduce_mean(fake)
    return real_loss + fake_loss

def g_loss(fake):
    return - tf.reduce_mean(fake)

@tf.function
def gradient_penalty(model, real, fake):
    shape = [tf.shape(real)[0]] + [1, 1, 1]
    alpha = tf.random.uniform(shape=shape, minval=0, maxval=1)
    interpolated = alpha * real  + (1-alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = model(interpolated)
    grad = tape.gradient(pred, interpolated)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((norm - 1.)**2)
    return gp

optD = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.99)
optG = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.99)

@tf.function
def trainD(real):
    noise = tf.random.normal((BATCH_SIZE, noise_dim))
    with tf.GradientTape() as tape:
        fake_images = g(noise, training=True)
        real_output = d(real, training=True)
        fake_output = d(fake_images, training=True)
        gp_loss = gradient_penalty(functools.partial(d, training=True), real, fake_images)
        loss = d_loss(real_output, fake_output)        
        disc_loss = loss + 10 * gp_loss
        
    d_grad = tape.gradient(disc_loss, d.trainable_variables)
    optD.apply_gradients(zip(d_grad, d.trainable_variables))
    
    
@tf.function
def trainG():
    noise = tf.random.normal((BATCH_SIZE, noise_dim))
    with tf.GradientTape() as tape:
        generated_images = g(noise, training=True)
        loss = g_loss(d(generated_images))
        
    g_grad = tape.gradient(loss, g.trainable_variables)
    optG.apply_gradients(zip(g_grad, g.trainable_variables))
    
def generate_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig, ax = plt.subplots(4, 4, figsize=(10,10))

    #Cliping values to 0 to 1 for ploting
    predictions = (predictions * 0.5) + 0.5
    
    for i, a in enumerate(ax.flat):
        a.imshow(predictions[i, :])
        a.axis('off')
    plt.show()


def save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig, ax = plt.subplots(4, 4, figsize=(10,10))
    
    #Cliping values to 0 to 1 for ploting
    predictions = (predictions * 0.5) + 0.5
    
    for i, a in enumerate(ax.flat):
        a.imshow(predictions[i, :])
        a.axis('off')
        
    plt.savefig('./WGAN_TF/Img/image_at_epoch_{:04d}.png'.format(epoch))

def save_weights (discriminator, generator):
    
    discriminator.save_weights("./WGAN_TF/D/pesos_D")
    generator.save_weights("./WGAN_TF/G/pesos_G")
    

seed = tf.random.normal((16, 128))
    
def train(dataset, epochs):
    Count_D = 0
    Count_G = 0
    
    for epoch in range(epochs):
        start = time.time()

        for image_batch in tqdm.tqdm(dataset, total=train_size//BATCH_SIZE):
          
            trainD(image_batch)
            Count_D = Count_D + 1
            if optD.iterations.numpy() % 5 == 0:
                trainG()
                Count_G = Count_G + 1
                
        display.clear_output(wait=True)
        if (epoch % 25) == 0:
            generate_images(g, epoch + 1, seed)
        
        if (epoch % 1) == 0:
            save_images(g, epoch + 1, seed)
        
        if (epoch % 10) == 0:
            save_weights(d,g)
            
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
    print("Veces entrenada D: {} \nVeces entrenada G: {} ".format(Count_D, Count_G) )
    
# train(train_dt, 10)