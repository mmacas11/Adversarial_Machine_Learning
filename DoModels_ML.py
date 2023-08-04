#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from SaConvLSTM import *
import operator
from DoSplitData import load_split_data
from DoVaeSupplementary import *
import keras


def ConvLSTMVae_Attention (stf,latent_dim,nft,step_size,channel,ini):
    inp_layer = tf.keras.layers.Input(shape=(step_size,nft,nft,channel))
    #Spatial encoder    
    #conv 1
    conv1 = (tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3),strides=2,padding='same', activation='relu', kernel_initializer=ini)))(inp_layer)   
    conv1 = tf.keras.layers.Dropout(0.3)(conv1) 
    #conv 2
    conv2 = (tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3),strides=2,padding='same',activation='relu',kernel_initializer=ini))) (conv1)
    conv2 = tf.keras.layers.Dropout(0.3)(conv2) 
    #conv 3
    conv3 = (tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3),strides=2,padding='same',activation='relu',kernel_initializer=ini))) (conv2)            
    conv3 = tf.keras.layers.Dropout(0.3)(conv3) 
    #1
    x = (SaConvLSTM2D(filters=128, kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=ini,activation='tanh',
                          dropout=0.3,return_sequences=True,stateful = stf)) (conv3)    
    #2
    x = (SaConvLSTM2D(filters=64, kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=ini,activation='tanh',
                          dropout=0.3,return_sequences=True,stateful = stf)) (x)    
    
    x = tf.keras.layers.Flatten()(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(inp_layer, [z_mean, z_log_var, z], name="encoder") 
    
    #decoder
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128, activation="tanh")(latent_inputs)
    x = tf.keras.layers.Dense(1*int((nft/2)/2)*int((nft/2)/2) * 128, activation="tanh")(x)
    x = tf.keras.layers.Reshape((1,int((nft/2)/2), int((nft/2)/2), 128))(x)
    
    #3
    x = (SaConvLSTM2D(filters=64, kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=ini,activation='tanh',
                          dropout=0.3,return_sequences=True,stateful = stf)) (x)            
    #Spatial decoder
    dec1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(128, 3, strides=1,padding='same',activation='relu',
                        kernel_initializer=ini)) (x)
    dec1 = tf.keras.layers.Dropout(0.3)(dec1) 
    
    dec2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(64, 3, strides=2,padding='same',activation='relu',
                        kernel_initializer=ini)) (dec1)
    dec2 = tf.keras.layers.Dropout(0.3)(dec2) 
    
    dec3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(32, 3, strides=2,padding='same',activation='relu',
                        kernel_initializer=ini)) (dec2)
    dec3 = tf.keras.layers.Dropout(0.3)(dec3) 
    #3
    dec4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(1, 3, strides=1,padding='same',activation='relu', 
                        kernel_initializer=ini)) (dec3)
    dec4 = tf.keras.layers.Dropout(0.3)(dec4) 
    decoder = tf.keras.Model(latent_inputs, dec4, name="decoder")
    mvae = VAE_TD(encoder, decoder)    
    return (mvae)


