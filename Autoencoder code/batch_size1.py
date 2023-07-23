import tensorflow as tf
from tensorflow.keras import layers, models
import random
import csv
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D, LeakyReLU, Add
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import argparse
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import pandas as pd
import sys

#****************************Datagenrator***********************************************
from numpy.random import seed

import tensorflow as tf
import random
import numpy as np
np.random.seed(1)
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical
import sys

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size, dim, shuffle,filename, column):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.filename = filename
        self.column = column
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size), dtype=int)  # sites

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #load based on the ID from csv filter by ID
            dataset = pd.read_csv(self.filename)
            dataset = dataset[dataset['Subject']==ID]
            path = dataset['Path'].values
            itk_img = sitk.ReadImage(path)
            np_img = sitk.GetArrayFromImage(itk_img)
            X[i,] = np.float32(np_img.reshape(self.dim[0], self.dim[1], self.dim[2], 1))
            y[i,] = dataset[self.column].values 
        

        return X, X



#******************creating the model*****************************************************
def create_encoder(input_shape):
    input_img=tf.keras.Input(shape=input_shape)
    
    x = layers.Conv3D(4, (5, 5, 5), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling3D((2, 2, 2), padding='same')(x)
    x = layers.Conv3D(16, (5, 5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2), padding='same')(x)
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2), padding='same')(x)
    x = layers.Conv3D(1, (3, 3, 3), activation='relu', padding='same')(x) #channel dimension
    encoded = layers.MaxPooling3D((2, 2, 2), padding='same')(x)
    return models.Model(input_img, encoded, name='encoder')

# Decoder
def create_decoder():
    encoded_input = tf.keras.Input(shape=(None, None, None, 1)) #update the channel dim here
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(encoded_input)
    x = layers.UpSampling3D((2, 2, 2))(x)
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D((2, 2, 2))(x)
    x = layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D((2, 2, 2))(x)
    x = layers.Conv3D(4, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D((2, 2, 2))(x)
    decoded = layers.Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
    return models.Model(encoded_input, decoded, name='decoder')

# Autoencoder
def create_autoencoder(input_shape=(160,192,160, 1)):
    encoder = create_encoder(input_shape)
    decoder = create_decoder()

    inputs = tf.keras.Input(shape=input_shape)
    encoded = encoder(inputs)
    decoded = decoder(encoded)

    return models.Model(inputs, decoded, name='autoencoder')


encoder = create_encoder(input_shape=(160, 192, 160, 1))
decoder = create_decoder()
autoencoder = models.Model(encoder.input, decoder(encoder.output), name='autoencoder')

autoencoder.compile(optimizer='adam', loss='cosine_similarity')
autoencoder.summary()


#*********************Taking the arguments********************************#
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-cycles', type=int, help='number of cycles')
parser.add_argument('-epochs', type=int, help='number of local epochs per cycle')
parser.add_argument('-batch_size', type=int, help='batch size')
args = parser.parse_args()

params = {'batch_size': args.batch_size,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }
CYCLES = args.cycles
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

fn_train_PD = args.fn_train
train_PD = pd.read_csv(fn_train_PD)
studies =  train_PD['Study'].unique()
np.random.seed(42)  
np.random.shuffle(studies)

#********************************function for training the model********
optimizer_encoder = Adam(learning_rate=0.001)
@tf.function
def train_step_PD(step, X, y_PD, batch_size):
    with tf.GradientTape() as tape:
        X=tf.cast(X, tf.float64)
        logits_PD = tf.cast(autoencoder(X, training=True), tf.float64)
        mse = tf.keras.losses.CosineSimilarity()
        train_loss_PD=mse(X, logits_PD)
     

    # compute gradient 
    grads = tape.gradient(train_loss_PD, autoencoder.trainable_weights)

    # update weights
    optimizer_encoder.apply_gradients(zip(grads, autoencoder.trainable_weights))


    return train_loss_PD, logits_PD

#************************TRaining the model*********************************#

for c in range(CYCLES):
    np.random.seed(42+c)  
    np.random.shuffle(studies)
    print("CYCLE --> "+str(c)+'\n')

    ########################
    total_loss=[]
    import math
    for s in studies:
        batch_size = BATCH_SIZE
        print("STUDY --> "+str(s))

        train_aux =  train_PD[train_PD['Study']==s]
        IDs_list = train_aux['Subject'].to_numpy()
        train_IDs = IDs_list
        if(len(train_IDs)<batch_size): 
            batch_size=len(train_IDs)
        
        for epoch in range(EPOCHS):
            training_generator = DataGenerator(train_IDs, batch_size, (params['imagex'], params['imagey'], params['imagez']), True, fn_train_PD, 'Group_bin')

            for batch in range(math.ceil(len(train_IDs)/batch_size)):
                step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
                X, y_PD = training_generator.__getitem__(step_batch)

                #********alternate of model fit******************
                train_loss_PD, logits_PD= train_step_PD(step_batch, X, y_PD, batch_size)

                print('\nBatch '+str(batch+1)+'/'+str(math.ceil(len(train_IDs)/batch_size)))
                print("LOSS PD -->", train_loss_PD)
 
    print("LOSS OF CYCLE-->",np.average(np.array(total_loss)))
    
autoencoder.save('Ashu_autoencoder.h5')
encoder.save('Ashu_encoder.h5')


  
                #*****************************************************