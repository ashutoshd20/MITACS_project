from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed = 1
import random
random.seed(1)
import numpy as np
np.random.seed(1)
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical
import sys

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, model, list_IDs, batch_size, dim, shuffle,filename, column_PD, column_SC):
        'Initialization'
        self.dim = dim
        self.model = model
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.filename = filename
        self.column_PD = column_PD
        self.column_SC = column_SC
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
        X, y_PD, y_SC = self.__data_generation(list_IDs_temp)
        return X, y_PD, y_SC

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, 10,12,10, 1))
        y_PD = np.empty((self.batch_size), dtype=int)  # sites
        y_SC = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #load based on the ID from csv filter by ID
            dataset = pd.read_csv(self.filename)
            dataset = dataset[dataset['Subject']==ID]
            path = dataset['Path'].values
            itk_img = sitk.ReadImage(path)
            np_img = sitk.GetArrayFromImage(itk_img)
            v = []
            v.append(np.float32(np_img.reshape(160,192,160, 1)))
            X[i,] = np.float32((np.array(self.model.predict(np.array(v))))[0])
            y_PD[i,] = dataset[self.column_PD].values
            y_SC[i,] = dataset[self.column_SC].values 
        return X, y_PD, y_SC # This line will take care of outputing the inputs for training and the labels
