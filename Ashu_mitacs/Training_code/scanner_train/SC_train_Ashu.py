from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from numpy import argmax
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D, LeakyReLU, Add
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import confusion_matrix
from datagenerator_pd import DataGenerator
from tensorflow.keras.utils import to_categorical
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import argparse
import sys
import math
import time

EPOCHS = 20
LEARNING_RATE = 0.001

autoencoder_Ashu = tf.keras.models.load_model('Ashu_encoder.h5')

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-en', type=str, help='encoder model args path')
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-fn_test', type=str, help='testing set')
parser.add_argument('-batch_size', type=int, help='batch size')
args = parser.parse_args()


params = {'batch_size': args.batch_size,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }


encoder = tf.keras.models.load_model(args.en)
encoder.trainable = False

optimizer_SC = Adam(learning_rate=0.001, decay=0.003)


def add_SC_layer(input_shape):
    inputs_PD = Input(shape=(input_shape))
    output_PD = Dense(23, activation='softmax')(inputs_PD)
    return Model(inputs_PD, output_PD)


classifier_SC = add_SC_layer(encoder.output.shape[1])


# loss
def categorical_cross_entropy_domain_classifier(y_true, y_pred):
    ccedc = tf.keras.losses.CategoricalCrossentropy()
    return ccedc(y_true, y_pred)/args.batch_size


# Dataset generator for SC classification
fn_train_SC = args.fn_train
train_SC = pd.read_csv(fn_train_SC)
train_IDs_list_SC = train_SC['Subject'].to_numpy()
train_IDs_SC = train_IDs_list_SC

fn_val_SC = args.fn_test
val_SC = pd.read_csv(fn_val_SC)
val_IDs_list_SC = val_SC['Subject'].to_numpy()
val_IDs_SC = val_IDs_list_SC


# train step for PD classifier
@tf.function
def train_step_SC(step, X, y_SC):
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_SC = classifier_SC(logits_enc, training=True)
        train_loss_SC = categorical_cross_entropy_domain_classifier(tf.one_hot(y_SC, 23), logits_SC)

    # compute gradient 
    grads = tape.gradient(train_loss_SC, classifier_SC.trainable_weights)

    # update weights
    optimizer_SC.apply_gradients(zip(grads, classifier_SC.trainable_weights))

    return train_loss_SC, logits_SC


# test step for PD classifier
@tf.function
def test_step_SC(step, X, y_SC):

    val_logits_enc = encoder(X, training=False) 
    val_logits_SC = classifier_SC(val_logits_enc, training=False)

    # Compute the loss value 
    val_loss_SC = categorical_cross_entropy_domain_classifier(tf.one_hot(y_SC, 23), val_logits_SC)
    
    return val_loss_SC, val_logits_SC


####################################################################################################################

# training PD classifier
for epoch in range(EPOCHS):
    # training
    training_generator_SC = DataGenerator(autoencoder_Ashu, train_IDs_SC, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_train_SC, 'Scanner')
    t1 = time.time()

    for batch in range(math.ceil(len(train_IDs_SC)/params['batch_size'])):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_SC = training_generator_SC.__getitem__(step_batch)
        train_loss_SC, logits_SC= train_step_SC(step_batch, X, y_SC)
        print('Batch '+str(batch+1)+'/'+str(math.ceil(len(train_IDs_SC)/params['batch_size'])))
        print("LOSS SC -->", train_loss_SC)
        for _ in range(args.batch_size):
            print("LOGITS SC -->", logits_SC[_])
            print("ACTUAL SC -->", y_SC[_])
            m = tf.keras.metrics.categorical_accuracy(tf.one_hot(y_SC, 23), logits_SC)
            print("ACCURACY -->", m, tf.math.reduce_mean(m))

    t2 = time.time()
    template = 'TRAINING - ETA: {} - epoch: {}\n'
    print(template.format(round((t2-t1)/60, 4), epoch+1))


    # validation
    val_generator_SC = DataGenerator(autoencoder_Ashu, val_IDs_SC, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_val_SC, 'Scanner')
    t3 = time.time()

    for batch in range(math.ceil(len(val_IDs_SC)/params['batch_size'])):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_SC = val_generator_SC.__getitem__(step_batch)
        val_loss_SC, val_logits_SC = test_step_SC(step_batch, X, y_SC)
        print('Batch '+str(batch+1)+'/'+str(math.ceil(len(val_IDs_SC)/params['batch_size'])))
        print("VAL LOSS SC -->", val_loss_SC)
        for _ in range(args.batch_size):
            #print("LOGITS SC -->", val_logits_SC[_])
            print("ACTUAL SC -->", y_SC[_])
            m = tf.keras.metrics.categorical_accuracy(tf.one_hot(y_SC, 23), val_logits_SC)
            print("ACCURACY -->", m, tf.math.reduce_mean(m))

    t4 = time.time()
    template = 'VALIDATION - ETA: {} - epoch: {}\n'
    print(template.format(round((t4-t3)/60, 4), epoch+1))
    

####################################################################################################################

classifier_SC.save('SC_classifier_BS'+str(args.batch_size)+'.h5')

####################################################################################################################