from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from datagenerator_pd import DataGenerator
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import argparse
import sys
import math
import time

EPOCHS = 12
LEARNING_RATE = 0.001

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-fn_test', type=str, help='testing set')
parser.add_argument('-batch_size', type=int, help='batch size') # run for BS2
args = parser.parse_args()


params = {'batch_size': args.batch_size,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }

ashu_encoder = tf.keras.models.load_model('Ashu_encoder.h5') #using the enocder developed before


#****************This is a small 2 layer for the bottleneck*****************
''' The bottle nech output is of shape (10,12,10,1) which is flattened to a layer a 1200 layer is used to pass it
to a 768 unit layer which is the input shape for Raissa's PD classifier
'''
def sfcn(inputLayer):
    #block 1
    x = Flatten(name="flat1")(inputLayer[0])
    x= Dense(1200)(x)
    x= Dense(768)(x)

    return x



optimizer_encoder = Adam(learning_rate=0.001)
optimizer_PD = Adam(learning_rate=0.001)


# encoder it can be also said as the bottleneck for the image output

inputA = Input(shape=(10, 12,10, 1), name="InputA")
feature_dense_enc = sfcn([inputA])
encoder = Model(inputs=[inputA], outputs=[feature_dense_enc])


def add_PD_layer(input_shape):
    inputs_PD = Input(shape=(input_shape))
    output_PD = Dense(1, activation='sigmoid')(inputs_PD)
    return Model(inputs_PD, output_PD)


classifier_PD = add_PD_layer(encoder.output.shape[1])


# loss
def categorical_cross_entropy_label_predictor(y_true, y_pred):
    ccelp = tf.keras.losses.BinaryCrossentropy()
    return ccelp(y_true, y_pred)/args.batch_size


def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)

 
# Dataset generator for PD classification
fn_train_PD = args.fn_train
train_PD = pd.read_csv(fn_train_PD)
train_IDs_list_PD = train_PD['Subject'].to_numpy()
train_IDs_PD = train_IDs_list_PD

fn_val_PD = args.fn_test
val_PD = pd.read_csv(fn_val_PD)
val_IDs_list_PD = val_PD['Subject'].to_numpy()
val_IDs_PD = val_IDs_list_PD


# train step for PD classifier
@tf.function
def train_step_PD(step, X, y_PD):
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_PD = classifier_PD(logits_enc, training=True)
        train_loss_PD = categorical_cross_entropy_label_predictor(y_PD, logits_PD)

    # compute gradient 
    grads = tape.gradient(train_loss_PD, [encoder.trainable_weights, classifier_PD.trainable_weights])

    # update weights
    optimizer_encoder.apply_gradients(zip(grads[0], encoder.trainable_weights))
    optimizer_PD.apply_gradients(zip(grads[1], classifier_PD.trainable_weights))

    return train_loss_PD, logits_PD


# test step for PD classifier
@tf.function
def test_step_PD(step, X, y_PD):

    val_logits_enc = encoder(X, training=False) 
    val_logits_PD = classifier_PD(val_logits_enc, training=False)

    # Compute the loss value 
    val_loss_PD = categorical_cross_entropy_label_predictor(y_PD, val_logits_PD)
    
    return val_loss_PD, val_logits_PD


####################################################################################################################

# training PD classifier
for epoch in range(EPOCHS):
    # training
    training_generator_PD = DataGenerator(ashu_encoder, train_IDs_PD, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_train_PD, 'Group_bin')
    t1 = time.time()

    for batch in range(math.ceil(len(train_IDs_PD)/params['batch_size'])):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_PD = training_generator_PD.__getitem__(step_batch)
        for j in range(0,3):  ####this determines the number of epochs here
            train_loss_PD, logits_PD= train_step_PD(step_batch, X, y_PD)
            print('\nBatch '+str(batch+1)+'/'+str(math.ceil(len(train_IDs_PD)/params['batch_size'])))
            print("LOSS PD -->", train_loss_PD)
            for _ in range(args.batch_size):
                print("LOGITS PD -->", logits_PD[_])
                print("ACTUAL PD -->", y_PD[_])
            m = tf.keras.metrics.binary_accuracy(y_PD.astype(float), logits_PD, threshold=0.5)
            print("ACCURACY -->", m, tf.math.reduce_mean(m))

    t2 = time.time()
    template = 'TRAINING disease - ETA: {} - cycle: {}\n'
    print(template.format(round((t2-t1)/60, 4), epoch+1))


    # validation
    val_generator_PD = DataGenerator(ashu_encoder, val_IDs_PD, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_val_PD, 'Group_bin')
    t3 = time.time()

    for batch in range(math.ceil(len(val_IDs_PD)/params['batch_size'])):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_PD = val_generator_PD.__getitem__(step_batch)
        val_loss_PD, val_logits_PD = test_step_PD(step_batch, X, y_PD)
        print('\nBatch '+str(batch+1)+'/'+str(math.ceil(len(val_IDs_PD)/params['batch_size'])))
        print("VAL LOSS PD -->", val_loss_PD)
        for _ in range(args.batch_size):
            print("LOGITS PD -->", val_logits_PD[_])
            print("ACTUAL PD -->", y_PD[_])
        m = tf.keras.metrics.binary_accuracy(y_PD.astype(float), val_logits_PD, threshold=0.5)
        print("ACCURACY -->", m, tf.math.reduce_mean(m))

    t4 = time.time()
    template = 'VALIDATION disease - ETA: {} - cycle: {}\n'
    print(template.format(round((t4-t3)/60, 4), epoch+1))

    # learning rate scheduling
    LEARNING_RATE = scheduler(epoch, LEARNING_RATE)
    optimizer_encoder = Adam(learning_rate=LEARNING_RATE)
    optimizer_PD = Adam(learning_rate=LEARNING_RATE)

####################################################################################################################

encoder.save('mini_encoder_BS'+str(args.batch_size)+'_ep3.h5')
classifier_PD.save('ASHU_PD_classifier_BS'+str(args.batch_size)+'_ep3.h5')

####################################################################################################################