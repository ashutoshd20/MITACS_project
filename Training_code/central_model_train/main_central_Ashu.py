from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
import tensorflow as tf
import csv
import numpy as np
import SimpleITK as sitk
from numpy import argmax
import pandas as pd
from tensorflow.keras.optimizers.legacy import Adam
from datagenerator import DataGenerator
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import argparse
import sys
import math
import time

EPOCHS = 30
LEARNING_RATE = 0.001

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-fn_test', type=str, help='testing set')
parser.add_argument('-batch_size', type=int, help='batch size')
parser.add_argument('-alpha', type=float, help='alpha')
parser.add_argument('-beta', type=float, help='beta')
parser.add_argument('-en', type=str, help='pretrained encoder')
parser.add_argument('-pd', type=str, help='pretrained PD classifier')
parser.add_argument('-sc', type=str, help='pretrained SC classifier')
args = parser.parse_args()


params = {'batch_size': args.batch_size,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }


# pretrained models
encoder = tf.keras.models.load_model(args.en)
classifier_PD = tf.keras.models.load_model(args.pd)
classifier_SC = tf.keras.models.load_model(args.sc)
ashu_encoder = tf.keras.models.load_model('Ashu_encoder.h5')

encoder.trainable = True
classifier_PD.trainable = True
classifier_SC.trainable = True


# optimizers
optimizer_encoder = Adam(learning_rate=0.001)
optimizer_PD = Adam(learning_rate=0.001)
optimizer_SC = Adam(learning_rate=0.001, decay=0.003)


# loss
def categorical_cross_entropy_label_predictor(y_true, y_pred):
    ccelp = tf.keras.losses.BinaryCrossentropy()
    return ccelp(y_true, y_pred)

def categorical_cross_entropy_domain_classifier(y_true, y_pred):
    ccedc = tf.keras.losses.CategoricalCrossentropy()
    return ccedc(y_true, y_pred)

def confusionLoss(logits_SC, batch_size):
    log_logits = tf.math.log(logits_SC)
    sum_log_logits = tf.math.reduce_sum(log_logits)
    # sum_log_logits = sum_log_logits.numpy()
    return -1*sum_log_logits / (batch_size * 23)


# scheduler
def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)

 
# Dataset generator for PD classification
fn_train = args.fn_train
train = pd.read_csv(fn_train)
train_IDs_list = train['Subject'].to_numpy()
train_IDs = train_IDs_list

fn_val = args.fn_test
val = pd.read_csv(fn_val)
val_IDs_list = val['Subject'].to_numpy()
val_IDs = val_IDs_list


# train step for PD classifier
@tf.function
def train_step(step, X, y_PD, y_SC):
    ###################################################
    # FIRST STEP
    classifier_SC.trainable = False
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_PD = classifier_PD(logits_enc, training=True)
        train_loss_PD = categorical_cross_entropy_label_predictor(y_PD, logits_PD)

    # compute gradient 
    grads = tape.gradient(train_loss_PD, [encoder.trainable_weights, classifier_PD.trainable_weights])

    # update weights
    optimizer_encoder.apply_gradients(zip(grads[0], encoder.trainable_weights))
    optimizer_PD.apply_gradients(zip(grads[1], classifier_PD.trainable_weights))
    classifier_SC.trainable = True
    ###################################################
    # SECOND STEP
    encoder.trainable = False
    classifier_PD.trainable = False
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_SC = classifier_SC(logits_enc, training=True)
        train_loss_SC = args.alpha * categorical_cross_entropy_domain_classifier(tf.one_hot(y_SC, 23), logits_SC)

    # compute gradient 
    grads = tape.gradient(train_loss_SC, classifier_SC.trainable_weights)

    # update weights
    optimizer_SC.apply_gradients(zip(grads, classifier_SC.trainable_weights))
    encoder.trainable = True
    classifier_PD.trainable = True
    ###################################################
    # THIRD STEP
    classifier_PD.trainable = False
    classifier_SC.trainable = False
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_SC = classifier_SC(logits_enc, training=True)
        confusion_loss = args.beta * confusionLoss(logits_SC, args.batch_size)

    # compute gradient 
    grads = tape.gradient(confusion_loss, encoder.trainable_weights)

    # update weights
    optimizer_encoder.apply_gradients(zip(grads, encoder.trainable_weights))
    classifier_PD.trainable = True
    classifier_SC.trainable = True
    ###################################################

    logits_enc = encoder(X, training=True)
    logits_PD = classifier_PD(logits_enc, training=True)
    logits_SC = classifier_SC(logits_enc, training=True)

    train_loss = categorical_cross_entropy_label_predictor(y_PD, logits_PD) + args.alpha * categorical_cross_entropy_domain_classifier(tf.one_hot(y_SC, 23), logits_SC) \
                + args.beta * confusionLoss(logits_SC, args.batch_size)

    return train_loss, logits_PD, logits_SC


# test step for PD classifier
@tf.function
def test_step(step, X, y_PD, y_SC):

    val_logits_enc = encoder(X, training=False) 
    val_logits_PD = classifier_PD(val_logits_enc, training=False)
    val_logits_SC = classifier_SC(val_logits_enc, training=False)

    # Compute the loss value 
    val_loss = categorical_cross_entropy_label_predictor(y_PD, val_logits_PD) + args.alpha * categorical_cross_entropy_domain_classifier(tf.one_hot(y_SC, 23), val_logits_SC) \
                + args.beta * confusionLoss(val_logits_SC, args.batch_size)
    
    return val_loss, val_logits_PD, val_logits_SC


####################################################################################################################

# training PD classifier
for epoch in range(EPOCHS):
    # training
    training_generator = DataGenerator(ashu_encoder, train_IDs, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_train, 'Group_bin', 'Scanner')
    t1 = time.time()

    for batch in range(math.ceil(len(train_IDs)/params['batch_size'])):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_PD, y_SC = training_generator.__getitem__(step_batch)
        train_loss, logits_PD, logits_SC = train_step(step_batch, X, y_PD, y_SC)
        print('\nBatch '+str(batch+1)+'/'+str(math.ceil(len(train_IDs)/params['batch_size'])))
        print("LOSS PD -->", train_loss)
        for _ in range(args.batch_size):
            print("LOGITS PD -->", logits_PD[_])
            print("ACTUAL PD -->", y_PD[_])
            # print("LOGITS SC -->", logits_SC[_])
            # print("ACTUAL SC -->", y_SC[_])
        m_PD = tf.keras.metrics.binary_accuracy(y_PD.astype(float), logits_PD, threshold=0.5)
        m_SC = tf.keras.metrics.categorical_accuracy(tf.one_hot(y_SC, 23), logits_SC)
        print("ACCURACY PD -->", m_PD, tf.math.reduce_mean(m_PD))
        print("ACCURACY SC -->", m_SC, tf.math.reduce_mean(m_SC))

    t2 = time.time()
    template = 'TRAINING - ETA: {} - epoch: {}\n'
    print(template.format(round((t2-t1)/60, 4), epoch+1))


    # validation
    val_generator = DataGenerator(ashu_encoder, val_IDs, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_val, 'Group_bin', 'Scanner')
    t3 = time.time()

    for batch in range(math.ceil(len(val_IDs)/params['batch_size'])):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_PD, y_SC = val_generator.__getitem__(step_batch)
        val_loss, val_logits_PD, val_logits_SC = test_step(step_batch, X, y_PD, y_SC)
        print('\nBatch '+str(batch+1)+'/'+str(math.ceil(len(val_IDs)/params['batch_size'])))
        print("VAL LOSS PD -->", val_loss)
        for _ in range(args.batch_size):
            print("LOGITS PD -->", val_logits_PD[_])
            print("ACTUAL PD -->", y_PD[_])
            # print("LOGITS SC -->", val_logits_SC[_])
            # print("ACTUAL SC -->", y_SC[_])
        m_PD = tf.keras.metrics.binary_accuracy(y_PD.astype(float), val_logits_PD, threshold=0.5)
        m_SC = tf.keras.metrics.categorical_accuracy(tf.one_hot(y_SC, 23), val_logits_SC)
        print("VAL ACCURACY PD -->", m_PD, tf.math.reduce_mean(m_PD))
        print("VAL ACCURACY SC -->", m_SC, tf.math.reduce_mean(m_SC))

    t4 = time.time()
    template = 'VALIDATION - ETA: {} - epoch: {}\n'
    print(template.format(round((t4-t3)/60, 4), epoch+1))

    # learning rate scheduling
    LEARNING_RATE = scheduler(epoch, LEARNING_RATE)
    optimizer_encoder = Adam(learning_rate=LEARNING_RATE)
    optimizer_PD = Adam(learning_rate=LEARNING_RATE)

####################################################################################################################

encoder.save('mini_encoder_unlearned_BS'+str(args.batch_size)+'.h5')
classifier_PD.save('PD_classifier_unlearned_BS'+str(args.batch_size)+'.h5')
classifier_SC.save('SC_classifier_unlearned_BS'+str(args.batch_size)+'.h5')

####################################################################################################################