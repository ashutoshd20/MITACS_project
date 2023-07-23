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
from tensorflow.keras.optimizers.legacy import Adam
from datagenerator import DataGenerator
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import argparse
import sys
import math
import time

LEARNING_RATE = 0.001

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-batch_size', type=int, help='batch size')
parser.add_argument('-alpha', type=float, help='alpha')
parser.add_argument('-beta', type=float, help='beta')
parser.add_argument('-cycles', type=int, help='no of cycles')
parser.add_argument('-epochs', type=int, help='no of epochs')
parser.add_argument('-en', type=str, help='pretrained encoder')
parser.add_argument('-pd', type=str, help='pretrained PD classifier')
parser.add_argument('-sc', type=str, help='pretrained SC classifier')
args = parser.parse_args()
ashu_encoder = tf.keras.models.load_model('Ashu_encoder.h5')

params = {'batch_size': args.batch_size,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }

CYCLES = args.cycles
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

# pretrained models
encoder = tf.keras.models.load_model(args.en)
classifier_PD = tf.keras.models.load_model(args.pd)
classifier_SC = tf.keras.models.load_model(args.sc)

encoder.trainable = True
classifier_PD.trainable = True
classifier_SC.trainable = True


# optimizers
optimizer_encoder = Adam(learning_rate=0.001)
optimizer_PD = Adam(learning_rate=0.002)
optimizer_SC = Adam(learning_rate=0.001, decay=0.003)


# loss
def categorical_cross_entropy_label_predictor(y_true, y_pred):
    ccelp = tf.keras.losses.BinaryCrossentropy()
    return ccelp(y_true, y_pred)/args.batch_size

def categorical_cross_entropy_domain_classifier(y_true, y_pred):
    ccedc = tf.keras.losses.CategoricalCrossentropy()
    return ccedc(y_true, y_pred)/args.batch_size

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
studies = train['Study'].unique()
np.random.seed(42)  
np.random.shuffle(studies)


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


####################################################################################################################

# training
for c in range(CYCLES):
    np.random.seed(42+c)  
    np.random.shuffle(studies)
    print("CYCLE --> "+str(c)+'\n')

    ########################
    for s in studies:
        batch_size = BATCH_SIZE
        print("STUDY --> "+str(s))

        train_aux =  train[train['Study']==s]
        IDs_list = train_aux['Subject'].to_numpy()
        train_IDs = IDs_list
        if(len(train_IDs)<batch_size): 
            batch_size=len(train_IDs)
        
        for epoch in range(EPOCHS):
            training_generator = DataGenerator(ashu_encoder,train_IDs, batch_size, (params['imagex'], params['imagey'], params['imagez']), True, fn_train, 'Group_bin', 'Scanner')
            t1 = time.time()

            for batch in range(math.ceil(len(train_IDs)/batch_size)):
                step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
                X, y_PD, y_SC = training_generator.__getitem__(step_batch)
                train_loss, logits_PD, logits_SC = train_step(step_batch, X, y_PD, y_SC)
                print('\nBatch '+str(batch+1)+'/'+str(math.ceil(len(train_IDs)/batch_size)))
                print("LOSS PD -->", train_loss)
                for _ in range(batch_size):
                    print("LOGITS PD -->", logits_PD[_])
                    print("ACTUAL PD -->", y_PD[_])
                    print("LOGITS SC -->", logits_SC[_])
                    print("ACTUAL SC -->", y_SC[_])
                m_PD = tf.keras.metrics.binary_accuracy(y_PD.astype(float), logits_PD, threshold=0.5)
                m_SC = tf.keras.metrics.categorical_accuracy(tf.one_hot(y_SC, 23), logits_SC)
                print("ACCURACY PD -->", m_PD, tf.math.reduce_mean(m_PD))
                print("ACCURACY SC -->", m_SC, tf.math.reduce_mean(m_SC))

            t2 = time.time()
            template = 'TRAINING - ETA: {} - epoch: {}\n'
            print(template.format(round((t2-t1)/60, 4), epoch+1))
    ########################

    # learning rate scheduling
    LEARNING_RATE = scheduler(epoch, LEARNING_RATE)
    optimizer_encoder = Adam(learning_rate=LEARNING_RATE)
    optimizer_PD = Adam(learning_rate=2*LEARNING_RATE)

    #model save
    if(c<=9 and c>=0 and c%2==0): 
        encoder.save('encoder_distributed_unlearned_BS'+str(BATCH_SIZE)+"_0"+str(c)+".h5") 
        classifier_PD.save('classifier_PD_distributed_unlearned_BS'+str(BATCH_SIZE)+"_0"+str(c)+".h5")
        classifier_SC.save('classifier_SC_distributed_unlearned_BS'+str(BATCH_SIZE)+"_0"+str(c)+".h5")
    elif(c>9 and c%2==1):
        encoder.save('encoder_distributed_unlearned_BS'+str(BATCH_SIZE)+"_"+str(c)+".h5")
        classifier_PD.save('classifier_PD_distributed_unlearned_BS'+str(BATCH_SIZE)+"_"+str(c)+".h5")
        classifier_SC.save('classifier_SC_distributed_unlearned_BS'+str(BATCH_SIZE)+"_"+str(c)+".h5")


####################################################################################################################

