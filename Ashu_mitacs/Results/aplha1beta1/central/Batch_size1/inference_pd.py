from pickle import FALSE
from datagenerator_pd import DataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from numpy.random import seed
import SimpleITK as sitk
seed(1)
tf.random.set_seed = 1
random.seed(1)
import argparse
import os
import sys
os.environ['TF_DETERMINISTIC_OPS'] = '1'


autoencoder_Ashu = tf.keras.models.load_model('Ashu_encoder.h5')

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn', type=str, help='filename to infer')
parser.add_argument('-en', type=str, help='model to infer')
parser.add_argument('-pd', type=str, help='model to infer')
parser.add_argument('-o', type=str, help='output name')
args = parser.parse_args()
import csv

def model_eval(y_test, y_pred_raw):
    y_pred = (y_pred_raw>=0.5)
    y_pred = y_pred.astype(int)
    y_test = y_test.to_frame()
    y_test = y_test.rename(columns={'Group_bin': 'ground_truth'})
    y_test['preds'] = y_pred
    y_test['preds_raw'] = y_pred_raw
    return y_test



def compute_metrics(df,fn):
    y_test = df['ground_truth'].values
    y_pred = df['preds'].values
    cm = confusion_matrix(y_test, y_pred)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    print("confusion_matrix =",cm)
    cm_df = pd.DataFrame(cm,
            index = ['HC','PD'], 
            columns = ['HC','PD'])
    plt.figure(figsize=(10,8))
    
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig(fn,bbox_inches='tight')
    plt.show()
    
    ac=accuracy_score(y_test, y_pred)
    print("accuracy =",ac)
    tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    print("metrics =",[ac, sens, spec])
    return [ac, sens, spec]
    

params = {'batch_size': 5,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }


fn_test = args.fn
test = pd.read_csv(fn_test)
IDs_list=test['Subject'].to_numpy()
test_IDs=IDs_list
test_generator=DataGenerator(autoencoder_Ashu, test_IDs, 1, (params['imagex'], params['imagey'], params['imagez']), False, fn_test, 'Group_bin')


# reload the best performing model for the evaluation
encoder=tf.keras.models.load_model(args.en)
pd_classifier=tf.keras.models.load_model(args.pd)
# make sure that model weights are non-trainable
encoder.trainable=False
pd_classifier.trainable=False

name=args.o

print(encoder.get_weights())

#Make predictions and save the results
y_test = test['Group_bin']
y_pred_encoder=encoder.predict(test_generator)
y_pred = pd_classifier.predict(y_pred_encoder)
preds = model_eval(y_test, y_pred)

df = pd.merge(preds, test, left_index=True, right_index=True)
df.to_csv(name+'_predictions.csv')

df_male = df.loc[df['Sex']=='M']
df_female = df.loc[df['Sex']=='F']

print("######Overall Metrics######")
metrics = compute_metrics(df,"cm_agg_"+name)
print("########Male Metrics#######")
metrics_male = compute_metrics(df_male,"cm_male_"+name)
print("######Female Metrics#######")
metrics_female = compute_metrics(df_female,"cm_female_"+name)

metrics_df = pd.DataFrame(['Acc', 'Sens', 'Spec'], columns=['metrics'])
metrics_df = metrics_df.set_index('metrics')
metrics_df['Aggregate'] = metrics
metrics_df['Male'] = metrics_male
metrics_df['Female'] = metrics_female
metrics_df.to_csv(name+'_metrics.csv')


