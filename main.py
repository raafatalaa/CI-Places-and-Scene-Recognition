import inception_resnet_v2 as incep_v2
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

tf.disable_eager_execution()

n_rows = 150
n_cols = 150

#read data

TRAIN_DIR="./dataset/train_images"
data = pd.read_csv("./dataset/train.csv")
data =data.iloc[0:1,:]
#print(data)
def create_train_data():
    training_data = []
    for ind in data.index:
        path = os.path.join(TRAIN_DIR,data["image_name"][ind])
        img_data = cv2.imread(path)#mmk n2ml hna grey scale
        img_data = cv2.resize(img_data, (n_rows, n_cols))
        training_data.append([np.array(img_data),data["label"][ind]])
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(img_data)
        # plt.show()
    random.shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

print(create_train_data())



#define model

def define_model(model, is_training):
    model.Image = tf.placeholder(tf.float32, shape=[None, n_rows, n_cols, 3])
    with incep_v2.slim.arg_scope(incep_v2.inception_resnet_v2_arg_scope()):
        model.logits, model.end_points = incep_v2.inception_resnet_v2(model.Image, is_training=False)

sess =tf.Session()

class Model_Class:
    def __init__(self, is_training):
        define_model(self, is_training=is_training)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
with tf.device('/cpu:0'):
    model = Model_Class(False)


