import keras
#from keras.layers.core import Layer
from keras.layers import Conv2D
import keras.backend as K
from keras.models import load_model
from numpy import savetxt
import tensorflow as tf
from tqdm import tqdm
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten

import cv2
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils
import random
import math
from tensorflow.keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

num_classes = 6

TRAIN_DIR="../input/ci-sc22-places-and-scene-recognition/train_images/train_images"
TEST_DIR="../input/ci-sc22-places-and-scene-recognition/test_images/test_images"
data = pd.read_csv("../input/ci-sc22-places-and-scene-recognition/train.csv")

n_rows = 150
n_cols = 150

def create_train_data():
    training_data = []
    for ind in data.index:
        path = os.path.join(TRAIN_DIR,data["image_name"][ind])
        img_data = cv2.imread(path)
        img_data = cv2.resize(img_data, (n_rows, n_cols))
        training_data.append([np.array(img_data),data["label"][ind]])

    random.shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for name in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, name)
        img_data = cv2.imread(path)
        img_data = cv2.resize(img_data, (n_rows, n_cols))
        testing_data.append(np.array(img_data))
    np.save('test_data.npy', testing_data)
    return testing_data

if (os.path.exists('train_data.npy')): # If you have already created the dataset:
    train_data = np.load('train_data.npy', allow_pickle=True)
else: # If dataset is not created:
    train_data = create_train_data()

if (os.path.exists('test_data.npy')):
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    test_data = create_test_data()


X=np.array([i[0] for i in train_data])
Y=np.array([i[1] for i in train_data])
#print(X.shape,Y.shape)
#Y=np_utils.to_categorical(Y,   num_classes)






X_train,  X_test,y_train, y_test = train_test_split(X,Y,test_size=0.2)

y_train = pd.get_dummies(data=y_train)
y_test = pd.get_dummies(data=y_test)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                       bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output



kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)



input_layer = Input(shape=(n_rows, n_cols, 3))

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=192,
                     filters_5x5_reduce=32,
                     filters_5x5=96,
                     filters_pool_proj=64,
                     name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=192,
                     filters_3x3_reduce=96,
                     filters_3x3=208,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_pool_proj=64,
                     name='inception_4a')


x1 = AveragePooling2D((5, 5), strides=3)(x)
x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(6, activation='softmax', name='auxilliary_output_1')(x1)

x = inception_module(x,
                     filters_1x1=160,
                     filters_3x3_reduce=112,
                     filters_3x3=224,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4d')


x2 = AveragePooling2D((5, 5), strides=3)(x)
x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(6, activation='softmax', name='auxilliary_output_2')(x2)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=384,
                     filters_3x3_reduce=192,
                     filters_3x3=384,
                     filters_5x5_reduce=48,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

x = Dropout(0.4)(x)

x = Dense(6, activation='softmax', name='output')(x)


model = Model(input_layer, x, name='inception_v1')


#model.summary()


epochs = 25
initial_lrate = 0.01

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

lr_sc = LearningRateScheduler(decay, verbose=1)

model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])


if (os.path.exists('./model.tfl')):
    model = load_model('model.tfl')
else:
    history = model.fit(X_train, y_train, validation_data=(X_test,  y_test), epochs=epochs, batch_size=256, callbacks=[lr_sc])
    model.save('model.tfl')

y_pred = model.predict(test_data)
#y_pred = y_pred.T
final_data=[]
def get_final_label(arr):
    num =np.max(arr)
    if np.max==arr[0]:
        return 0
    elif np.max==arr[1]:
        return 1
    elif np.max==arr[2]:
        return 2
    elif np.max==arr[3]:
        return 3
    elif np.max==arr[4]:
        return 4
    else :
        return 5
it=0
for name in os.listdir(TEST_DIR):
    final_data.append([name,get_final_label( y_pred[it])])
    it += 1
pd.DataFrame(final_data).to_csv("./submission.csv")



