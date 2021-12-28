import cv2
import numpy as np
import os
from random import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import pandas as pd
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# ==================================================================
#
# ==================================================================
TRAIN_DIR = './dataset/train_images'
TEST_DIR = './dataset/test_images'
IMG_SIZE = 150
LR = 0.001
MODEL_NAME = 'scene-detection-cnn'
data = pd.read_csv("./dataset/train.csv")


# ==================================================================
# Create Data
# ==================================================================
def get_label(id):
    if id == 0:
        return np.array([1, 0, 0, 0, 0, 0])
    elif id == 1:
        return np.array([0, 1, 0, 0, 0, 0])
    elif id == 2:
        return np.array([0, 0, 1, 0, 0, 0])
    elif id == 3:
        return np.array([0, 0, 0, 1, 0, 0])
    elif id == 4:
        return np.array([0, 0, 0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 0, 0, 1])


def create_train_data():
    training_data = []
    for ind in data.index:
        path = os.path.join(TRAIN_DIR, data["image_name"][ind])
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), get_label(data["label"][ind])])
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(img_data)
        # plt.show()
    # shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for ind in data.index:
        path = os.path.join(TRAIN_DIR, data["image_name"][ind])
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append(img_data)
    # shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


if (os.path.exists('train_data.npy')):  # If you have already created the dataset:
    train_data = np.load('train_data.npy', allow_pickle=True)
    # train_data = create_train_data()
else:  # If dataset is not created:
    train_data = create_train_data()

# print(train_data.shape)
if (os.path.exists('test_data.npy')):
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    test_data = create_test_data()

# ==================================================================
# CNN Model
# ==================================================================
train = train_data
test = test_data
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

tf.reset_default_graph()
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 5, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)

fully_layer = fully_connected(pool5, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
print(X_train.shape)

if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
              validation_set=({'input': X_test}, {'targets': y_test}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')


# ==================================================================
# Testing
# ==================================================================
# for img in X_test:
#     # test_img = img.reshape(IMG_SIZE, IMG_SIZE, 1)
#     prediction = model.predict([img])[0]
def get_final_label(arr):
    num = np.max(arr)
    if num == arr[0]:
        return 0
    elif num == arr[1]:
        return 1
    elif num == arr[2]:
        return 2
    elif num == arr[3]:
        return 3
    elif num == arr[4]:
        return 4
    else:
        return 5


final_data = []
it = 0
for name in os.listdir(TEST_DIR):
    path = os.path.join(TEST_DIR, name)
    img = cv2.imread(path, 0)
    test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 1)
    pred = model.predict([test_img])[0]
    final_data.append([name, get_final_label(pred)])
    it += 1
pd.DataFrame(final_data).to_csv("./submission.csv")
# ==================================================================
# ==================================================================
