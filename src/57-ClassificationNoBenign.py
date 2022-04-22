from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import time
import logging
import os
from numpy import argmax
import pickle
from six.moves import xrange
from sklearn.model_selection import train_test_split,KFold
import keras
import networkx as nx
# import matplotlib.pyplot as plt
import os
import sys
import pygraphviz
import numpy as np
from shutil import copyfile
import random
import pickle
import pygraphviz
from networkx.drawing.nx_agraph import write_dot
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from keras.models import model_from_json
from PIL import Image
from sklearn.utils import shuffle
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import collections
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.decomposition import PCA
FamilyNames = ["Gafgyt","Mirai","Tsunami"]
Base = "/home/ahmed/Documents/Projects/IOT-CFG-ATTACK-Journal/Pickles/"

def read_data(FamilyNames,Base):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(len(FamilyNames)):
        f = open(Base+FamilyNames[i],"rb")
        data = pickle.load(f)
        label = i
        labels = [label]*len(data)
        labels = np.asarray(labels)
        for j in range(len(data)):
            if j < 0.8*len(data):
                x_train.append(data[j])
                y_train.append(labels[j])
            else:
                x_test.append(data[j])
                y_test.append(labels[j])
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_train,y_train,x_test,y_test


# Classification

x_train,y_train,x_test,y_test = read_data(FamilyNames,Base)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# LR
clf = LogisticRegression(random_state=0).fit(x_train, y_train)
print("LR",clf.score(x_test, y_test))
f = open("../Models/SBD/LR","wb")
pickle.dump(clf,f)

# SVM
clf = SVC().fit(x_train, y_train)
print("SVM",clf.score(x_test, y_test))
f = open("../Models/SBD/SVM","wb")
pickle.dump(clf,f)

# RF
clf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
print("RF",clf.score(x_test, y_test))
f = open("../Models/SBD/RF","wb")
pickle.dump(clf,f)

# DNN
x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
model = Sequential()
model.add(keras.layers.Dense(128, activation='relu',input_shape=(x_train_n.shape[1:])))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(3, activation="softmax"))
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.fit(x_train_n,y_train, epochs=100, batch_size=16,verbose=0)
scores = model.evaluate(x_test_n, y_test, verbose=0)
print('Test accuracy:', scores[1])
model_json = model.to_json()
with open("../Models/SBD/DNN.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../Models/SBD/DNN.h5")

# CNN
x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
model = Sequential()
filter = 8
model = Sequential()
model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu",input_shape=(x_train_n.shape[1:])))
# model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
model.add(keras.layers.Conv1D(filter*2,3,padding="same",activation="relu"))
# model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(3, activation="softmax"))
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.fit(x_train_n,y_train, epochs=100, batch_size=64,verbose=0)
scores = model.evaluate(x_test_n, y_test, verbose=0)
print('Test accuracy:', scores[1])
model_json = model.to_json()
with open("../Models/SBD/CNN.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../Models/SBD/CNN.h5")
