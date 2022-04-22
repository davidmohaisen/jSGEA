from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from sklearn.metrics import confusion_matrix
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
import matplotlib.pyplot as plt
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
from keras.models import model_from_json
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import seaborn as sns
FamilyNames = ["Benign","Gafgyt","Mirai","Tsunami"]
Base = "/home/ahmed/Documents/Projects/IOT-CFG-ATTACK-Journal/Pickles/"



def read_data(FamilyNames,Base,detection=True):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(len(FamilyNames)):
        f = open(Base+FamilyNames[i],"rb")
        data = pickle.load(f)
        label = i
        if detection and label > 1:
            label = 1
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


# Detection

x_train,y_train,x_test,y_test = read_data(FamilyNames,Base,detection=False)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# RF
f = open("../Models/Detection/RF","rb")
clf = pickle.load(f)
y_true = y_test
y_pred = clf.predict(x_test)

All = [600,600,600,219]
Incorrect = [0,0,0,0]
for i in range(len(y_true)):
    if y_true[i] == 0 and y_pred[i] != 0:
        Incorrect[0] += 1
    elif y_true[i] != 0 and y_pred[i] == 0:
        Incorrect[y_true[i]] += 1

f = open("../Models/SBD/RF","rb")
clf = pickle.load(f)
y_pred_class = clf.predict(x_test)

for i in range(len(y_true)):
    if y_pred[i] == 0 or y_true[i] ==0:
        continue
    if y_true[i] != (y_pred_class[i]+1):
        Incorrect[y_true[i]] += 1

print("RF",1-np.asarray(Incorrect)/np.asarray(All))

MergedPreds = []
for i in range(len(y_pred)):
    if y_true[i] == 0:
        MergedPreds.append(y_pred[i])
    else:
        if y_pred[i] == 0:
            MergedPreds.append(0)
        else:
            MergedPreds.append(y_pred_class[i]+1)

print("RF",(1-sum(Incorrect)/sum(All)),f1_score(y_true,MergedPreds,average="weighted"))


# DNN
x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
json_file = open('../Models/Detection/DNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../Models/Detection/DNN.h5")
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
y_true = y_test
y_pred = np.argmax(model.predict(x_test_n),axis=1)
All = [600,600,600,219]
Incorrect = [0,0,0,0]
for i in range(len(y_true)):
    if y_true[i] == 0 and y_pred[i] != 0:
        Incorrect[0] += 1
    elif y_true[i] != 0 and y_pred[i] == 0:
        Incorrect[y_true[i]] += 1
x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
json_file = open('../Models/SBD/DNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../Models/SBD/DNN.h5")
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
y_pred_class = np.argmax(model.predict(x_test_n),axis=1)
for i in range(len(y_true)):
    if y_pred[i] == 0 or y_true[i] ==0:
        continue
    if y_true[i] != (y_pred_class[i]+1):
        Incorrect[y_true[i]] += 1

print("DNN",1-np.asarray(Incorrect)/np.asarray(All))
MergedPreds = []
for i in range(len(y_pred)):
    if y_true[i] == 0:
        MergedPreds.append(y_pred[i])
    else:
        if y_pred[i] == 0:
            MergedPreds.append(0)
        else:
            MergedPreds.append(y_pred_class[i]+1)

print("DNN",(1-sum(Incorrect)/sum(All)),f1_score(y_true,MergedPreds,average="weighted"))


# CNN
x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
json_file = open('../Models/Detection/CNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../Models/Detection/CNN.h5")
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
y_true = y_test
y_pred = np.argmax(model.predict(x_test_n),axis=1)
All = [600,600,600,219]
Incorrect = [0,0,0,0]
for i in range(len(y_true)):
    if y_true[i] == 0 and y_pred[i] != 0:
        Incorrect[0] += 1
    elif y_true[i] != 0 and y_pred[i] == 0:
        Incorrect[y_true[i]] += 1
x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
json_file = open('../Models/SBD/CNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../Models/SBD/CNN.h5")
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
y_pred_class = np.argmax(model.predict(x_test_n),axis=1)
for i in range(len(y_true)):
    if y_pred[i] == 0 or y_true[i] ==0:
        continue
    if y_true[i] != (y_pred_class[i]+1):
        Incorrect[y_true[i]] += 1

print("CNN",1-np.asarray(Incorrect)/np.asarray(All))
MergedPreds = []
for i in range(len(y_pred)):
    if y_true[i] == 0:
        MergedPreds.append(y_pred[i])
    else:
        if y_pred[i] == 0:
            MergedPreds.append(0)
        else:
            MergedPreds.append(y_pred_class[i]+1)

print("CNN",(1-sum(Incorrect)/sum(All)),f1_score(y_true,MergedPreds,average="weighted"))
