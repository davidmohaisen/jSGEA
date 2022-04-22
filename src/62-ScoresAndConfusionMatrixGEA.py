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

def DrawConfusionMatrix(co):
    co = list(co)
    max = [0]*4
    for i in range(len(co)):
        for j in range(len(co[i])):
            max[j] += co[j][i]
    coN = []
    for i in range(len(co)):
        coN.append([])
        for j in range(len(co[i])):
            if i ==0:
                coN[-1].append(0)
                continue
            coN[-1].append(float(float(co[i][j])/float(max[i])))
            coN[i][j] = round(coN[i][j],3)
    co = coN


    co = np.asarray(co)
    df = pd.DataFrame(co, columns=["Benign","Gafgyt","Mirai","Tsunami"],index=["Benign","Gafgyt","Mirai","Tsunami"])
    sns.set(font_scale=2.0)
    ax = sns.heatmap(df,annot_kws={"size": 25},annot=True,cmap=sns.cubehelix_palette(500, hue=0.01, rot=0, light=0.98, dark=0),fmt='g',cbar=False)
    plt.show()

def read_data(Family,Mod,label,detection=True):

    base = "../Pickles/SGEA/"

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    bModel = "Classification"
    if detection:
        bModel = "Detection"
    f = open(base+bModel+Mod+Family,"rb")
    data = pickle.load(f)
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

def read_data_Detection(Mod,detection=True):

    base = "../Pickles/SGEA/"
    bModel = "Classification"
    if detection:
        bModel = "Detection"
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    Familys = ["Gafgyt","Mirai","Tsunami"]
    for Family in Familys:
        label = Familys.index(Family) +1
        f = open(base+bModel+Mod+Family,"rb")
        data = pickle.load(f)
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



FamilyNames = ["Gafgyt","Mirai","Tsunami"]
Sizes = ["Small","Median","Large"]

# Detection
x_train,y_train,x_test,y_test = read_data_Detection("RF",detection=True)
f = open("../Models/Detection/RF","rb")
clf = pickle.load(f)
y_true = y_test
y_pred = clf.predict(x_test)
print("RF",clf.score(x_test, y_test),f1_score(y_true, y_pred),1-recall_score(y_true, y_pred, pos_label=1),1-recall_score(y_true, y_pred, pos_label=0))

# DNN
x_train,y_train,x_test,y_test = read_data_Detection("DNN",detection=True)
x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
json_file = open('../Models/Detection/DNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../Models/Detection/DNN.h5")
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
scores = model.evaluate(x_test_n, y_test, verbose=0)
y_true = y_test
y_pred = np.argmax(model.predict(x_test_n),axis=1)
print('DNN', scores[1],f1_score(y_true, y_pred),1-recall_score(y_true, y_pred, pos_label=1),1-recall_score(y_true, y_pred, pos_label=0))

# CNN
x_train,y_train,x_test,y_test = read_data_Detection("CNN",detection=True)

x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
json_file = open('../Models/Detection/CNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../Models/Detection/CNN.h5")
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
scores = model.evaluate(x_test_n, y_test, verbose=0)
y_true = y_test
y_pred = np.argmax(model.predict(x_test_n),axis=1)
print('CNN', scores[1],f1_score(y_true, y_pred),1-recall_score(y_true, y_pred, pos_label=1),1-recall_score(y_true, y_pred, pos_label=0))

exit()

for Family in FamilyNames:
    label = FamilyNames.index(Family) +1
    for Size in Sizes:
        print(Family,Size)
        # Classification

        x_train,y_train,x_test,y_test = read_data(FamilyNames,Base,Family,Size,label,detection=False)
        # print(x_train.shape)
        # print(y_train.shape)
        # print(x_test.shape)
        # print(y_test.shape)


        # # LR
        # f = open("../Models/Classification/LR","rb")
        # clf = pickle.load(f)
        # print("LR",clf.score(x_test, y_test))
        #
        # # SVM
        # f = open("../Models/Classification/SVM","rb")
        # clf = pickle.load(f)
        # print("SVM",clf.score(x_test, y_test))
        # RF
        f = open("../Models/Classification/RF","rb")
        clf = pickle.load(f)
        y_true = y_test
        y_pred = clf.predict(x_test)
        print("RF",1-clf.score(x_test, y_test),f1_score(y_true, y_pred,average="weighted"))
        # co = confusion_matrix(y_true, y_pred)
        # DrawConfusionMatrix(co)
        # DNN
        x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        json_file = open('../Models/Classification/DNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("../Models/Classification/DNN.h5")
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
        scores = model.evaluate(x_test_n, y_test, verbose=0)
        y_true = y_test
        y_pred = np.argmax(model.predict(x_test_n),axis=1)
        print('DNN', 1-scores[1],f1_score(y_true, y_pred,average="weighted"))
        # co = confusion_matrix(y_true, y_pred)
        # DrawConfusionMatrix(co)
        # CNN
        x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        json_file = open('../Models/Classification/CNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("../Models/Classification/CNN.h5")
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
        scores = model.evaluate(x_test_n, y_test, verbose=0)
        y_true = y_test
        y_pred = np.argmax(model.predict(x_test_n),axis=1)
        print('CNN', 1-scores[1],f1_score(y_true, y_pred,average="weighted"))
        # co = confusion_matrix(y_true, y_pred)
        # DrawConfusionMatrix(co)
