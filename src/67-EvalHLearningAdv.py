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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def read_data_SGEA(mod):
    FamilyNames = ["Gafgyt","Mirai","Tsunami"]
    Base = "../Pickles/SubgraphIso/SGEA"

    x_test = []
    y_test = []
    for i in range(len(FamilyNames)):
        f = open(Base+mod+FamilyNames[i],"rb")
        data = pickle.load(f)
        label = 1
        labels = [label]*len(data)
        labels = np.asarray(labels)
        for j in range(len(data)):
            x_test.append(data[j])
            y_test.append(labels[j])
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_test,y_test


def read_data_GEA(size):
    FamilyNames = ["Gafgyt","Mirai","Tsunami"]
    Base = "../Pickles/SubgraphIso/GEA"

    x_test = []
    y_test = []
    for i in range(len(FamilyNames)):
        f = open(Base+FamilyNames[i]+size,"rb")
        data = pickle.load(f)
        label = 1
        labels = [label]*len(data)
        labels = np.asarray(labels)
        for j in range(len(data)):
            x_test.append(data[j])
            y_test.append(labels[j])
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_test,y_test


f1 = open("../subgraph_Isomorphic/1.pkl","rb")
features1 = pickle.load(f1)

f2 = open("../subgraph_Isomorphic/2.pkl","rb")
features2 = pickle.load(f2)

f3 = open("../subgraph_Isomorphic/3.pkl","rb")
features3 = pickle.load(f3)

sum_benign = [0]*30000
for i in range(len(features1)):
    for j in range(len(features1[i])):
        for k in range(len(features1[i][j])):
            sum_benign[k] += features1[i][j][k]
for i in range(len(features2)):
    for j in range(len(features2[i])):
        for k in range(len(features2[i][j])):
            sum_benign[k] += features2[i][j][k]
for i in range(len(features3)):
    for j in range(len(features3[i])):
        for k in range(len(features3[i][j])):
            sum_benign[k] += features3[i][j][k]

x_test_benign = []
for i in range(len(features3)):
    for j in range(len(features3[i])):
        t = []
        for k in range(len(features3[i][j])):
            if sum_benign[k] < 100:
                t.append(features3[i][j][k])
        x_test_benign.append(t)
x_test_benign = np.asarray(x_test_benign)


maxFPRs = [0.01,0.03,0.05,0.1]
for maxFPR in maxFPRs:
    print(maxFPR)
    # Detection
    print("GEA Evaluation:")
    sizes = ["Small","Median","Large"]
    for size in sizes:
        print(size)
        x_test,y_test = read_data_GEA(size)

        x_test_n = []
        for j in range(len(x_test)):
            t = []
            for k in range(len(x_test[j])):
                if sum_benign[k] < 100:
                    t.append(x_test[j][k])
            x_test_n.append(t)

        x_test = np.asarray(x_test_n)
        y_test = np.asarray(y_test)



        # RF
        f = open("../Models/SBD/SBDRF","rb")
        clf = pickle.load(f)
        benign_proba = clf.predict_proba(x_test_benign)
        Adv_proba = clf.predict_proba(x_test)
        proba = np.concatenate((benign_proba,Adv_proba))
        y_proba_true = np.concatenate(([[1,0]]*len(x_test_benign),[[0,1]]*len(x_test)))

        print("RF",roc_auc_score(y_proba_true,proba,max_fpr=maxFPR))

        # y_proba_true_ROC = np.concatenate(([0]*len(x_test_benign),[1]*len(x_test)))
        # fpr, tpr, thresholds = roc_curve(y_proba_true_ROC, proba[:, 1])
        # plt.plot(fpr, tpr, marker='.', label='Logistic')
        # # axis labels
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # # show the legend
        # plt.legend()
        # # show the plot
        # plt.show()
        # exit()


        # DNN
        x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        x_test_benign_n = np.reshape(x_test_benign,(x_test_benign.shape[0],x_test_benign.shape[1],1))
        json_file = open('../Models/SBD/SBDDNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("../Models/SBD/SBDDNN.h5")
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
        benign_proba = model.predict(x_test_benign_n)
        Adv_proba =  model.predict(x_test_n)
        proba = np.concatenate((benign_proba,Adv_proba))
        y_proba_true = np.concatenate(([[1,0]]*len(x_test_benign),[[0,1]]*len(x_test)))

        print("DNN",roc_auc_score(y_proba_true,proba,max_fpr=maxFPR))


        x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        x_test_benign_n = np.reshape(x_test_benign,(x_test_benign.shape[0],x_test_benign.shape[1],1))
        json_file = open('../Models/SBD/SBDCNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("../Models/SBD/SBDCNN.h5")
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
        benign_proba = model.predict(x_test_benign_n)
        Adv_proba =  model.predict(x_test_n)
        proba = np.concatenate((benign_proba,Adv_proba))
        y_proba_true = np.concatenate(([[1,0]]*len(x_test_benign),[[0,1]]*len(x_test)))

        print("CNN",roc_auc_score(y_proba_true,proba,max_fpr=maxFPR))




    # Detection
    print("SGEA Evaluation:")

    x_test,y_test = read_data_SGEA("RF")

    x_test_n = []
    for j in range(len(x_test)):
        t = []
        for k in range(len(x_test[j])):
            if sum_benign[k] < 100:
                t.append(x_test[j][k])
        x_test_n.append(t)

    x_test = np.asarray(x_test_n)
    y_test = np.asarray(y_test)



    # RF
    f = open("../Models/SBD/SBDRF","rb")
    clf = pickle.load(f)
    benign_proba = clf.predict_proba(x_test_benign)
    Adv_proba = clf.predict_proba(x_test)
    proba = np.concatenate((benign_proba,Adv_proba))
    y_proba_true = np.concatenate(([[1,0]]*len(x_test_benign),[[0,1]]*len(x_test)))

    print("RF",roc_auc_score(y_proba_true,proba,max_fpr=maxFPR))


    # DNN

    x_test,y_test = read_data_SGEA("DNN")

    x_test_n = []
    for j in range(len(x_test)):
        t = []
        for k in range(len(x_test[j])):
            if sum_benign[k] < 100:
                t.append(x_test[j][k])
        x_test_n.append(t)

    x_test = np.asarray(x_test_n)
    y_test = np.asarray(y_test)



    x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    x_test_benign_n = np.reshape(x_test_benign,(x_test_benign.shape[0],x_test_benign.shape[1],1))
    json_file = open('../Models/SBD/SBDDNN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../Models/SBD/SBDDNN.h5")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
    benign_proba = model.predict(x_test_benign_n)
    Adv_proba =  model.predict(x_test_n)
    proba = np.concatenate((benign_proba,Adv_proba))
    y_proba_true = np.concatenate(([[1,0]]*len(x_test_benign),[[0,1]]*len(x_test)))

    print("DNN",roc_auc_score(y_proba_true,proba,max_fpr=maxFPR))

    x_test,y_test = read_data_SGEA("CNN")

    x_test_n = []
    for j in range(len(x_test)):
        t = []
        for k in range(len(x_test[j])):
            if sum_benign[k] < 100:
                t.append(x_test[j][k])
        x_test_n.append(t)

    x_test = np.asarray(x_test_n)
    y_test = np.asarray(y_test)


    x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    x_test_benign_n = np.reshape(x_test_benign,(x_test_benign.shape[0],x_test_benign.shape[1],1))
    json_file = open('../Models/SBD/SBDCNN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../Models/SBD/SBDCNN.h5")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
    benign_proba = model.predict(x_test_benign_n)
    Adv_proba =  model.predict(x_test_n)
    proba = np.concatenate((benign_proba,Adv_proba))
    y_proba_true = np.concatenate(([[1,0]]*len(x_test_benign),[[0,1]]*len(x_test)))

    print("CNN",roc_auc_score(y_proba_true,proba,max_fpr=maxFPR))
