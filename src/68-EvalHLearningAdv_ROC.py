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

fpr, tpr = [],[]

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

    y_proba_true_ROC = np.concatenate(([0]*len(x_test_benign),[1]*len(x_test)))
    fprT, tprT, _ = roc_curve(y_proba_true_ROC, proba[:, 1])
    fpr.append(fprT)
    tpr.append(tprT)

    # plt.plot(fpr, tpr,'k-', linewidth=2)
    # # axis labels
    # plt.xlabel('False Positive Rate', fontsize=20)
    # plt.ylabel('True Positive Rate', fontsize=20)
    # # show the plot
    # axes = plt.gca()
    # axes.set_xlim([0,1])
    # axes.set_ylim([0,1])
    # axes.tick_params(axis='both', which='major', labelsize=16)
    # plt.show()
    # exit()

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
y_proba_true_ROC = np.concatenate(([0]*len(x_test_benign),[1]*len(x_test)))
fprT, tprT, _ = roc_curve(y_proba_true_ROC, proba[:, 1])
fpr.append(fprT)
tpr.append(tprT)


labels = ["GEA-Small","GEA-Median","GEA-Large","SGEA"]
colors = ["k-","g-","b-","m-"]
for i in range(len(fpr)):
    plt.plot(fpr[i], tpr[i],colors[i], linewidth=2,label=labels[i])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
axes = plt.gca()
axes.set_xlim([-0.0001,1])
axes.set_ylim([0.0,1.0025])
axes.tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc="lower right",prop={'size': 16})
plt.show()
