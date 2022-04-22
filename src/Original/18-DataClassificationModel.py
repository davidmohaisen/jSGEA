import networkx as nx
# import matplotlib.pyplot as plt
import netlsd
import os
import sys
import pygraphviz
import numpy as np
from shutil import copyfile
import random
import pickle
import networkx.algorithms.isomorphism as iso
from networkx.drawing.nx_agraph import write_dot
import time
import netlsd
from shutil import copyfile
import random


f = open("../ModelData/pickle/names","rb")
names = pickle.load(f)
f = open("../ModelData/pickle/features","rb")
features = pickle.load(f)


list = [[i for i in range(len(names[j]))] for j in range(len(names))]

for i in range(len(list)):
    random.shuffle(list[i])



x_train = []
x_train_names = []
y_train = []
x_test = []
x_test_names = []

y_test = []
labels = [-1,0,1,2]
for i in range(len(list)):
    if i == 0:
        continue
    counter = 0
    for j in range(len(list[i])):
        counter += 1
        if counter < 0.8*len(features[i]):
            x_train.append(features[i][list[i][j]])
            x_train_names.append(names[i][list[i][j]])
            y_train.append(labels[i])
        else:
            x_test.append(features[i][list[i][j]])
            x_test_names.append(names[i][list[i][j]])
            y_test.append(labels[i])

f = open("../ModelData/pickle/classificationData","wb")
pickle.dump([x_train,y_train,x_test,y_test,x_train_names,x_test_names],f)
