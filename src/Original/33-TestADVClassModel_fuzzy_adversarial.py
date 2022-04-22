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
import keras
from keras.models import Sequential
from keras.models import model_from_json


f = open("../PredictSeq/mapping","rb")
mapping = pickle.load(f)

FamilyNames = ["Benign","Gafgyt","Mirai","Tsunami"]
patternOrigin = ["Gafgyt","Mirai","Tsunami"]
uniqueNames = [[],[],[],[]]
filesNames = [[],[],[],[]]
for i in range(len(mapping)):
    for j in range(len(mapping[i])):
        uniqueNames[i].append(mapping[i][j][0])
        filesNames[i].append(mapping[i][j][1])

sizes = "1000"

samples = []
classes = []

f = open("../ModelData/adversarial pickles/graphs_adv_GEA/sequences/"+sizes,"rb")
data = pickle.load(f)


for sample in data:
    # if sample[1] != 'Large/':
    #     continue
    try:
        sampleClass = 1
        classes.append(sampleClass)
        samples.append(sample[3])
    except:
        continue

f = open("../ModelData/adversarial pickles/graphs_adv_SGEA/sequences/"+sizes,"rb")
data = pickle.load(f)
data = data[418:]


for sample in data:
    try:
        sampleClass = 1
        classes.append(sampleClass)
        samples.append(sample[2])
    except:
        continue


classes = np.asarray(classes)
samples = np.asarray(samples)

classes = keras.utils.to_categorical(classes, num_classes=2)
print(classes.shape)
samples = samples.reshape((samples.shape[0],samples.shape[1]*samples.shape[2]))
print(samples.shape)

num_graphs = []
more = 0
for i in range(len(samples)):
    num_graphs.append(sum(samples[i]))
    if sum(samples[i]) >= 5:
        more += 1
num_graphs.sort()
# print(num_graphs)
print(more,len(num_graphs))
