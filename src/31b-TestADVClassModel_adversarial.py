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
samples = samples.reshape((samples.shape[0],samples.shape[1]*samples.shape[2],1))
print(samples.shape)


  #### Restore Model ####
json_file = open("../ModelData/adversarial pickles/model/Model_softmax.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../ModelData/adversarial pickles/model/Model_softmax.h5")
print("Loaded model from disk")
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])





preds = model.predict(samples)
correct = [0,0]
all = [0,0]
for i in range(len(samples)):
    label = np.argmax(classes[i])
    plabel = np.argmax(preds[i])
    all[label]+= 1
    if plabel > 0.5:
        plabel = 1
    else:
        plabel = 0
    if label == plabel:
        correct[int(plabel)] += 1
print(all)
print(correct)
# print(correct/all)
