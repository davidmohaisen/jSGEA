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
from os import listdir
from os.path import isfile, join
from networkx.algorithms import isomorphism
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
import collections


  #### Restore Model ####
json_file = open("../subgraph_Isomorphic/model/Model_softmax.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../subgraph_Isomorphic/model/Model_softmax.h5")
print("Loaded model from disk")
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])


#benign#

print("Benign:")
f1 = open("../subgraph_Isomorphic/1.pkl","rb")
features1 = pickle.load(f1)
x = []
y = []
for i in range(len(features1)):
    for j in range(len(features1[i])):
        x.append(features1[i][j])
        y.append(0)
x = x[:600]
y = y[:600]
samples = np.asarray(x)
classes = np.asarray(y)

# classes = keras.utils.to_categorical(classes, num_classes=4)
samples = samples.reshape((samples.shape[0],samples.shape[1],1))



# Fit the model

preds = model.predict(samples)
correct = 0
all = 0
for i in range(len(samples)):
    label = classes[i]
    plabel = np.argmax(preds[i])
    all+= 1
    if label == plabel:
        correct += 1
# print(all)
# print(correct)
print(correct/all)


#SGEA#
print("SGEA:")
f1 = open("../subgraph_Isomorphic/SGEA.pkl","rb")
features1 = pickle.load(f1)
x = []
y = []
for i in range(len(features1)):
    for j in range(len(features1[i])):
        x.append(features1[i][j])
        y.append(1)
samples = np.asarray(x)
classes = np.asarray(y)

# classes = keras.utils.to_categorical(classes, num_classes=4)
samples = samples.reshape((samples.shape[0],samples.shape[1],1))



# Fit the model

preds = model.predict(samples)
correct = 0
all = 0
for i in range(len(samples)):
    label = classes[i]
    plabel = np.argmax(preds[i])
    all+= 1
    if label == plabel:
        correct += 1
# print(all)
# print(correct)
print(correct/all)

#GEA_small#
print("GEA Small:")
f1 = open("../subgraph_Isomorphic/GEA_SM.pkl","rb")
features1 = pickle.load(f1)
x = []
y = []
for i in range(len(features1)):
    for j in range(len(features1[i][0])):
        x.append(features1[i][0][j])
        y.append(1)
samples = np.asarray(x)
classes = np.asarray(y)

# classes = keras.utils.to_categorical(classes, num_classes=4)
samples = samples.reshape((samples.shape[0],samples.shape[1],1))



# Fit the model

preds = model.predict(samples)
correct = 0
all = 0
for i in range(len(samples)):
    label = classes[i]
    plabel = np.argmax(preds[i])
    all+= 1
    if label == plabel:
        correct += 1
# print(all)
# print(correct)
print(correct/all)



#GEA_Medium#
print("GEA Medium:")
f1 = open("../subgraph_Isomorphic/GEA_SM.pkl","rb")
features1 = pickle.load(f1)
x = []
y = []
for i in range(len(features1)):
    for j in range(len(features1[i][1])):
        x.append(features1[i][1][j])
        y.append(1)
samples = np.asarray(x)
classes = np.asarray(y)
c = collections.Counter(y)

# classes = keras.utils.to_categorical(classes, num_classes=4)
samples = samples.reshape((samples.shape[0],samples.shape[1],1))



# Fit the model

preds = model.predict(samples)
correct = 0
all = 0
for i in range(len(samples)):
    label = classes[i]
    plabel = np.argmax(preds[i])
    all+= 1
    if label == plabel:
        correct += 1
# print(all)
# print(correct)
print(correct/all)

# #GEA_Large#
# print("GEA Large:")
# f1 = open("../subgraph_Isomorphic/GEA_SM.pkl","rb")
# features1 = pickle.load(f1)
# x = []
# y = []
# for i in range(len(features1)):
#     for j in range(len(features1[i][1])):
#         x.append(features1[i][1][j])
#         y.append(1)
# samples = np.asarray(x)
# classes = np.asarray(y)
# c = collections.Counter(y)
#
# # classes = keras.utils.to_categorical(classes, num_classes=4)
# samples = samples.reshape((samples.shape[0],samples.shape[1],1))
#
#
#
# # Fit the model
#
# preds = model.predict(samples)
# correct = 0
# all = 0
# for i in range(len(samples)):
#     label = classes[i]
#     plabel = round(preds[i][0],0)
#     all+= 1
#     if label == plabel:
#         correct += 1
# # print(all)
# # print(correct)
# print(correct/all)
