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
f1 = open("../subgraph_Isomorphic/1.pkl","rb")
features1 = pickle.load(f1)

f2 = open("../subgraph_Isomorphic/2.pkl","rb")
features2 = pickle.load(f2)

f3 = open("../subgraph_Isomorphic/3.pkl","rb")
features3 = pickle.load(f3)

f4 = open("../subgraph_Isomorphic/4.pkl","rb")
features4 = pickle.load(f4)

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



x = []
y = []
for i in range(len(features1)):
    for j in range(len(features1[i])):
        t = []
        for k in range(len(features1[i][j])):
            if sum_benign[k] < 100:
                t.append(0)
        x.append(t)
        y.append(0)
for i in range(len(features2)):
    for j in range(len(features2[i])):
        t = []
        for k in range(len(features2[i][j])):
            if sum_benign[k] < 100:
                t.append(0)
        x.append(t)
        y.append(0)
for i in range(len(features3)):
    for j in range(len(features3[i])):
        t = []
        for k in range(len(features3[i][j])):
            if sum_benign[k] < 100:
                t.append(0)
        x.append(t)
        y.append(0)
for i in range(len(features4)):
    for j in range(len(features4[i])):
        t = []
        for k in range(len(features4[i][j])):
            if sum_benign[k] < 100:
                t.append(features4[i][j][k])
        if sum(t) == 0 :
            continue
        x.append(t)
        y.append(1)

samples = np.asarray(x)
classes = np.asarray(y)
c = collections.Counter(y)
print(c)

classes = keras.utils.to_categorical(classes, num_classes=2)
print(classes.shape)
samples = samples.reshape((samples.shape[0],samples.shape[1],1))
print(samples.shape)


filter = 32
# create model
model = Sequential()
model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu"))
model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
model.add(keras.layers.MaxPooling1D())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv1D(filter*2,3,padding="same",activation="relu"))
model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))
model.add(keras.layers.MaxPooling1D())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation="softmax"))
# model.add(keras.layers.Dense(1, activation="sigmoid"))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

# Fit the model
model.fit(samples,classes, epochs=30, batch_size=32)

preds = model.predict(samples)
correct = 0
all = 0
for i in range(len(samples)):
    label = np.argmax(classes[i])
    plabel = np.argmax(preds[i])
    all+= 1
    if label == plabel:
        correct += 1
print(all)
print(correct)
print(correct/all)


model_json = model.to_json()
with open("../subgraph_Isomorphic/model/Model_softmax_remove_zero.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../subgraph_Isomorphic/model/Model_softmax_remove_zero.h5")
print("model saved")
