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
from sklearn.ensemble import RandomForestClassifier
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


x_train = []
x_test = []
y_train = []
y_test = []
for i in range(len(features1)):
    for j in range(len(features1[i])):
        t = []
        for k in range(len(features1[i][j])):
            if sum_benign[k] < 100:
                t.append(features1[i][j][k])
        x_train.append(t)
        y_train.append(0)
for i in range(len(features2)):
    for j in range(len(features2[i])):
        t = []
        for k in range(len(features2[i][j])):
            if sum_benign[k] < 100:
                t.append(features2[i][j][k])
        x_train.append(t)
        y_train.append(0)
for i in range(len(features3)):
    for j in range(len(features3[i])):
        t = []
        for k in range(len(features3[i][j])):
            if sum_benign[k] < 100:
                t.append(features3[i][j][k])
        x_test.append(t)
        y_test.append(0)
for i in range(len(features4)):
    for j in range(len(features4[i])):
        t = []
        for k in range(len(features4[i][j])):
            if sum_benign[k] < 100:
                t.append(features4[i][j][k])
        if sum(t) == 0 :
            continue
        if j < 0.8*len(features4[i]):
            x_train.append(t)
            y_train.append(1)
        # else:
        #     x_test.append(t)
        #     y_test.append(1)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(collections.Counter(y_test))
print(collections.Counter(y_train))

# RF
clf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
print("RF",clf.score(x_test, y_test))
f = open("../Models/SBD/SBDRF","wb")
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
model.add(keras.layers.Dense(2, activation="softmax"))
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.fit(x_train_n,y_train, epochs=5, batch_size=64,verbose=1)
scores = model.evaluate(x_test_n, y_test, verbose=0)
print('Test accuracy:', scores[1])
model_json = model.to_json()
with open("../Models/SBD/SBDDNN.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../Models/SBD/SBDDNN.h5")

# CNN
x_train_n = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test_n = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
model = Sequential()
filter = 8
model = Sequential()
model.add(keras.layers.Conv1D(filter,8,padding="same",activation="relu",input_shape=(x_train_n.shape[1:])))
# model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
model.add(keras.layers.Conv1D(filter*2,8,padding="same",activation="relu"))
# model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(2, activation="softmax"))
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.fit(x_train_n,y_train, epochs=5, batch_size=32,verbose=1)
scores = model.evaluate(x_test_n, y_test, verbose=0)
print('Test accuracy:', scores[1])
model_json = model.to_json()
with open("../Models/SBD/SBDCNN.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../Models/SBD/SBDCNN.h5")
