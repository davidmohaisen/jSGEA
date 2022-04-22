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
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.models import model_from_json
f = open("../ModelData/pickle/detectionData","rb")
x_train,y_train,x_test,y_test,_, _ = pickle.load(f)

x_train = np.asarray(x_train)
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = np.asarray(x_test)
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

# y_train = keras.utils.to_categorical(y_train)
# y_test = keras.utils.to_categorical(y_test)

y_train = np.asarray(y_train)
y_train = y_train.reshape((y_train.shape[0],1))

y_test = np.asarray(y_test)
y_test = y_test.reshape((y_test.shape[0],1))



filter = 128
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
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(2, activation="softmax"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# Compile model
# model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, epochs=100, batch_size=32)

scores = model.evaluate(x_test, y_test)
#
# preds = model.predict(x_test)
# classes = [0,0]
# all = [0,0]
# for i in range(len(y_test)):
#     label = np.argmax(y_test[i])
#     plabel = np.argmax(preds[i])
#     all[label]+= 1
#     if label == plabel:
#         classes[label] += 1
# print(all)
# print(classes)
#
# print(sum(classes)/sum(all))

preds = model.predict(x_test)
correct = 0
all = 0
for i in range(len(y_test)):
    print(y_test[i],preds[i][0])
    label = y_test[i]
    plabel = round(preds[i][0],0)
    all+= 1
    if label == plabel:
        correct += 1
print(all)
print(correct)
print(correct/all)

model_json = model.to_json()
with open("../ModelData/model/DetectionModel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../ModelData/model/DetectionModel.h5")
print("model saved")
