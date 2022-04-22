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

sizes = ["1000"]

samples = []
classes = []
for size in sizes:

    f = open("../PredictSeq/"+size,"rb")
    data = pickle.load(f)

    for sample in data:
        try:
            sampleClass = FamilyNames.index(sample[0])
            if sampleClass != 0:
                sampleClass = 1
            ind = uniqueNames[sampleClass].index(sample[1])
            for i in range(len(filesNames[sampleClass][ind])):
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
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation="softmax"))
# model.add(keras.layers.Dense(2, activation="sigmoid"))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

# Fit the model
model.fit(samples,classes, epochs=20, batch_size=32)

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
with open("../ModelData/adversarial pickles/model/Model_softmax.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../ModelData/adversarial pickles/model/Model_softmax.h5")
print("model saved")
