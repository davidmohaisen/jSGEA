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

sizes = ["100","200","300","400","500","600","700","800","900","1000"]
# sizes = ["1000"]

for size in sizes:
    f = open("../PredictSeq/"+size+"_weights","rb")
    weights = pickle.load(f)

    correct = [0,0,0,0]
    all = [0,0,0,0]
    f = open("../PredictSeq/"+size,"rb")
    data = pickle.load(f)

    for sample in data:
        try:
            sampleClass = FamilyNames.index(sample[0])
            ind = uniqueNames[sampleClass].index(sample[1])
            all[sampleClass]+= len(filesNames[sampleClass][ind])

            classesScores_all = [[],[],[]]
            classesScores = [0,0,0]
            for i in range(len(sample[2])):
                for j in range(len(sample[2][i])):
                    if sample[2][i][j] != 0 :
                        classesScores_all[i].append(sample[2][i][j]*max(weights[i][j]))
            for i in range(len(classesScores_all)):
                if len(classesScores_all[i])!= 0:
                    classesScores[i] = max(classesScores_all[i])
            # if sampleClass == 3:
                # print(classesScores)
            predClass = -1
            if sum(classesScores)==0 :
                predClass = 0
            else:
                predClass = classesScores.index(max(classesScores))+1
            if predClass == sampleClass:
                correct[sampleClass]+= len(filesNames[sampleClass][ind])


        except:
            continue
    print(size)
    print(all)
    print(correct)
    # exit()
