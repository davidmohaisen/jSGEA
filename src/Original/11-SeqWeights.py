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

for size in sizes:
    f = open("../PredictSeq/"+size,"rb")
    data = pickle.load(f)
    perPattern = []
    perPatternWeight = []
    for i in range(3):
        perPattern.append([])
        perPatternWeight.append([])
        for j in range(10000):
            perPattern[i].append([0,0,0,0])
            perPatternWeight[i].append([0,0,0,0])
    for i in range(len(data)):
        ind = FamilyNames.index(data[i][0])
        for j in range(len(data[i][2])):
            for k in range(len(data[i][2][j])):
                try:
                    id = FamilyNames.index(data[i][0])
                    unID = uniqueNames[id].index(data[i][1])
                    count = len(filesNames[id][unID])
                    perPattern[j][k][ind]+= data[i][2][j][k]*count
                except:
                    continue
    for i in range(len(perPattern)):
        for j in range(len(perPattern[i])):
            if sum(perPattern[i][j]) == 0:
                perPatternWeight[i][j][i+1] = 1
            else:
                perPatternWeight[i][j][i+1] = 1.0*perPattern[i][j][i+1]/sum(perPattern[i][j])
    f = open("../PredictSeq/"+size+"_weights","wb")
    pickle.dump(perPatternWeight,f)
