import networkx as nx
# import matplotlib.pyplot as plt
import os
import sys
import pygraphviz
import numpy as np
from shutil import copyfile
import random
import pickle
import networkx.algorithms.isomorphism as iso
from networkx.drawing.nx_agraph import write_dot

#IoT malware features
FamilyNames = ["Gafgyt","Mirai","Tsunami"]
for name in FamilyNames:
    f = open("../GraphsAsInput/origins/gSpan_iso_"+name+"_origins.pkl","rb")
    data = pickle.load(f)
    arrFlags = [0]*len(data)
    for rank in range(4):
        graphsOutput = open("../GraphsAsOutput/gSpan_iso_"+name+"_"+str(rank+1)+".output","r")

        lines = graphsOutput.readlines()
        count = 0
        listOfGraphs = []
        G=nx.Graph()
        nodesCount = 0
        edgesCount = 0
        for i in range(len(lines)):
            if lines[i].count("where")!= 0:
                startIndex = lines[i].index("[")
                endIndex = lines[i].index("]")
                values = lines[i][startIndex+1:endIndex]
                values = values.split(", ")
                for value in values:
                    arrFlags[int(value)]+=1

    CountZero = 0
    CountSamples = 0
    Samples = 0
    for i in range(len(arrFlags)):
        Samples += len(data[i])
        if arrFlags[i] == 0:
            CountZero += 1
            CountSamples += len(data[i])
    print(name)
    print(CountSamples,"/",Samples)
    print(arrFlags)
