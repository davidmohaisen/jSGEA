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

FamilyNames = ["Benign","Gafgyt","Mirai","Tsunami"]
patternsNames = ["Gafgyt","Mirai","Tsunami"]
patterns = []
heats = []
for name in patternsNames:
    heats.append([])
    f = open("../FilteredOutputPatterns/"+name,"rb")
    pattern = pickle.load(f)
    pattern = pattern[:10000]
    patterns.append(pattern)
    for i in range(len(pattern)):
        heats[-1].append(netlsd.heat(pattern[i][1]))

arr = []
for i in range(10):
    arr.append([])






################################ GEA #################################
families = ["Gafgyt/","Mirai/","Tsunami/"]
sizes = ["Small/","Median/","Large/"]
load_base = "../ModelData/adversarial pickles/graphs_adv_GEA/toAdv_output/"
for family in families:
    for size in sizes:
        base = load_base+family+size
        for file in os.listdir(base):
            print(file)
            f = open(base+file,"r")
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].replace("\n","")
            graphs = []
            G=nx.Graph()
            nodesCount = 0
            edgesCount = 0
            for i in range(len(lines)):
                if lines[i].count("v ")!= 0:
                    G.add_node(nodesCount)
                    nodesCount += 1
                elif lines[i].count("e ")!= 0:
                    edgesCount += 1
                    line = lines[i]
                    line = line.split(" ")
                    G.add_edge(int(line[1]),int(line[2]))
                if lines[i].count("t ")!= 0 and i != 0:
                    graphs.append(G)
                    G=nx.Graph()
                    nodesCount = 0
                    edgesCount = 0

            graphs.append(G)
            if len(list(graphs[0].nodes())) == 0:
                print("here")
                continue
            localHeat = []
            for i in range(len(graphs)):
                if len(list(graphs[i].nodes())) != 0 and len(list(graphs[i].edges())) != 0:
                    localHeat.append(netlsd.heat(graphs[i]))
            seq = [[0]*10000,[0]*10000,[0]*10000]
            for k in range(1001):
                if len(localHeat) > k:
                    for i in range(len(seq)):
                        for j in range(len(seq[i])):
                            distance = netlsd.compare(localHeat[k], heats[i][j])
                            if round(distance,3) == 0 :
                                seq[i][j] += 1
                if k%100 == 0 and k!= 0:
                    arr[int(k/100)-1].append([family,size,file,seq])
                    seq = seq[:]

for i in range(len(arr)):
    f = open("../ModelData/adversarial pickles/graphs_adv_GEA/sequences/"+str(100*(i+1)),"wb")
    pickle.dump(arr[i],f)





################################ SGEA ###############################
families = ["Gafgyt/","Mirai/","Tsunami/"]
load_base = "../ModelData/adversarial pickles/graphs_adv_SGEA/output/"

for family in families:
    base = load_base+family
    for file in os.listdir(base):
        print(file)
        f = open(base+file,"r")
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("\n","")
        graphs = []
        G=nx.Graph()
        nodesCount = 0
        edgesCount = 0
        for i in range(len(lines)):
            if lines[i].count("v ")!= 0:
                G.add_node(nodesCount)
                nodesCount += 1
            elif lines[i].count("e ")!= 0:
                edgesCount += 1
                line = lines[i]
                line = line.split(" ")
                G.add_edge(int(line[1]),int(line[2]))
            if lines[i].count("t ")!= 0 and i != 0:
                graphs.append(G)
                G=nx.Graph()
                nodesCount = 0
                edgesCount = 0

        graphs.append(G)
        if len(list(graphs[0].nodes())) == 0:
            print("here")
            continue
        localHeat = []
        for i in range(len(graphs)):
            if len(list(graphs[i].nodes())) != 0 and len(list(graphs[i].edges())) != 0:
                localHeat.append(netlsd.heat(graphs[i]))
        seq = [[0]*10000,[0]*10000,[0]*10000]
        for k in range(1001):
            if len(localHeat) > k:
                for i in range(len(seq)):
                    for j in range(len(seq[i])):
                        distance = netlsd.compare(localHeat[k], heats[i][j])
                        if round(distance,3) == 0 :
                            seq[i][j] += 1
            if k%100 == 0 and k!= 0:
                arr[int(k/100)-1].append([family,file,seq])
                seq = seq[:]


for i in range(len(arr)):
    f = open("../ModelData/adversarial pickles/graphs_adv_SGEA/sequences/"+str(100*(i+1)),"wb")
    pickle.dump(arr[i],f)
