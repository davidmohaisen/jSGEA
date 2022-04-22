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
# FamilyNames = ["Tsunami"]
for name in FamilyNames:
    print(name)
    f = open("../GraphsAsInput/origins/gSpan_iso_"+name+"_origins.pkl","rb")
    origins = pickle.load(f)
    graphs = []
    graphs_contrinuted = []
    toWheres = []
    count = 0
    for rank in range(4):
        # print(rank)
        graphsOutput = open("../GraphsAsOutput/gSpan_iso_"+name+"_"+str(rank+1)+".output","r")
        lines = graphsOutput.readlines()
        G=nx.Graph()
        nodesCount = 0
        edgesCount = 0
        toAddWhere = []
        for i in range(len(lines)):
            if len(graphs) >= 2000000:
                break
            if lines[i].count("v ")!= 0:
                G.add_node(nodesCount)
                nodesCount += 1
            elif lines[i].count("e ")!= 0:
                edgesCount += 1
                line = lines[i]
                line = line.split(" ")
                G.add_edge(int(line[1]),int(line[2]))
            if lines[i].count("where")!= 0:
                startIndex = lines[i].index("[")
                endIndex = lines[i].index("]")
                values = lines[i][startIndex+1:endIndex]
                values = values.split(", ")
                for value in values:
                    toAddWhere.append(int(value))
            if lines[i].count("t ")!= 0 and i != 0:
                graphs.append(G)
                toWheres.append(toAddWhere)
                toAddWhere = []
                G=nx.Graph()
                nodesCount = 0
                edgesCount = 0


    # print(len(graphs))
    # exit()

    taken = []
    all = [0]*len(origins)
    while True:
        if count >= 10000:
            break
        all_local = [0]*len(origins)
        for j in range(20,4,-1):
            for i in range(len(graphs)):
                if count >= 10000:
                    break
                # print("here")
                if len(graphs[i]) == j:
                    if graphs[i] not in taken:
                        flagContributed = False
                        for w in toWheres[i]:
                            if all_local[w] == 0:
                                flagContributed = True
                            all_local[w] = 1
                            all[w] += 1
                        if flagContributed:
                            taken.append(graphs[i])
                            graphs_contrinuted.append([count,graphs[i],toWheres[i]])
                            count+=1
                            print(name,"Graph Added",len(graphs[i].nodes),count)

    f = open("../FilteredOutputPatterns/"+name,"wb")
    print(len(all),sum(all))
    pickle.dump(graphs_contrinuted,f)
