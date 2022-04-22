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
# FamilyNames = ["Mirai","Tsunami"]
FamilyNames = ["Gafgyt","Mirai","Tsunami"]
for name in FamilyNames:
    graphsInput = open("../GraphsAsInput/gSpan_iso_"+name+".input","r")
    lines = graphsInput.readlines()
    count = 0
    listOfGraphs = []
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
            listOfGraphs.append(G)
            count+=1
            # write_dot(G,"./GraphsAsInput/Graphs/"+str(count)+".dot")
            G=nx.Graph()
            nodesCount = 0
            edgesCount = 0





    GraphBases = [[] for i in range(len(listOfGraphs))]


    counter = 0
    directory = "../Dataset/"+name+"/"
    AllString = []
    listOfFiles = len(os.listdir(directory))
    countFilesProcessed = 0
    for files in os.listdir(directory):
        countFilesProcessed += 1
        print(countFilesProcessed,"/",listOfFiles)
        if ((countFilesProcessed == 823 or countFilesProcessed == 833) and name == "Mirai") or (countFilesProcessed == 1186 and name == "Gafgyt"):
            continue
        # print(files)
        nodes_density = []
        loc = directory + files
        g = ""
        try:
            g = nx.drawing.nx_agraph.read_dot(loc)
            g = g.to_undirected()
        except:
            print("Passed Sample")
            pass
        if g!= "" :
            nodes = list(nx.nodes(g))
            if len(nodes) > 100:
                continue
            for i in range(len(listOfGraphs)):
                nodes_unique = list(nx.nodes(listOfGraphs[i]))
                if len(nodes) != len(nodes_unique) :
                    continue
                if nx.is_isomorphic(listOfGraphs[i], g):
                    GraphBases[i].append(files)
                    break
    f = open("../GraphsAsInput/origins/gSpan_iso_"+name+"_origins.pkl","wb")
    pickle.dump(GraphBases,f)
