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
mapping = []
for l in range(len(FamilyNames)):
    mapping.append([])
    counter = 0
    directory = "../Dataset/"+FamilyNames[l]+"/"
    AllString = []
    uniqueGraphs = []
    whereGraphs = []
    listOfFiles = len(os.listdir(directory))
    countFilesProcessed = 0
    for files in os.listdir(directory):
        countFilesProcessed += 1
        print(countFilesProcessed,"/",listOfFiles)
        if (countFilesProcessed == 833 and l == 2) or (countFilesProcessed == 2073 and l == 0) or (countFilesProcessed == 496 and l == 0) or (countFilesProcessed == 1120 and l == 2):
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
            flagExists = False
            for i in range(len(uniqueGraphs)):
                nodes_unique = list(nx.nodes(uniqueGraphs[i]))
                if len(nodes) != len(nodes_unique) :
                    continue
                if nx.is_isomorphic(uniqueGraphs[i], g):
                    flagExists = True
                    whereGraphs[i].append(files)
                    break
            if flagExists == False :
                uniqueGraphs.append(g)
                whereGraphs.append([files])


    directory = "../GraphsAsInput/perSample/"+FamilyNames[l]+"/"
    graphs = []
    names = []
    for file in os.listdir(directory):
        f = open(directory+file,"r")
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("\n","")
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
        graphs.append(G)
        names.append(file)

    HeatGraphs = []
    for graph in graphs:
        HeatGraphs.append(netlsd.heat(graph))
    HeatUniqueGraphs = []
    for graph in uniqueGraphs:
        HeatUniqueGraphs.append(netlsd.heat(graph))
    for i in range(len(HeatGraphs)):
        print(i,len(HeatGraphs))
        for j in range(len(HeatUniqueGraphs)):
            distance = netlsd.compare(HeatGraphs[i], HeatUniqueGraphs[j])
            if round(distance,5) == 0 :
                mapping[l].append([names[i],whereGraphs[i]])
                break
f = open("../PredictSeq/mapping","wb")
pickle.dump(mapping,f)


# sizes = ["100","200","300","400","500","600","700","800","900","1000"]
# for size in sizes:
#     f = open("../PredictSeq/"+size,"rb")
#     data = pickle.load(f)
