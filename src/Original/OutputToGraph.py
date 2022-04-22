import networkx as nx
# import matplotlib.pyplot as plt
import os
import sys
import pygraphviz
import numpy as np
from shutil import copyfile
import random
import pickle
import pygraphviz
from networkx.drawing.nx_agraph import write_dot
family = ["Benign","Gafgyt","Mirai","Tsunami"]


f = open("./GraphsAsOutput/gSpan_iso.output","r")
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
        write_dot(G,"./GraphsAsInput/Graphs/"+str(count)+".dot")
        G=nx.Graph()
        nodesCount = 0
        edgesCount = 0
