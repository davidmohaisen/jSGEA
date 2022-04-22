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
from os import listdir
from os.path import isfile, join


base = "../Dataset/Benign/"
nodes = ['_negvdi2_s.o.dot', 'setopt.c.o.dot', 'zip.o.dot']

families = ["Gafgyt/","Mirai/","Tsunami/"]
sizes = ["Small/","Median/","Large/"]
save_base = "../Dataset/GEA/"

for i in range(len(nodes)):
    print(sizes[i])
    loc = base + nodes[i]
    g1 = ""
    try:
        g1 = nx.drawing.nx_agraph.read_dot(loc)
        g1 = nx.DiGraph(g1)
        baseRead = "/home/ahmed/Documents/Projects/IOT-CFG-ATTACK-Journal/Dataset/"
        for l1 in range(len(families)):
            print("___________________________________________")
            print(sizes[i],families[l1])
            directory = baseRead + families[l1]
            count = 0
            for files in os.listdir(directory):
                try:
                    loc = directory + files
                    g2 = nx.drawing.nx_agraph.read_dot(loc)
                    g2 = nx.DiGraph(g2)
                    g = nx.compose(g1,g2)
                    n1 = (g1.nodes())
                    n2 = (g2.nodes())
                    xs = []
                    n1 = np.asarray(n1)
                    n2 = np.asarray(n2)
                    nEntry_Label = int(n1[0], 16)
                    nEntry_Label -= 12
                    g.add_node(hex(nEntry_Label))
                    g.add_edge(hex(nEntry_Label),n1[0])
                    g.add_edge(hex(nEntry_Label),n2[0])
                    nExit_Label = int(n1[len(n1)-1], 16)
                    nExit_Label += 12
                    g.add_node(hex(nExit_Label))
                    g.add_edge(n1[len(n1)-1],hex(nExit_Label))
                    g.add_edge(n2[len(n2)-1],hex(nExit_Label))
                    write_dot(g, save_base+families[l1]+sizes[i]+str(count)+".dot")
                    count+= 1
                except:
                    pass
    except:
        print("Passed Sample")
        pass
