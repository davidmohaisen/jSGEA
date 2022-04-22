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

f = open("../ModelData/adversarial pickles/UniqueSamples","rb")
graphs,Labels = pickle.load(f)
nodes = []
base = "../Dataset/Benign/"
# onlyfiles = [f for f in listdir(base) if isfile(join(base, f))]
# sizes = []
# for i in range(len(onlyfiles)):
#     loc = base + onlyfiles[i]
#     g = ""
#     try:
#         g = nx.drawing.nx_agraph.read_dot(loc)
#     except:
#         print("Passed Sample")
#         pass
#     if g!= "" :
#         node_cnt = len(list(nx.nodes(g)))
#         sizes.append(node_cnt)
#
# nodes.append(onlyfiles[sizes.index(10)])
# nodes.append(onlyfiles[sizes.index(23)])
# nodes.append(onlyfiles[sizes.index(1075)])
# print(nodes)

families = ["Gafgyt/","Mirai/","Tsunami/"]
sizes = ["Small/","Median/","Large/"]
save_base = "../ModelData/adversarial pickles/graphs_adv_GEA/original/"

nodes = ['_negvdi2_s.o.dot', 'setopt.c.o.dot', 'zip.o.dot']
for i in range(len(nodes)):
    loc = base + nodes[i]
    g1 = ""
    try:
        g1 = nx.drawing.nx_agraph.read_dot(loc)
    except:
        print("Passed Sample")
        pass
    if g1!= "" :
        for j in range(len(graphs)):
            g2 = graphs[j]
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
            write_dot(g, save_base+families[Labels[j]]+sizes[i]+str(j)+".dot")
