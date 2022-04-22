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

#IoT malware features
FamilyNames = ["Gafgyt","Mirai","Tsunami"]
heatList = []
patterns_all = []
for name in FamilyNames:
    heatList.append([])
    print(name)
    f = open("../FilteredOutputPatterns/"+name,"rb")
    patterns = pickle.load(f)
    patterns = patterns[:10000]
    patterns_all.append(patterns)
    for i in range(len(patterns)):
        heatList[-1].append(netlsd.heat(patterns[i][1]))
    # distance = netlsd.compare(heatList[-1][0], heatList[-1][0])

# heatList_tuple = []
# for i in range(len(heatList)):
#     heatList_tuple.append(list(tuple(j) for j in heatList[i]))
# taken = []
# for i in range(len(heatList_tuple)):
#     for j in range(len(heatList_tuple[i])):
#         for k in range(len(heatList_tuple)):
#             if k == i:
#                 continue
#             if heatList_tuple[i][j] in heatList_tuple[k] and heatList_tuple[i][j] not in taken:
#                 taken.append(heatList_tuple[i][j])
#                 print("Found from: ",FamilyNames[i]," and: ",FamilyNames[k],"Size:",len(patterns_all[i][j][1]))
