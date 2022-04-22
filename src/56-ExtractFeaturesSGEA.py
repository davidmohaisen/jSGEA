from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import time
import logging
import os
from numpy import argmax
import pickle
from six.moves import xrange
from sklearn.model_selection import train_test_split,KFold
import keras
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

#IoT malware features
FamilyNames = ["Gafgyt","Mirai","Tsunami"]
Type = ["Detection","Classification"]
Mod = ["RF","DNN","CNN"]
base = "../Dataset/SGEA/"
saveBase = "../Pickles/SGEA/"

for l3 in range(len(Type)):
    for l2 in range(len(Mod)):
        for l1 in range(len(FamilyNames)):
            print("___________________________________________")
            print(FamilyNames[l1])
            directory = base + Type[l3]+"/" + Mod[l2]+"/" + FamilyNames[l1]+"/"
            x_all = []
            for files in os.listdir(directory):
                print(len(x_all))
                loc = directory + files
                AllString = []
                nodes_density = []
                g = nx.drawing.nx_agraph.read_dot(loc)
                g = nx.DiGraph(g)

                #### Start from here ####
                node_cnt = len(list(nx.nodes(g)))
                edge_cnt = len(list(nx.edges(g)))
                avg_shortest_path = ""
                shortest_path = []
                closeness = []
                diameter = 0
                radius = 0
                current_flow_closeness = ""
                try:
                    avg_shortest_path = nx.average_shortest_path_length(g)
                    shortest_path = nx.shortest_path(g)
                    closeness = nx.algorithms.centrality.closeness_centrality(g)
                    shortest_betweenness = nx.algorithms.centrality.betweenness_centrality(g)
                    degree_centrality = nx.algorithms.centrality.degree_centrality(g)
                    density = nx.density(g)
                    sp_length = []
                    for i in shortest_path:
                        sp_length.append(shortest_path[i])
                    shortestPathsArray = []
                    for i in range(len(sp_length)):
                        for x in sp_length[i] :
                            if (len(sp_length[i][x])-1)==0 :
                                continue
                            shortestPathsArray.append((len(sp_length[i][x])-1))

                    maxShortestPath = np.max(shortestPathsArray)
                    minShortestPath = np.min(shortestPathsArray)
                    meanShortestPath = np.mean(shortestPathsArray)
                    medianShortestPath = np.median(shortestPathsArray)
                    stdShortestPath = np.std(shortestPathsArray)
                    closeness_list = list(closeness.values())
                    betweenness_list = list(shortest_betweenness.values())
                    degree_list = list(degree_centrality.values())
                    x_all.append([np.max(degree_list),np.min(degree_list),np.mean(degree_list),np.median(degree_list),np.std(degree_list),np.max(betweenness_list),np.min(betweenness_list),np.mean(betweenness_list),np.median(betweenness_list),np.std(betweenness_list) ,np.max(closeness_list),np.min(closeness_list),np.mean(closeness_list),np.median(closeness_list),np.std(closeness_list),maxShortestPath,minShortestPath,meanShortestPath,medianShortestPath,stdShortestPath,node_cnt,edge_cnt,density])

                except:
                    print("Unexpected error:", loc)

            x_all = np.asarray(x_all)
            print(x_all.shape)
            f = open(saveBase+Type[l3]+Mod[l2]+FamilyNames[l1],"wb")
            pickle.dump(x_all,f)
