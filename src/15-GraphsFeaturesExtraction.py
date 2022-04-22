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
from shutil import copyfile


f = open("../PredictSeq/mapping","rb")
mapping = pickle.load(f)


base = "../Dataset/"
families = ["Benign","Gafgyt","Mirai","Tsunami"]

names = [[],[],[],[]]
features = [[],[],[],[]]

for l in range(len(mapping)):
    counter = 0
    for m in range(len(mapping[l])):
        for n in range(len(mapping[l][m][1])):
            print(mapping[l][m][1][n])
            nodes_density = []
            loc = base + families[l] + "/" + mapping[l][m][1][n]
            g = ""
            try:
                g = nx.drawing.nx_agraph.read_dot(loc)
            except:
                print("Passed Sample")
                pass
            if g!= "" :
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

                except:
                    print("Unexpected error:", loc)
                sp_length = []
                for i in shortest_path:
                    sp_length.append(shortest_path[i])
                shortestPathsArray = []
                for i in range(len(sp_length)):
                    for x in sp_length[i] :
                        if (len(sp_length[i][x])-1)==0 :
                            continue
                        shortestPathsArray.append((len(sp_length[i][x])-1))

                if (len(shortestPathsArray))== 0 :
                    counter += 1
                    continue
                maxShortestPath = np.max(shortestPathsArray)
                minShortestPath = np.min(shortestPathsArray)
                meanShortestPath = np.mean(shortestPathsArray)
                medianShortestPath = np.median(shortestPathsArray)
                stdShortestPath = np.std(shortestPathsArray)
                closeness_list = list(closeness.values())
                betweenness_list = list(shortest_betweenness.values())
                degree_list = list(degree_centrality.values())

                names[l].append(mapping[l][m][1][n])
                features[l].append([np.max(degree_list),np.min(degree_list),np.mean(degree_list),np.median(degree_list),np.std(degree_list),np.max(betweenness_list),np.min(betweenness_list),np.mean(betweenness_list),np.median(betweenness_list),np.std(betweenness_list),np.max(closeness_list),np.min(closeness_list),np.mean(closeness_list),np.median(closeness_list),np.std(closeness_list),maxShortestPath,minShortestPath,meanShortestPath,medianShortestPath,stdShortestPath,node_cnt,edge_cnt,density])

f = open("../ModelData/pickle/names","wb")
pickle.dump(names,f)
f = open("../ModelData/pickle/features","wb")
pickle.dump(features,f)
