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
from keras.models import model_from_json
import keras
f = open("../ModelData/adversarial pickles/UniqueSamples","rb")
graphs,Labels = pickle.load(f)
nodes = []
base = "../Dataset/Benign/"

#
# print(Labels.count(0))
# print(Labels.count(1))
# print(Labels.count(2))
#
# exit()
#
#
#



families = ["Gafgyt/","Mirai/","Tsunami/"]
save_base = "../ModelData/adversarial pickles/graphs_adv_SGEA/original/"


patterns = []
base_benign = "../ModelData/adversarial pickles/Benign_Patterns/"
onlyfiles = [f for f in listdir(base_benign) if isfile(join(base_benign, f))]
for i in range(len(onlyfiles)):
    loc = base_benign + onlyfiles[i]
    g = ""
    try:
        g = nx.drawing.nx_agraph.read_dot(loc)
        patterns.append(g)
    except:
        print("Passed Sample")


  #### Restore Model ####
json_file = open("../ModelData/model/DetectionModel.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../ModelData/model/DetectionModel.h5")
print("Loaded model from disk")
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])




for i in range(len(graphs)):
    g1 = graphs[i]
    g1 = nx.DiGraph(g1)
    for j in range(len(patterns)):

        g2 = patterns[j]
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



        ### Start from here ####
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
            continue
        sp_length = []
        for k in shortest_path:
            sp_length.append(shortest_path[k])
        shortestPathsArray = []
        for k in range(len(sp_length)):
            for x in sp_length[k] :
                if (len(sp_length[k][x])-1)==0 :
                    continue
                shortestPathsArray.append((len(sp_length[k][x])-1))

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

        sample_features = [[np.max(degree_list),np.min(degree_list),np.mean(degree_list),np.median(degree_list),np.std(degree_list),np.max(betweenness_list),np.min(betweenness_list),np.mean(betweenness_list),np.median(betweenness_list),np.std(betweenness_list),np.max(closeness_list),np.min(closeness_list),np.mean(closeness_list),np.median(closeness_list),np.std(closeness_list),maxShortestPath,minShortestPath,meanShortestPath,medianShortestPath,stdShortestPath,node_cnt,edge_cnt,density]]
        sample_features = np.asarray(sample_features)
        sample_features = sample_features.reshape((sample_features.shape[0],sample_features.shape[1],1))
        preds = model.predict(sample_features)
        # print(round(preds[0][0],0))
        if round(preds[0][0],0) == 0:
            write_dot(g, save_base+families[Labels[i]]+str(i)+".dot")
            break
