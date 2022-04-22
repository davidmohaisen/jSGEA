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




  #### Restore Model ####
json_file = open("../ModelData/model/DetectionModel.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../ModelData/model/DetectionModel.h5")
print("Loaded model from disk")
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])




families = ["Gafgyt/","Mirai/","Tsunami/"]
sizes = ["Small/","Median/","Large/"]
load_base = "../ModelData/adversarial pickles/graphs_adv_GEA/original/"

for family in families:
    for size in sizes:
        base = load_base+family+size
        files = [f for f in listdir(base) if isfile(join(base, f))]
        for file in files:
            g = nx.drawing.nx_agraph.read_dot(base+file)
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
                write_dot(g, "../ModelData/adversarial pickles/graphs_adv_GEA/toAdv/"+family+size+file)
            else:
                write_dot(g, "../ModelData/adversarial pickles/graphs_adv_GEA/toClassifier/"+family+size+file)
