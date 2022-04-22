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
from os import listdir
from os.path import isfile, join
#IoT malware features
FamilyNames = ["Benign","Gafgyt","Mirai","Tsunami"]
load_base = "../ModelData/adversarial pickles/graphs_adv_GEA/toAdv/"
families = ["Gafgyt/","Mirai/","Tsunami/"]
sizes = ["Small/","Median/","Large/"]
save_base = "../ModelData/adversarial pickles/graphs_adv_GEA/toAdv_input/"

for family in families:
    for size in sizes:
        print(family,size)
        base = load_base+family+size
        files = [f for f in listdir(base) if isfile(join(base, f))]
        countFilesProcessed = 0
        uniqueGraphs = []
        counter = 0
        for file in files:
            countFilesProcessed += 1
            nodes_density = []
            loc = base+file
            g = ""
            try:
                g = nx.drawing.nx_agraph.read_dot(loc)
                g = g.to_undirected()
                uniqueGraphs.append(g)
            except:
                print("Passed Sample")
                pass

        print(len(uniqueGraphs))
        for g in uniqueGraphs:
            nodes = list(nx.nodes(g))
            edges = list(nx.edges(g))

            #### Start from here ####
            strToAdd = "t # "+str(counter)+"\n"
            for i in range(len(nodes)):
                strToAdd += "v "+str(i)+" 1\n"
            for i in range(len(edges)):
                strToAdd += "e "+str(nodes.index(edges[i][0]))+" "+str(nodes.index(edges[i][1]))+" 1\n"
            csv_out = open(save_base+family+size+str(counter)+".txt","a")
            csv_out.write(strToAdd)
            csv_out.write("t # -1")
            csv_out.close()
            counter+=1
