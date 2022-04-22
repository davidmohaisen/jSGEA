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
#IoT malware features
FamilyNames = ["Gafgyt","Mirai","Tsunami"]


for l in range(len(FamilyNames)):
    counter = 0
    directory = "./Dataset/"+FamilyNames[l]+"/"
    AllString = []
    uniqueGraphs = []
    listOfFiles = len(os.listdir(directory))
    countFilesProcessed = 0
    for files in os.listdir(directory):
        countFilesProcessed += 1
        print(countFilesProcessed,"/",listOfFiles)
        if (countFilesProcessed == 833 and l == 1) or (countFilesProcessed == 1120 and l == 1):
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
                    break
            if flagExists == False :
                uniqueGraphs.append(g)

    print(FamilyNames[l],len(uniqueGraphs))
    for g in uniqueGraphs:
        nodes = list(nx.nodes(g))
        edges = list(nx.edges(g))

        #### Start from here ####
        strToAdd = "t # "+str(counter)+"\n"
        for i in range(len(nodes)):
            strToAdd += "v "+str(i)+" 1\n"
        for i in range(len(edges)):
            strToAdd += "e "+str(nodes.index(edges[i][0]))+" "+str(nodes.index(edges[i][1]))+" 1\n"
        csv_out = open('./GraphsAsInput/gSpan_iso_'+FamilyNames[l]+'.input',"a")
        csv_out.write(strToAdd)
        csv_out.close()
        counter+=1

    csv_out = open('./GraphsAsInput/gSpan_iso_'+FamilyNames[l]+'.input',"a")
    csv_out.write("t # -1")
    csv_out.close()
