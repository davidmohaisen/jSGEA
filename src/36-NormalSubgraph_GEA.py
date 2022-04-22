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
import keras
from keras.models import Sequential
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join
from networkx.algorithms import isomorphism
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











FamilyNames = ["Benign","Gafgyt","Mirai","Tsunami"]
patternsNames = ["Gafgyt","Mirai","Tsunami"]
patterns = []
for name in patternsNames:
    f = open("../FilteredOutputPatterns/"+name,"rb")
    pattern = pickle.load(f)
    pattern = pattern[:10000]
    patterns.append(pattern)

arr = []
for i in range(10):
    arr.append([])


load_base = "../ModelData/adversarial pickles/graphs_adv_GEA/toAdv/"
families = ["Gafgyt/","Mirai/","Tsunami/"]
sizes = ["Small/","Median/","Large/"]
cc = 0
features = [[[],[],[]],[[],[],[]],[[],[],[]]]
for family in families:
    for size in sizes:
        print(family,size)
        base = load_base+family+size
        files = [f for f in listdir(base) if isfile(join(base, f))]
        countFilesProcessed = 0
        counter = 0
        for file in files:
            print(cc)
            countFilesProcessed += 1
            nodes_density = []
            loc = base+file
            g = ""
            try:
                g = nx.drawing.nx_agraph.read_dot(loc)
                g = g.to_undirected()
                cc += 1
                counter = 0
                f = [0]*30000
                for i in range(len(patterns)):
                    for j in range(len(patterns[i])):
                        GM = isomorphism.GraphMatcher(g,patterns[i][j][1])
                        iso = GM.subgraph_is_isomorphic()
                        # print(counter,iso)
                        f[counter] = int(iso)
                        counter += 1
                l = families.index(family)
                k = sizes.index(size)
                features[l][k].append(f)

            except:
                print("Passed Sample")
                pass



f = open("../subgraph_Isomorphic/GEA.pkl","wb")
pickle.dump(features,f)
