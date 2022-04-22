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




f = open("../PredictSeq/mapping","rb")
mapping = pickle.load(f)


base = "../Dataset/"
families = ["Benign","Gafgyt","Mirai","Tsunami"]

names = [[],[],[],[]]
features = [[],[],[],[]]
cc = 0
for l in range(len(mapping)):
    if l != 0:
        continue
    for m in range(len(mapping[l])):
        cc += 1
        if cc < 800 or cc >= 1600:
            continue
        print(cc, mapping[l][m][1][0])
        nodes_density = []
        loc = base + families[l] + "/" + mapping[l][m][1][0]
        g = ""
        try:
            g = nx.drawing.nx_agraph.read_dot(loc)
        except:
            print("Passed Sample")
            pass
        if g!= "" :
            g = g.to_undirected()
            counter = 0
            f = [0]*30000
            for i in range(len(patterns)):
                for j in range(len(patterns[i])):
                    GM = isomorphism.GraphMatcher(g,patterns[i][j][1])
                    iso = GM.subgraph_is_isomorphic()
                    # print(counter,iso)
                    f[counter] = int(iso)
                    counter += 1
            for i in range(len(mapping[l][m][1])):
                features[l].append(f)

f = open("../subgraph_Isomorphic/2.pkl","wb")
pickle.dump(features,f)
