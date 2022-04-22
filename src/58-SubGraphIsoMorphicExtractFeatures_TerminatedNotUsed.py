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










patternsNames = ["Gafgyt","Mirai","Tsunami"]
patterns = []
for name in patternsNames:
    f = open("../FilteredOutputPatterns/"+name,"rb")
    pattern = pickle.load(f)
    pattern = pattern[:10000]
    patterns.append(pattern)


#IoT malware features
FamilyNames = ["Benign","Gafgyt","Mirai","Tsunami"]
base = "../Dataset/"

for l1 in range(len(FamilyNames)):
    print("___________________________________________")
    print(FamilyNames[l1])
    directory = base + FamilyNames[l1]+"/"
    x_all = []
    for files in os.listdir(directory):
        print(len(x_all))
        loc = directory + files
        AllString = []
        nodes_density = []
        g = nx.drawing.nx_agraph.read_dot(loc)
        g = nx.DiGraph(g)
        g = g.to_undirected()
        freq = [0]*30000
        counter = 0
        for i in range(len(patterns)):
            for j in range(len(patterns[i])):
                g2 = patterns[i][j][1].to_undirected()
                GM = isomorphism.GraphMatcher(g,g2)
                iso = GM.subgraph_is_isomorphic()
                freq[counter] = int(iso)
                counter += 1
        x_all.append(freq)
    f = open("../Pickles/SubgraphIso/"+FamilyNames[l1],"wb")
    pickle.dump(freq,f)
