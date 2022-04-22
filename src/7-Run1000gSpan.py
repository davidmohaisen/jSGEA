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

#IoT malware features
FamilyNames = ["Benign","Gafgyt","Mirai","Tsunami"]
# FamilyNames = ["Mirai"]
heatList = []
patterns_all = []
for name in FamilyNames:
    directory = "../GraphsAsInput/perSample/"+name+"/"
    doneDirectory = "../GraphsAsOutput/perSample/"+name+"/"
    DonelistOfFiles = os.listdir(doneDirectory)

    for files in os.listdir(directory):
        if files in DonelistOfFiles:
            print("taken")
            continue
        print(files)
        cmd = "python -m gspan_mining -s 1 -l 5 -u 20 ../GraphsAsInput/perSample/"+name+"/"+files+" > ../GraphsAsOutput/perSample/"+name+"/"+files+" &"
        # cmd = "python -m gspan_mining -s 1 -l 20 -u 20 ../GraphsAsInput/perSample/"+name+"/"+files+" > ../GraphsAsOutput/perSample/"+name+"/"+files+" &"
        os.system(cmd)
        time.sleep(30)
