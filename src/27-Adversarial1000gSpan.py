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
from os import listdir
from os.path import isfile, join

################################ GEA #################################
families = ["Gafgyt/","Mirai/","Tsunami/"]
sizes = ["Small/","Median/","Large/"]
load_base = "../ModelData/adversarial pickles/graphs_adv_GEA/toAdv_input/"
save_base = "../ModelData/adversarial pickles/graphs_adv_GEA/toAdv_output/"


for family in families:
    for size in sizes:
        base = load_base+family+size
        files = [f for f in listdir(base) if isfile(join(base, f))]
        DonelistOfFiles = os.listdir(save_base+family+size)

        for file in files:
            print(file)
            if file in DonelistOfFiles:
                print("taken")
                continue
            cmd = "python -m gspan_mining -s 1 -l 5 -u 20 \""+load_base+family+size+file+"\" > \""+save_base+family+size+file+"\" &"

            # cmd = "python -m gspan_mining -s 1 -l 20 -u 20 \""+load_base+family+size+file+"\" > \""+save_base+family+size+file+"\" &"
            os.system(cmd)
            time.sleep(60)


################################ SGEA ###############################
families = ["Gafgyt/","Mirai/","Tsunami/"]
load_base = "../ModelData/adversarial pickles/graphs_adv_SGEA/input/"
save_base = "../ModelData/adversarial pickles/graphs_adv_SGEA/output/"

for family in families:
    base = load_base+family
    files = [f for f in listdir(base) if isfile(join(base, f))]
    DonelistOfFiles = os.listdir(save_base+family)
    for file in files:
        print(file)
        if file in DonelistOfFiles:
            print("taken")
            continue
        cmd = "python -m gspan_mining -s 1 -l 5 -u 20 \""+load_base+family+file+"\" > \""+save_base+family+file+"\" &"
        # cmd = "python -m gspan_mining -s 1 -l 20 -u 20 \""+load_base+family+file+"\" > \""+save_base+family+file+"\" &"
        os.system(cmd)
        time.sleep(20)
