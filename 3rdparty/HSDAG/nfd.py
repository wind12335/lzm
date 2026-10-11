# NFD
import networkx as nx
import numpy as np
import networkit as nk
import math
from collections import Counter
from scipy import stats
from tqdm import tqdm

def node_dimension(G,weight=True):
    nn = 0
    node_dimension = {}
    if weight == None:
        G_nk = nk.nxadapter.nx2nk(G)
    else:
        G_nk = nk.nxadapter.nx2nk(G,weightAttr='weight')
    for node in G.nodes():
        grow = []
        r_g = []
        num_g = []
        num_nodes = 0
        grow = nk.distance.Dijkstra(G_nk, int(node), storePaths=False).run().getDistances()
        grow.sort()
        if weight == True:
            grow = [math.ceil(d) for d in grow]
        grow = grow[1:]
        num = Counter(grow)
        for i,j in num.items():
            num_nodes += j
            if i>0:
                #if np.log(num_nodes) < 0.95*np.log(G.number_of_nodes()):
                r_g.append(i)
                num_g.append(num_nodes)
#                 # delete
#                 if np.log(num_nodes) > 0.9*np.log(G.number_of_nodes()):
#                     break
        x = np.log(r_g)
        y = np.log(num_g)

#         if len(r_g) < 1:
#             print("local",node)
        if len(r_g) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            node_dimension[node] = slope
        else:
            node_dimension[node] = 0
        nn += 1
    return node_dimension

#Fractal Dimension
