import argparse
import pickle
import os
import networkx as nx
from graph import Graph
from merge import twolayersmerge

parser = argparse.ArgumentParser(description="merge")
parser.add_argument('--model_name', type=str, default="GNMT")
parser.add_argument('--threshold', type=int, default=2)

args = parser.parse_args()
config_params = vars(args)

name = ""
if config_params['model_name'] == "GNMT":
    name = "gnmt"
if config_params['model_name'] == "NasNet":
    name = "nasnet"
if config_params['model_name'] == "PnasNet":
    name = "pnasnet"
if config_params['model_name'] == "NMT_4layers":
    name = "nmt"
if config_params['model_name'] == "InceptionV3":
    name = "inceptionv3"

graph_path = "../../datasets/" + config_params['model_name'] + \
    "/" + name + "_" + str(config_params['threshold']) + ".pkl"
# graph_path = graph_path + os.listdir(graph_path)[0]

with open(graph_path, 'rb') as file:
    data = pickle.load(file)

nx_graph = data['nx_graph']
G = Graph()

for node in nx_graph.nodes():
    G.add_node(node, 
               nx_graph.nodes[node]['compute_cost'], 
               nx_graph.nodes[node]['memory'], 
               nx_graph.nodes[node]['device'])

edges = nx_graph.edges(data='tensor')
for edge in list(edges):
    G.add_edge(edge[0], edge[1], edge[2][0]['num_bytes'])

print("{} :".format(config_params['model_name']))
twolayersmerge(G, config_params['threshold'])