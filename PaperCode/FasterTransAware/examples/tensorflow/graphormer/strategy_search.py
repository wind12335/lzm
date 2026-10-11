#coding=utf-8
import os
import argparse
import pickle
import networkx as nx
from graph import Graph
from merge import twolayersmerge
from interface import PyAwareInterface

CONFIG_PARAMS_KEYS = ['seed', 'n_devs', 'radial_mp', 'sage_sample_ratio', 
                      'pgnn_c', 'pgnn_neigh_cutoff', 'pgnn_anchor_exponent',
                      'pgnn_aggregation', 'n_eps', 'disc_factor', 'output_save_path',
                      'turn_based_baseline', 'print_freq', 'save_freq', 'eval_freq',
                      'bl_n_rnds', 'mem_penalty', 'max_mem', 'max_runtime_mem_penalized',
                      'node_traversal_order', 'model_name', 'max_num_outputs', 
                      'max_output_size', 'hidden_dim', 'only_graphormer', 'threshold']

SIMULATOR_PARAMS_KEYS = ['lr_init', 'lr_dec', 'lr_start_decay_step', 'lr_decay_steps',
                         'lr_min', 'lr_dec_approach', 'ent_dec_init', 'ent_dec',
                         'ent_start_dec_step', 'ent_dec_steps', 'ent_dec_min',
                         'ent_dec_lin_steps', 'ent_dec_approach', 'optimizer_type']

parser = argparse.ArgumentParser(description="reinforcement learning search")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n_devs', type=int, default=4)
parser.add_argument('--radial_mp', type=int, default=1)
parser.add_argument('--sage_sample_ratio', type=float, default=1.0)
parser.add_argument('--pgnn_c', type=float, default=0.2)
parser.add_argument('--pgnn_neigh_cutoff', type=int, default=4)
parser.add_argument('--pgnn_anchor_exponent', type=int, default=4)
parser.add_argument('--pgnn_aggregation', type=str, default="max")
parser.add_argument('--n_eps', type=int, default=40)
parser.add_argument('--disc_factor', type=float, default=1.0)
parser.add_argument('--output_save_path', type=str, default="./TestAwareOutput")
parser.add_argument('--turn_based_baseline', type=bool, default=True)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--eval_freq', type=int, default=5)
parser.add_argument('--bl_n_rnds', type=int, default=1000)
parser.add_argument('--mem_penalty', type=float, default=3.0)
parser.add_argument('--max_mem', type=int, default=10)
parser.add_argument('--max_runtime_mem_penalized', type=float, default=10.0)
parser.add_argument('--node_traversal_order', type=str, default="random")
parser.add_argument('--model_name', type=str, default="GNMT")
parser.add_argument('--max_num_outputs', type=int, default=5)
parser.add_argument('--max_output_size', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--only_graphormer', type=bool, default=False)
parser.add_argument('--threshold', type=int, default=1)

parser.add_argument('--lr_init', type=float, default=1e-3)
parser.add_argument('--lr_dec', type=float, default=0.95)
parser.add_argument('--lr_start_decay_step', type=float, default=1e9)
parser.add_argument('--lr_decay_steps', type=int, default=100)
parser.add_argument('--lr_min', type=float, default=1e-3)
parser.add_argument('--lr_dec_approach', type=str, default="exponential")
parser.add_argument('--ent_dec_init', type=float, default=1.0)
parser.add_argument('--ent_dec', type=float, default=0.95)
parser.add_argument('--ent_start_dec_step', type=float, default=1e9)
parser.add_argument('--ent_dec_steps', type=int, default=100)
parser.add_argument('--ent_dec_min', type=float, default=0.0)
parser.add_argument('--ent_dec_lin_steps', type=int, default=0)
parser.add_argument('--ent_dec_approach', type=str, default="linear")
parser.add_argument('--optimizer_type', type=str, default="adam")

# 解析参数
args = parser.parse_args()

config_params = {}
simulator_params = {}

for key, value in vars(args).items():
    if key in CONFIG_PARAMS_KEYS:
        config_params[key] = value
    if key in SIMULATOR_PARAMS_KEYS:
        simulator_params[key] = value

# graph_path = "../datasets/" + config_params['model_name'] + "/"
# graph_path = graph_path + os.listdir(graph_path)[0]

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

graph_path = "/work/home/acrlrlzu6q/wangwei/211050065/PaperCode/FasterTransformer/examples/tensorflow/graphormer/datasets/" + config_params['model_name'] + \
    "/" + name + "_" + str(config_params['threshold']) + ".pkl"

with open(graph_path, 'rb') as file:
    data = pickle.load(file)

nx_graph = data['nx_graph']
G = Graph()

node_traversal_order = []
for node in nx_graph.nodes():
    node_traversal_order.append(node)

for node in nx_graph.nodes():
    G.add_node(node, 
               nx_graph.nodes[node]['compute_cost'], 
               nx_graph.nodes[node]['memory'], 
               nx_graph.nodes[node]['device'])

for i in range(len(node_traversal_order) - 1):
    G.add_edge(node_traversal_order[i], node_traversal_order[i + 1], 0)

# edges = nx_graph.edges(data='tensor')
# for edge in list(edges):
#     G.add_edge(edge[0], edge[1], edge[2][0]['num_bytes'])

# twolayersmerge(G, 2)

aware_interface = PyAwareInterface(config_params, simulator_params, G)

# aware_interface.comfirm_params()
aware_interface.startup_strategy()
aware_interface.get_best_placement()

