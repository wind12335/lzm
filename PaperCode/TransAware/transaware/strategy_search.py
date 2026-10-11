#coding=utf-8
import os
import argparse
import pickle
import networkx as nx
from graph import Graph
from merge import twolayersmerge
from interface import PyAwareInterface

# 定义强化学习策略搜索全局变量，导入所需要的参数
CONFIG_PARAMS_KEYS = ['seed', 'n_devs', 'radial_mp', 'sage_sample_ratio', 
                      'pgnn_c', 'pgnn_neigh_cutoff', 'pgnn_anchor_exponent',
                      'pgnn_aggregation', 'n_eps', 'disc_factor', 'output_save_path',
                      'turn_based_baseline', 'print_freq', 'save_freq', 'eval_freq',
                      'bl_n_rnds', 'mem_penalty', 'max_mem', 'max_runtime_mem_penalized',
                      'node_traversal_order', 'model_name', 'max_num_outputs', 
                      'max_output_size', 'hidden_dim', 'only_graphormer', 'threshold', 'debug_placement']

# 定义模拟执行引擎所需的全局变量
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
parser.add_argument('--debug_placement', action='store_true',
                    help='打印每轮各 GPU 上的算子数量，用于检查 placement 是否变化')

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

# 根据CONFIG_PARAMS_KEYS和SIMULATOR_PARAMS_KEYS将强化学习参数和模拟器参数分开
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

# 导入数据集位置，例如需要导入GNMT计算图在经过阈值为32的双层节点融合处理后的数据，
# 其位置在../../datasets/GNMT/gnmt_32.pkl
# 该pkl文件中包含一个双层节点融合之后的抽象计算图表示，格式为networkx graph
#graph_path = "../../datasets/" + config_params['model_name'] + \
#    "/" + name + "_" + str(config_params['threshold']) + ".pkl"

# --- 修改代码逻辑 ---

if config_params['model_name'] == "NMT_4layers":
    graph_path = "./datasets/NMT_4layers/nmt_4layers_baseline.pkl"
elif config_params['model_name'] == "InceptionV3":
    graph_path = "./datasets/InceptionV3/inception_v3_baseline.pkl"
# [新增] 增加 GNMT 的判断分支
elif config_params['model_name'] == "GNMT":
    graph_path = "./datasets/GNMT/gnmt_baseline.pkl"
else:
    # 默认逻辑
    graph_path = "./datasets/" + config_params['model_name'] + \
        "/" + name + "_" + str(config_params['threshold']) + ".pkl"


print(f"正在读取数据集路径: {graph_path}") # 打印出来方便确认


# 读取计算图
with open(graph_path, 'rb') as file:
    data = pickle.load(file)

# 将计算图从networkx转换为强化学习所需的Graph格式
nx_graph = data['nx_graph']
G = Graph()

node_traversal_order = []
for node in nx_graph.nodes():
    node_traversal_order.append(node)

#for node in nx_graph.nodes():
#    G.add_node(node, 
#               nx_graph.nodes[node]['compute_cost'], 
#               nx_graph.nodes[node]['memory'], 
#               nx_graph.nodes[node]['device'])


for node in nx_graph.nodes():
    # --- 修改开始：兼容不同的键名 ---
    # 1. 获取计算开销 (兼容 'compute_cost' 和 'cost')
    if 'compute_cost' in nx_graph.nodes[node]:
        c_cost = nx_graph.nodes[node]['compute_cost']
    else:
        c_cost = nx_graph.nodes[node]['cost']
    
    # 2. 获取内存 (兼容 'memory' 和 'mem')
    if 'memory' in nx_graph.nodes[node]:
        c_mem = nx_graph.nodes[node]['memory']
    else:
        c_mem = nx_graph.nodes[node]['mem']
    # ---------------------------

    G.add_node(node, 
               c_cost,  # 使用获取到的变量
               c_mem,   # 使用获取到的变量
               nx_graph.nodes[node]['device'])



# for i in range(len(node_traversal_order) - 1):
#     G.add_edge(node_traversal_order[i], node_traversal_order[i + 1], 0)

edges = nx_graph.edges(data='tensor')
for edge in list(edges):
    G.add_edge(edge[0], edge[1], edge[2][0]['num_bytes'])

# 如果希望在这里再实现一遍双层节点融合，可执行以下注释代码
# 该函数对应TransAware方法的3.2小节 基于节点特性和阈值约束的双层节点融合技术
# twolayersmerge(G, 2)

# transaware接口
aware_interface = PyAwareInterface(config_params, simulator_params, G)

# 自动搜索并行策略
# aware_interface.comfirm_params()

# --- [新增] 开始计时 ---
import time
print("\n[TransAware] 开始策略搜索...")
start_timestamp = time.time()
# -----------------------

aware_interface.startup_strategy()

# --- [新增] 停止计时并输出 ---
end_timestamp = time.time()
total_time = end_timestamp - start_timestamp

print("\n" + "="*40)
print(f"【TransAware 策略搜索统计】")
print(f"总耗时 (Total Search Time): {total_time:.4f} 秒")
print("="*40 + "\n")

best_pl = aware_interface.get_best_placement()
if config_params.get('debug_placement', False):
    placer = aware_interface.progress_placer
    n_devs = config_params['n_devs']
    print("[debug] best_placement nodes: {}".format(len(best_pl) if best_pl else 0))
    if best_pl:
        counts = [0] * n_devs
        for d in best_pl.values():
            idx = int(str(d).split(':')[-1]) if isinstance(d, str) else int(d)
            if 0 <= idx < n_devs:
                counts[idx] += 1
        print("[debug] GPU split: {}".format(
            "  ".join("GPU{}={}".format(i, c) for i, c in enumerate(counts))))

