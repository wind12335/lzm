#coding=utf-8
"""TransAware 策略搜索入口 —— PyTorch 移植版。

与原 TF 版 ``strategy_search.py`` 对应。与原版的两点差异：
1. 用自定义 unpickler 屏蔽 pkl 中的 tensorflow 对象，无需安装 TF 即可加载；
2. 数据集路径默认指向原 TransAware 项目的 datasets 目录（可用 --data_path 覆盖）。
"""
import os
import sys
import argparse
import pickle
import networkx as nx

# 将包目录加入 sys.path，使包内的扁平 import 与原版保持一致
_PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transaware_pytorch')
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from graph import Graph
from interface import PyAwareInterface


class StubUnpickler(pickle.Unpickler):
    """屏蔽 tensorflow 类的自定义 unpickler。

    数据集 pkl 中除 ``nx_graph`` 外还包含 TF 的 ``Meta_graph_def`` / ``step_stats``
    对象；在无 TF 环境下用空 stub 类替换，使 ``nx_graph`` 可正常提取。
    """

    def find_class(self, module, name):
        if 'tensorflow' in module or module.startswith('tf'):
            return type('Stub', (), {})
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type('Stub', (), {})


# 定义强化学习策略搜索全局变量，导入所需要的参数
CONFIG_PARAMS_KEYS = ['seed', 'n_devs', 'radial_mp', 'sage_sample_ratio',
                      'pgnn_c', 'pgnn_neigh_cutoff', 'pgnn_anchor_exponent',
                      'pgnn_aggregation', 'n_eps', 'disc_factor', 'output_save_path',
                      'turn_based_baseline', 'print_freq', 'save_freq', 'eval_freq',
                      'bl_n_rnds', 'mem_penalty', 'max_mem', 'max_runtime_mem_penalized',
                      'node_traversal_order', 'model_name', 'max_num_outputs',
                      'max_output_size', 'hidden_dim', 'only_graphormer', 'threshold', 'debug_placement',
                      'count_model_states',
                      'n_rounds']

# 定义模拟执行引擎所需的全局变量
SIMULATOR_PARAMS_KEYS = ['lr_init', 'lr_dec', 'lr_start_decay_step', 'lr_decay_steps',
                         'lr_min', 'lr_dec_approach', 'ent_dec_init', 'ent_dec',
                         'ent_start_dec_step', 'ent_dec_steps', 'ent_dec_min',
                         'ent_dec_lin_steps', 'ent_dec_approach', 'optimizer_type']

# 默认数据集根目录：指向原 TransAware 项目的 datasets（不复制数据）
# 兼容两种目录布局：
#   1) 本仓库内：PaperCode/TransAware/transaware_pytorch/ -> 同级 ../transaware/datasets
#   2) 原独立目录：lzm/pytorch迁移/transaware_pytorch -> ../../测试hierarchical和transAware/.../datasets
def _find_datasets_dir():
    _here = os.path.dirname(os.path.abspath(__file__))
    _sibling = os.path.normpath(os.path.join(_here, '..', 'transaware', 'datasets'))
    if os.path.isdir(_sibling):
        return _sibling
    _legacy = os.path.normpath(os.path.join(
        _here, '..', '..', '测试hierarchical和transAware',
        'PaperCode', 'TransAware', 'transaware', 'datasets'))
    return _legacy if os.path.isdir(_legacy) else _sibling

DEFAULT_DATASETS_DIR = _find_datasets_dir()

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
parser.add_argument('--n_rounds', type=int, default=18,
                    help='策略搜索轮数（原版固定 18）')
parser.add_argument('--max_num_outputs', type=int, default=5)
parser.add_argument('--max_output_size', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--only_graphormer', type=bool, default=False)
parser.add_argument('--threshold', type=int, default=1)
parser.add_argument('--debug_placement', action='store_true',
                    help='打印每轮各 GPU 上的算子数量，用于检查 placement 是否变化')
parser.add_argument('--count_model_states', action='store_true',
                    help='[增强变体] 模拟器峰值显存计入节点持久状态(参数/优化器状态), '
                         '使 max_mem 约束对大模型可见; 默认关闭=原版行为')
parser.add_argument('--data_path', type=str, default=DEFAULT_DATASETS_DIR,
                    help='数据集根目录（默认指向原 TransAware 的 datasets）')

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

# 数据集路径：定位到 <data_path>/<model_name>/<pkl>
data_path = args.data_path

if config_params['model_name'] == "BERT_Huge":
    graph_path = os.path.join(data_path, "BERT_Huge", "bert_huge_baseline.pkl")
elif config_params['model_name'] == "NMT_4layers":
    graph_path = os.path.join(data_path, "NMT_4layers", "nmt_4layers_baseline.pkl")
elif config_params['model_name'] == "InceptionV3":
    graph_path = os.path.join(data_path, "InceptionV3", "inception_v3_baseline.pkl")
elif config_params['model_name'] == "GNMT":
    graph_path = os.path.join(data_path, "GNMT", "gnmt_baseline.pkl")
else:
    graph_path = os.path.join(
        data_path, config_params['model_name'],
        name + "_" + str(config_params['threshold']) + ".pkl")

print(f"正在读取数据集路径: {graph_path}")

# 读取计算图（用 stub unpickler 屏蔽 tensorflow 对象）
if not os.path.exists(graph_path):
    raise FileNotFoundError(
        "数据集不存在: {}。可用 --data_path 指定数据集根目录。".format(graph_path))
with open(graph_path, 'rb') as file:
    data = StubUnpickler(file).load()

# 将计算图从networkx转换为强化学习所需的Graph格式
nx_graph = data['nx_graph']
G = Graph()

node_traversal_order = []
for node in nx_graph.nodes():
    node_traversal_order.append(node)

for node in nx_graph.nodes():
    # 兼容不同的键名
    if 'compute_cost' in nx_graph.nodes[node]:
        c_cost = nx_graph.nodes[node]['compute_cost']
    else:
        c_cost = nx_graph.nodes[node]['cost']

    if 'memory' in nx_graph.nodes[node]:
        c_mem = nx_graph.nodes[node]['memory']
    else:
        c_mem = nx_graph.nodes[node]['mem']

    G.add_node(node,
               c_cost,
               c_mem,
               nx_graph.nodes[node]['device'])

edges = nx_graph.edges(data='tensor')
for edge in list(edges):
    G.add_edge(edge[0], edge[1], edge[2][0]['num_bytes'])

# transaware接口
aware_interface = PyAwareInterface(config_params, simulator_params, G)

# 自动搜索并行策略
import time
print("\n[TransAware] 开始策略搜索...")
start_timestamp = time.time()

aware_interface.startup_strategy()

end_timestamp = time.time()
total_time = end_timestamp - start_timestamp

print("\n" + "=" * 40)
print(f"【TransAware 策略搜索统计】")
print(f"总耗时 (Total Search Time): {total_time:.4f} 秒")
print("=" * 40 + "\n")

best_pl = aware_interface.get_best_placement()

# [统一模型实验 2026-10-11] 将最优 placement 落盘为 JSON (节点名 -> GPU 序号)
if best_pl:
    import json as _json
    _pl_out = {}
    for _k, _d in best_pl.items():
        _idx = int(str(_d).split(':')[-1]) if isinstance(_d, str) else int(_d)
        _pl_out[str(_k)] = _idx
    _out_dir = config_params.get('output_save_path', './TestAwareOutput')
    os.makedirs(_out_dir, exist_ok=True)
    _out_path = os.path.join(_out_dir, 'best_placement_%s_%ddevs.json'
                             % (config_params['model_name'], config_params['n_devs']))
    with open(_out_path, 'w') as _f:
        _json.dump(_pl_out, _f, indent=1)
    print('[dump] best placement -> %s' % _out_path)

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
