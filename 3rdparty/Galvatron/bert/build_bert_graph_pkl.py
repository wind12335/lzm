# -*- coding: utf-8 -*-
"""构建 BERT-Huge 统一计算图 pkl（喂给 transAware / Trinity 两基线）。

图粒度 = placement_runner_bert.py 的执行单元（35 节点链）：
    embeddings -> layer_0 -> ... -> layer_31 -> pooler -> cls

所有数值来自本机 8x Titan V 的真机测量/推导，两个基线的模拟器直接消费：
  - compute_cost (us) : 按 FLOPs 比例分摊 naive 放置实测步时 1.8099s
                        (锚点: placement_runner 4L/GPU 实测, HDP 协议取中位)
  - memory (bytes)    : 混合精度 Adam 持久状态 16B/param
                        (fp16 param+grad 4B + fp32 master 4B + fp32 m,v 8B)
  - edge num_bytes    : bsz*seq*hidden*2B(fp16 激活)

pkl 键 (与 readfile.py / strategy_search.py 的读取逻辑对齐):
  nx_graph           : networkx DiGraph, 节点 {compute_cost, memory, device},
                       边 {'tensor': [{'num_bytes': X}]}     <- transAware 用
  Meta_graph_def     : TF MetaGraphDef, 35 个链式 op        <- Trinity 用
  step_stats         : StepStats, 设备名含 'all', 时长=真机 us <- Trinity cost_dict
  op_perf            : SimpleNamespace 列表 {node, op_memory.output_memory} <- Trinity out_d
  ungrouped_mapping  : 恒等映射

运行环境: tf env (需要 tensorflow 构造 MetaGraphDef; networkx 也在该环境)
  /data/miniconda/envs/tf/bin/python build_bert_graph_pkl.py
"""
import os
import sys
import json
import pickle
import types
import networkx as nx

# ---------------- 模型与锚点常量 ----------------
BSZ = 8
SEQ = 512
HID = 1280
VOCAB = 30522
NUM_LAYERS = 32
ANCHOR_TOTAL_MS = 1810.0   # naive 4L/GPU 放置实测步时 (HDP 协议, 2026-10-11)
EDGE_BYTES = BSZ * SEQ * HID * 2          # fp16 激活: 10,485,760 B
BYTES_PER_PARAM = 16                      # 混合精度 Adam 持久状态

# ---------------- 节点定义 ----------------
# FLOPs 等效层数 (12*H^2 MAC/token 为 1 个单位)
UNIT_LAYER = 12 * HID * HID
emb_flops = 0.1 * UNIT_LAYER                                  # gather+add+LN, 近似
layer_flops = UNIT_LAYER
pooler_flops = HID * HID                                      # h*h 一次
cls_flops = VOCAB * HID + HID * HID                           # MLM 解码 + transform

units = [('embeddings', emb_flops), ]
units += [('layer_%d' % i, layer_flops) for i in range(NUM_LAYERS)]
units += [('pooler', pooler_flops), ('cls', cls_flops)]

total_flops_units = sum(f for _, f in units)
per_unit_ms = ANCHOR_TOTAL_MS / total_flops_units

# 参数量 -> 持久显存
def params_of(name):
    if name == 'embeddings':
        return VOCAB * HID + SEQ * HID + 2 * HID + 2 * HID    # word+pos+tok+LN
    if name.startswith('layer_'):
        return 12 * HID * HID                                  # attn 4H^2 + mlp 8H^2
    if name == 'pooler':
        return HID * HID + HID
    if name == 'cls':                                          # transform + 解码器(解绑)
        return HID * HID + HID + VOCAB * HID + VOCAB
    raise KeyError(name)

node_costs = {n: int(round(f * per_unit_ms * 1000)) for n, f in units}   # us
node_mems = {n: int(params_of(n) * BYTES_PER_PARAM) for n, _ in units}   # bytes

# ---------------- 1) nx_graph (transAware) ----------------
G = nx.DiGraph()
for n, _ in units:
    G.add_node(n, compute_cost=node_costs[n], memory=node_mems[n], device='/GPU:0')
for (u, _), (v, _) in zip(units[:-1], units[1:]):
    G.add_edge(u, v, tensor=[{'num_bytes': EDGE_BYTES}])

# ---------------- 2) MetaGraphDef + step_stats + op_perf (Trinity) ----------------
import tensorflow.compat.v1 as tf1
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.framework import step_stats_pb2
tf1.disable_v2_behavior()

# 链式图: 用 5 种轮换单输入算子防止 grappler 的连续同类折叠;
# 节点成本来自 step_stats(真机), 因此算子类型只需"结构上能存活"。
UNARY_FNS = [tf1.nn.relu, tf1.tanh, tf1.sigmoid, tf1.nn.elu, tf1.nn.softsign]

g = tf1.Graph()
with g.as_default():
    prev = tf1.placeholder(tf1.float32, [BSZ, SEQ, HID], name='embeddings')
    for i in range(NUM_LAYERS):
        prev = UNARY_FNS[i % len(UNARY_FNS)](prev, name='layer_%d' % i)
    prev = tf1.sigmoid(prev, name='pooler')
    prev = tf1.tanh(prev, name='cls')
    tf1.add_to_collection('train_op', prev)
    tf1.add_to_collection('fetches', prev)

meta = tf1.train.export_meta_graph(graph=g)
for node in meta.graph_def.node:
    node.device = '/job:localhost/replica:0/task:0/device:GPU:0'

# step_stats: 设备名含 'all' 才会被 get_op_costs 采纳; 时长=真机 us
step_stats = step_stats_pb2.StepStats()
dev_stat = step_stats.dev_stats.add()
dev_stat.device = '/job:localhost/replica:0/task:0/device:GPU:0/all'
for n, _ in units:
    ns = dev_stat.node_stats.add()
    ns.node_name = n
    ns.all_start_micros = 0
    ns.op_start_rel_micros = 0
    ns.all_end_rel_micros = node_costs[n]        # cost = end - op_start

# op_perf: SimpleNamespace 可 pickle; out_d 取 op.op_memory.output_memory
out_bytes = {u: EDGE_BYTES for u, _ in units[:-1]}
out_bytes[units[-1][0]] = 0
op_perf = [types.SimpleNamespace(
    node=n, op_memory=types.SimpleNamespace(output_memory=[out_bytes[n]]))
    for n, _ in units]

payload = {
    'nx_graph': G,
    'Meta_graph_def': meta,
    'step_stats': step_stats,
    'op_perf': op_perf,
    'ungrouped_mapping': {n: n for n, _ in units},
}

# ---------------- 落盘 ----------------
OUT = '/data/coding/lzm/PaperCode/TransAware/transaware/datasets/BERT_Huge/bert_huge_baseline.pkl'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'wb') as f:
    pickle.dump(payload, f)

# Trinity 端用同一份 (其 datasets 目录独立)
TRI = '/data/coding/lzm/PaperCode/TransAware/Hierarchical/Trinity/datasets/BERT_Huge/bert_huge_baseline.pkl'
os.makedirs(os.path.dirname(TRI), exist_ok=True)
with open(TRI, 'wb') as f:
    pickle.dump(payload, f)

# 成本表 (文档用)
doc = {
    'anchor_total_ms': ANCHOR_TOTAL_MS, 'anchor': 'placement_runner naive 4L/GPU 实测 (HDP 协议)',
    'edge_bytes': EDGE_BYTES, 'bytes_per_param': BYTES_PER_PARAM,
    'nodes': {n: {'compute_cost_us': node_costs[n], 'memory_bytes': node_mems[n],
                  'memory_MB': round(node_mems[n] / 1048576, 1)} for n, _ in units},
    'totals': {'sum_cost_ms': round(sum(node_costs.values()) / 1000, 1),
               'sum_mem_GB': round(sum(node_mems.values()) / 1073741824, 2),
               'num_nodes': len(units), 'num_edges': len(units) - 1},
}
with open(os.path.join(os.path.dirname(OUT), 'bert_graph_costs.json'), 'w') as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)

print('written:', OUT)
print('written:', TRI)
print('nodes=%d edges=%d sum_cost=%.1fms sum_mem=%.2fGB' % (
    len(units), len(units) - 1, doc['totals']['sum_cost_ms'], doc['totals']['sum_mem_GB']))
print('sample:', json.dumps(doc['nodes']['layer_0']), json.dumps(doc['nodes']['cls']))
