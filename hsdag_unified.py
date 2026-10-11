#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[归档副本] 本文件是 /data/coding/HSDAG/hsdag_unified.py 的只读归档（随统一实验入库）。
运行请到原位置: cd /data/coding/HSDAG && python hsdag_unified.py（依赖同目录 model.py/layer.py 等）。

HSDAG → 统一 BERT-Huge 对比框架迁移适配器（方案 A：算法移植）

原版 (NeurIPS'24 HSDAG) 的策略栈原样保留:
  EmbeddingModel 式 one-hot 类型嵌入(+node_id) / GPN(Graph Parsing Network)
  / ParsingNet 个性化聚类 / PPO(REINFORCE 式更新, 单条折扣回报)
  / 全放单一设备 → 1e10 罚(原版语义)

对本机与统一口径的 4 处偏离(全部如实披露):
  D1 绕过 OpenVINO 运行时(其 GPU 插件仅支持 Intel GPU, 本机 8xTitanV 不可见):
     计算图直接来自统一 pkl(bert_huge_baseline.pkl, 真机 TitanV 成本烘焙)
  D2 同置分组改为单点组: 原版 merge_operations 在 35 节点纯链上级联合并成 1 组,
     策略退化为单动作(必触发 1e10); 单点组让 GNN 看到全部 35 节点
  D3 num_devices 2(CPU+1GPU) → 8(GPU)
  D4 sim_runtime(OpenVINO HETERO 真执行) → 统一 HDP 族事件驱动模拟器
     (与 TransAware/Trinity 同一评估后端, 消费同一份烘焙成本)——
     三个 op 级基线因此共享同一 reward, 差异只在搜索算法本身

运行(galvatron env, CPU 即可):
  /data/miniconda/envs/galvatron/bin/python hsdag_unified.py --calib   # 验收校准
  /data/miniconda/envs/galvatron/bin/python hsdag_unified.py           # 正式搜索
输出: HSDAG_output/{best_result_unified.json, training_records_unified.json}
"""
import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor

HSDAG_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = '/data/coding/lzm/PaperCode/TransAware/transaware_pytorch/transaware_pytorch'
for _p in (HSDAG_DIR, SIM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import GPN                    # noqa: E402  (原版策略网络)
from simulator import AwareSimulator     # noqa: E402  (统一评估后端)

PKL = '/data/coding/lzm/PaperCode/TransAware/transaware/datasets/BERT_Huge/bert_huge_baseline.pkl'
UNITS = ['embeddings'] + ['layer_%d' % i for i in range(32)] + ['pooler', 'cls']
TYPE_OF = lambda n: {'embeddings': 'Embedding', 'pooler': 'Pooler',
                     'cls': 'ClsHead'}.get(n, 'BertLayer')
OUT_DIR = os.path.join(HSDAG_DIR, 'HSDAG_output')


class StubUnpickler(pickle.Unpickler):
    """屏蔽 pkl 内嵌 tensorflow 对象(与 transaware strategy_search 同款)。"""

    def find_class(self, module, name):
        if 'tensorflow' in module or module.startswith('tf'):
            return type('Stub', (), {})
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type('Stub', (), {})


def load_unified_graph(n_devs):
    """统一 pkl → (HSDAG 图 'Group_i'+x 特征, 模拟器图, 单点分组映射)."""
    with open(PKL, 'rb') as f:
        data = StubUnpickler(f).load()
    nxg = data['nx_graph']                       # compute_cost/memory/device, 边 tensor num_bytes

    # D2: 单点同置组(原版 merge_operations 在纯链上退化为 1 组)
    groups = [[n] for n in UNITS]
    group_map = {n: 'Group_%d' % i for i, n in enumerate(UNITS)}
    group_op_map = {g: [n] for g, n in zip(group_map.values(), UNITS)}

    # D1: one-hot 类型嵌入 + node_id(复刻 EmbeddingModel.pre_process 的输出构造)
    unique_types = sorted({TYPE_OF(n) for n in UNITS})
    emb_size = len(unique_types)
    x = torch.zeros(len(UNITS), emb_size + 1)
    for i, n in enumerate(UNITS):
        x[i, unique_types.index(TYPE_OF(n))] = 1.0
        x[i, emb_size] = i                       # 原版: cat(op_embedding, node_id)

    new_G = nx.DiGraph()
    for i, n in enumerate(UNITS):
        new_G.add_node('Group_%d' % i, x=x[i].clone())
    for u, v in zip(UNITS[:-1], UNITS[1:]):
        new_G.add_edge(group_map[u], group_map[v])

    # D4: 统一评估后端可用的图(边权=激活字节数)
    sim_g = nx.DiGraph()
    for n in UNITS:
        sim_g.add_node(n, compute_cost=nxg.nodes[n]['compute_cost'],
                       memory=nxg.nodes[n]['memory'], device='/GPU:0')
    for u, v in nxg.edges():
        sim_g.add_edge(u, v, weight=nxg[u][v]['tensor'][0]['num_bytes'])

    sim = AwareSimulator({'n_devs': n_devs}, sim_g)
    return new_G, group_map, group_op_map, sim, x, groups


class PPO(nn.Module):
    """原版 main.PPO 逐行照抄(仅去掉 OpenVINO 相关的 sim/get_runtime)。"""

    def __init__(self, gnn_params, action_dim):
        super(PPO, self).__init__()
        self.gpnn = GPN(gnn_params)
        self.classifier = nn.Sequential(
            nn.Linear(gnn_params['output_channel'], gnn_params['hidden_channel']),
            nn.ReLU(),
            nn.Linear(gnn_params['hidden_channel'], action_dim),
            nn.Softmax(dim=-1)
        )

    def act(self, state):
        clusters, node_ids = self.gpnn(state)
        action_probs = self.classifier(clusters)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action, action_logprob, dist_entropy, clusters, node_ids


def eval_runtime(sim, placement_int, n_devs):
    """D4+原版罚: 全放单一设备 → 1e10(原版语义); 否则统一模拟器实测估计(秒)."""
    if len(set(placement_int.values())) == 1:
        return 1e10
    pl = dict(placement_int)
    run_time, _, _ = sim.simulate(pl)
    return run_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_devs', type=int, default=8)
    ap.add_argument('--max_episodes', type=int, default=25)   # 原版默认
    ap.add_argument('--update_timestep', type=int, default=20)  # 原版默认
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--calib', action='store_true',
                    help='验收校准: 全放 GPU0, 统一模拟器应输出 ~1.81s')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    new_G, group_map, group_op_map, sim, x, groups = load_unified_graph(args.n_devs)

    # ---- 验收校准(统一口径契约 §2.3) ----
    # 直接打统一模拟器(绕过 1e10 罚): 全放 GPU0 必须输出 ~1.81s
    rt0, _, _ = sim.simulate({n: 0 for n in UNITS})
    rt_spread, _, _ = sim.simulate({n: i % args.n_devs for i, n in enumerate(UNITS)})
    pen = eval_runtime(sim, {n: 3 for n in UNITS}, args.n_devs)
    print('[calib] all-on-GPU0 -> %.6f s (期望 ~1.81) | 交错放置 -> %.6f s | '
          '全放单设备罚 -> %g' % (rt0, rt_spread, pen))
    if not (1.5 < rt0 < 2.2):
        print('[calib] FAIL: 成本链路未对齐, 终止'); sys.exit(1)
    if args.calib:
        print('[calib] OK'); return

    # ---- 原版 Config(仅改 in_channel/num_devices/max_episodes) ----
    cfg = {
        'num_devices': args.n_devs, 'in_channel': x.shape[1],
        'output_channel': x.shape[1], 'hidden_channel': 128,
        'layer_trans': 2, 'layer_gnn1': 2, 'layer_gnn2': 1, 'layer_deepsets': 1,
        'layer_parsingnet': 2, 'gnn_model': 'GCN', 'dropout_network': 0.2,
        'dropout_parsing': 0.0, 'link_ignore_self_loop': True, 'act_final': True,
        'learning_rate': 0.001, 'update_timestep': args.update_timestep,
        'eps_clip': 0.2, 'gamma': 0.99, 'discount_rate': 0.95,
        'device': 'cpu',
    }
    device = torch.device(cfg['device'])
    model = PPO(cfg, cfg['num_devices']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    max_eps = args.max_episodes
    update_timestep = cfg['update_timestep']

    # ---- rollout buffer(原版) ----
    class Buffer:
        def __init__(self):
            self.actions = []; self.states = []; self.logprobs = []
            self.entropys = []; self.rewards = []; self.clusters = []; self.node_ids = []
        def clear(self):
            for a in (self.actions, self.states, self.logprobs, self.entropys,
                      self.rewards, self.clusters, self.node_ids):
                del a[:]
    buffer = Buffer()

    best_runtime, best_placement = float('inf'), None
    training_records = []
    node_to_index = {node: i for i, node in enumerate(new_G.nodes())}
    initial_graph = new_G.copy()

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    print('[HSDAG-unified] 搜索开始: %d episodes x %d rollouts, n_devs=%d'
          % (max_eps, update_timestep, args.n_devs))

    state = new_G
    for eps in range(max_eps):
        state = initial_graph.copy()
        ep_reward, ep_best_runtime, ep_best_placement = 0, float('inf'), None

        for t in range(update_timestep):
            node_features = torch.stack(
                [state.nodes[n]['x'] for n in state.nodes()]).to(device)
            edge_index = torch.tensor(
                [(node_to_index[u], node_to_index[v]) for u, v in state.edges()],
                dtype=torch.long).t().contiguous().to(device)
            batch = torch.zeros(len(state.nodes()), dtype=torch.long).to(device)
            if edge_index.size(1) == 0:
                adj_t = SparseTensor(row=torch.tensor([]), col=torch.tensor([]),
                                     sparse_sizes=(len(state.nodes()), len(state.nodes()))).to(device)
            else:
                dense_adj = to_dense_adj(edge_index).squeeze(0)
                adj_t = SparseTensor.from_dense(dense_adj).to(device)
            state_data = Data(x=node_features, edge_index=edge_index,
                              batch=batch, adj_t=adj_t)

            actions, action_logprobs, dist_entropys, clusters, node_ids = model.act(state_data)
            actions_clone = actions.clone().detach()
            clusters_clone = clusters.clone().detach()
            actions_np = actions_clone.cpu().numpy()
            clusters_np = clusters_clone.cpu().numpy()
            node_ids_np = [list(nid) for nid in node_ids]   # GPN 返回 list[list[int]]

            # placement: 组动作展开到节点(原版 get_device_placement)
            device_placement = {}
            for i in range(len(clusters_np)):
                cluster_device = actions_np[i]
                for node_id in node_ids_np[i]:
                    device_placement[int(node_id)] = cluster_device
            # 组序号 → 统一单元名
            placement_units = {UNITS[nid]: int(dev) for nid, dev in device_placement.items()}

            runtime = eval_runtime(sim, placement_units, args.n_devs)
            reward = 1 / runtime                      # 原版 reward
            ep_reward += reward
            if runtime < ep_best_runtime:
                ep_best_runtime = runtime
                ep_best_placement = placement_units

            buffer.actions.append(actions)
            buffer.states.append(state_data)
            buffer.logprobs.append(action_logprobs)
            buffer.entropys.append(dist_entropys)
            buffer.rewards.append(reward)
            buffer.clusters.append(clusters)
            buffer.node_ids.append(node_ids)

            # 原版: 个性化反馈(节点特征 += 所属簇编号)
            flat_cluster_assignments = []
            node_to_group = {}
            for group_idx, group in enumerate(node_ids_np):
                for node in group:
                    node_to_group[int(node)] = group_idx
            sorted_nodes = sorted(node_to_group.keys())
            flat_cluster_assignments = [node_to_group[node] for node in sorted_nodes]
            cluster_assignments_tensor = torch.tensor(flat_cluster_assignments)
            final_tensor = clusters_clone[cluster_assignments_tensor]
            node_features = node_features + final_tensor

            for i, node in enumerate(state.nodes()):
                state.nodes[node]['x'] = node_features[i]

        # ---- 原版更新(REINFORCE, 单条折扣回报) ----
        cumulative_rewards = 0.0
        logprobs = torch.cat(buffer.logprobs)
        for i, value in enumerate(buffer.rewards[::-1]):
            cumulative_rewards += value * (cfg['discount_rate'] ** i)
        policy_loss = (-logprobs.squeeze() * (cumulative_rewards)).sum()
        optimizer.zero_grad()
        policy_loss.mean().backward()
        optimizer.step()
        buffer.clear()

        print('Episode %d: Average Reward = %.6g, Best Runtime = %.6g'
              % (eps, ep_reward / update_timestep, ep_best_runtime))
        if ep_best_runtime < best_runtime:
            best_runtime = ep_best_runtime
            best_placement = ep_best_placement
        training_records.append({'episode': eps,
                                 'avg_reward': ep_reward / update_timestep,
                                 'best_runtime': ep_best_runtime,
                                 'best_placement': ep_best_placement})

    search_time = time.time() - t_start
    print('Training completed. Best runtime: %g, search time: %.2f s' %
          (best_runtime, search_time))

    result = {'best_runtime': best_runtime, 'best_placement': best_placement,
              'search_time_s': round(search_time, 2),
              'n_devs': args.n_devs, 'max_episodes': max_eps,
              'update_timestep': update_timestep, 'seed': args.seed,
              'placement_order': UNITS,
              'deviations': ['D1 no-OpenVINO(unified pkl graph)',
                             'D2 singleton co-location groups(chain degeneracy fix)',
                             'D3 num_devices 2->8',
                             'D4 reward: unified HDP simulator(TitanV baked costs)']}
    with open(os.path.join(OUT_DIR, 'best_result_unified.json'), 'w') as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(OUT_DIR, 'training_records_unified.json'), 'w') as f:
        json.dump(training_records, f, indent=1)
    print('[dump] %s/best_result_unified.json' % OUT_DIR)


if __name__ == '__main__':
    main()
