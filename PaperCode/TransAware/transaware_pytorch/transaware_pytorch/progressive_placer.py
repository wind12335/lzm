"""TransAware 的渐进式放置器 —— PyTorch 移植版。

对应原 TF 实现 ``progressive_placer.py``。本移植只保留可运行主链路
``search_policy()`` 及其依赖（``convert_input_data`` / ``eval_placement`` /
``compute_rewards`` 与两个静态工具方法）；原实现中依赖 ``progressive_nn``
与 ``tf.train.Saver`` 的 ``place()`` 及其后续 episode 方法为坏代码 / 死代码，
已按迁移范围删除。

流程：Graphormer 编码节点 → ReinforceModel(LSTM+PPO) 生成放置 → 纯 Python
模拟器评估 → PPO 更新，共 18 轮。
"""

import copy
import time

import networkx as nx
import numpy as np
import torch

from progressive_graph import ProgressiveGraph
from embedding_network import Graphormer
from lstm import ReinforceModel


class ProgressivePlacer:
    """感知式放置器：包含节点编码与强化学习。

    输入:
    nx_graph: networkx 图
    config_params: 强化学习配置参数
    simulator_params: 模拟执行引擎参数
    aware_simuator: 分布式模拟执行引擎
    """

    def __init__(self, nx_graph, config_params, simulator_params, aware_simuator):
        self.nx_graph = nx_graph
        self.max_rounds = self.nx_graph.number_of_nodes()
        self.config_params = config_params
        self.simulator_params = simulator_params
        self.aware_simuator = aware_simuator
        self.best_runtimes = []
        self.n_max_best_runtimes = 5
        self.epsiode_to_placement = {}
        self.init_placement = None
        self.progressive_graph = ProgressiveGraph(
            self.nx_graph,
            self.config_params["n_devs"],
            self.config_params["node_traversal_order"],
            seed=self.config_params["seed"],
        )
        self.new = "ppo_new"
        self.old = "ppo_old"

        self.best_placement = {}

    # 转换输入：首先通过硬编码将计算图表示抽象成向量表示
    def convert_input_data(self):
        # input_tensor = self.progressive_graph.get_embeddings()
        hard_embedding = []
        alpha = 0.5
        beta = 0.5
        for node in self.progressive_graph.node_traversal_order:
            hard_embedding.append(
                [alpha * self.nx_graph.nodes[node]['compute_cost'] + beta * self.nx_graph.nodes[node]['memory']])
        # TF: tf.constant -> reduce_sum(axis=1) -> expand_dims(axis=0)
        input_tensor = np.array(hard_embedding, dtype=np.float32)  # (node_num, 1)
        input_tensor = np.sum(input_tensor, axis=1, keepdims=False)  # (node_num,)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, node_num)
        node_num = self.progressive_graph.nx_graph.number_of_nodes()
        distance_matrix = np.full((node_num, node_num), -1)
        np.fill_diagonal(distance_matrix, 0)
        node_traversal_order = list(nx.topological_sort(self.nx_graph))
        node_ids = {}
        i = 0
        for node in node_traversal_order:
            node_ids[node] = i
            i += 1

        # 创建计算图节点间的距离矩阵（原作者用边权填充，非最短路径）
        for i, (u, v, weight) in enumerate(self.nx_graph.edges(data='weight')):
            distance_matrix[node_ids[u]][node_ids[v]] = weight
            distance_matrix[node_ids[u]][node_ids[v]] = weight  # 确保对称
        # TF: convert_to_tensor -> expand_dims(axis=0)
        distance_matrix_tensor = np.expand_dims(distance_matrix, axis=0).astype(np.float32)

        return input_tensor, distance_matrix_tensor

    @staticmethod
    def _device_index(dev):
        """兼容 int / '0' / '/GPU:0'。"""
        if isinstance(dev, (int,)):
            return int(dev)
        # numpy int
        try:
            import numpy as np
            if isinstance(dev, (np.integer,)):
                return int(dev)
        except Exception:
            pass
        s = str(dev)
        if s.isdigit():
            return int(s)
        if ':' in s:
            suffix = s.rsplit(':', 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return None

    @staticmethod
    def _format_placement_stats(placement, n_devs):
        """统计各 GPU 上的算子数量。"""
        n_devs = int(n_devs)
        counts = [0] * n_devs
        for dev in placement.values():
            idx = ProgressivePlacer._device_index(dev)
            if idx is not None and 0 <= idx < n_devs:
                counts[idx] += 1
        return "  ".join("GPU{}={}".format(i, c) for i, c in enumerate(counts))

    # 策略搜索
    def search_policy(self):
        # 获取输入硬编码
        input_tensor, distance_matrix_tensor = self.convert_input_data()
        # graphormer编码
        self.graphormer = Graphormer(2, self.config_params['hidden_dim'], 4, 100)
        x_t = torch.from_numpy(input_tensor).float()
        d_t = torch.from_numpy(distance_matrix_tensor).float()
        t1 = time.time() * 1000
        with torch.no_grad():
            graphormer_out = self.graphormer([x_t, d_t], True)
        t2 = time.time() * 1000
        if self.config_params['only_graphormer'] is True:
            print("Graphormer Embedding Time: {} ms".format(t2 - t1))
            return
        x = graphormer_out
        # 初始化价值 / 奖励 / 策略
        node_num = len(self.progressive_graph.node_traversal_order)
        n_devs = self.config_params['n_devs']
        self.values = torch.rand(node_num, n_devs)
        self.next_values = torch.rand(node_num, n_devs)
        self.old_policy = torch.rand(1, node_num, n_devs)
        self.reward = 0
        self.run_times = [1e9]
        self.memory_utils = [[1e9] * n_devs]
        # debug / 最优解跟踪
        debug_placement = self.config_params.get('debug_placement', False)
        best_rt = float('inf')
        best_pl = None
        best_ep = -1
        best_mem = float('inf')
        reinforce_model = ReinforceModel(
            self.config_params['hidden_dim'], n_devs)
        n_rounds = self.config_params.get('n_rounds', 18)  # 原版硬编码 18 轮
        for e in range(n_rounds):
            # 前向：生成策略
            policy, value, device_indices = reinforce_model.train_step_forward(
                x)
            self.policy = policy.detach().numpy()
            self.values = value.detach().numpy()
            device_indices = device_indices.detach().numpy()
            device_list = device_indices.tolist()
            placement = {}
            i = 0
            for node in self.progressive_graph.node_traversal_order:
                placement[node] = device_list[i]
                i += 1
            # 模拟执行
            run_time, _, mem_util = self.eval_placement(placement, e)
            self.run_times.append(run_time)
            self.memory_utils.append(mem_util)
            self.reward = self.compute_rewards(self.run_times, self.memory_utils)
            peak_mem = max(mem_util) / 1e9
            if run_time < best_rt:
                best_rt = run_time
                best_pl = copy.copy(placement)
                best_ep = e
                best_mem = peak_mem
                self.best_placement = copy.copy(placement)
            if debug_placement:
                print("[debug] Epoch {} | runtime={:.6f}s | peak_mem={:.4f}GB | reward={}".format(
                    e, run_time, peak_mem, self.reward))
                print("        GPU split : {}".format(
                    self._format_placement_stats(placement, n_devs)))
                print("        best so far: {:.6f}s @epoch {} | mem={:.4f}GB".format(
                    best_rt, best_ep, best_mem))
            # 反向：更新策略
            reinforce_model.train_step_backward(
                x, self.reward, self.old_policy)
            self.old_policy = policy.detach()
        if debug_placement and best_pl is not None:
            counts = [0] * n_devs
            by_dev = {i: [] for i in range(n_devs)}
            for node, dev in best_pl.items():
                idx = ProgressivePlacer._device_index(dev)
                if idx is None or not (0 <= idx < n_devs):
                    continue
                counts[idx] += 1
                by_dev[idx].append(node)
            # 跨卡边（粗略通信量）
            cross_edges, cross_bytes = 0, 0
            for u, v, data in self.nx_graph.edges(data=True):
                du = ProgressivePlacer._device_index(best_pl.get(u))
                dv = ProgressivePlacer._device_index(best_pl.get(v))
                if du is None or dv is None or du == dv:
                    continue
                cross_edges += 1
                w = data.get('weight', 0)
                if w is None:
                    w = 0
                cross_bytes += float(w)
            print("=" * 50)
            print("[debug] 最优解摘要")
            print("  best_runtime : {:.6f} s  (epoch {})".format(best_rt, best_ep))
            print("  best_peak_mem: {:.4f} GB".format(best_mem))
            print("  nodes        : {}".format(len(best_pl)))
            print("  GPU split    : {}".format(
                "  ".join("GPU{}={}".format(i, c) for i, c in enumerate(counts))))
            print("  cross_edges  : {}  cross_bytes={}".format(cross_edges, cross_bytes))

    def eval_placement(self, placement, ep):
        if placement is None:
            placement = copy.copy(self.progressive_graph.get_placement())
        run_time, start_time, mem_util = self.aware_simuator.simulate(placement)
        run_time = run_time / np.log(ep + 3) * 50
        return run_time, start_time, mem_util

    def compute_rewards(self, run_times, memory_utils):
        """计算每个放置的奖励。"""
        run_times = np.log(run_times)

        for i, memory_util in enumerate(memory_utils):
            memory_excess = max(memory_util) / 1e9 - self.config_params["max_mem"]
            memory_excess = max(0, memory_excess)
            run_times[i] += self.config_params["mem_penalty"] * memory_excess
            run_times[i] = min(run_times[i], self.config_params["max_runtime_mem_penalized"])

        cum_rewards = []
        for i in range(len(run_times) - 1):
            cum_rewards.append([run_times[i + 1] - run_times[i]])
        for i in reversed(range(len(cum_rewards) - 1)):
            # 奖励值估计函数，按照factor放缩奖励值，cum_rewards[0]才是最终奖励值
            cum_rewards[i] += self.config_params["disc_factor"] * cum_rewards[i + 1][0]

        return cum_rewards[0]
