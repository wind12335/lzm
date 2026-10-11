import copy
import time
import os
import networkx as nx
from collections import deque
from heapq import heappushpop


import tensorflow as tf
import numpy as np
from progressive_graph import ProgressiveGraph
from progressive_nn import MessagePassingProgressiveNN
from embedding_network import Graphormer
from LSTM import PPOStrategyGenerator, ppo_loss, compute_gae, td_update, ReinforceModel
import log
from tensorflow.python.training import training_util

__doc__ = "The placer of aware algorithm, including node embedding and reinforcement learning"


tf.disable_eager_execution()


class ProgressivePlacer:
    """
    The placer of aware algorithm, including node embedding and reinforcement learning
    Input:
    nx_graph: networkx graph
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
        # self.model = MessagePassingProgressiveNN(
        #     progressive_graph=self.progressive_graph,
        #     config_params=self.config_params,
        #     simulator_params=self.simulator_params,
        # )
        # self.model_saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2)
        self.new = "ppo_new"
        self.old = "ppo_old"

        self.best_placement = {}
    
    # 转换输入，该部分首先通过硬编码将计算图表示抽象成向量表示
    def convert_input_data(self):
        # input_tensor = self.progressive_graph.get_embeddings()
        hard_embedding = []
        alpha = 0.5
        beta = 0.5
        for node in self.progressive_graph.node_traversal_order:
            hard_embedding.append([alpha * self.nx_graph.nodes[node]['compute_cost'] + beta * self.nx_graph.nodes[node]['memory']])
        input_tensor = tf.constant(hard_embedding, dtype=tf.float32)
        input_tensor = tf.reduce_sum(input_tensor, axis=1, keepdims=False)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        node_num = self.progressive_graph.nx_graph.number_of_nodes()
        distance_matrix = np.full((node_num, node_num), -1)
        np.fill_diagonal(distance_matrix, 0)
        node_traversal_order = list(nx.topological_sort(self.nx_graph))
        node_ids = {}
        i = 0
        for node in node_traversal_order:
            node_ids[node] = i
            i += 1
        
        # 创建计算图节点间的邻接矩阵，用于表示节点间的连接状态和距离
        for i, (u, v, weight) in enumerate(self.nx_graph.edges(data='weight')):
            distance_matrix[node_ids[u]][node_ids[v]] = weight  # 索引调整，假设节点标签从1开始
            distance_matrix[node_ids[u]][node_ids[v]] = weight  # 确保矩阵是对称的，因为这是一个无向图
        distance_matrix_tensor = tf.convert_to_tensor(distance_matrix, dtype=tf.float32)
        distance_matrix_tensor = tf.expand_dims(distance_matrix_tensor, axis=0)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        # 出现兼容性问题，关闭jit加速
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF


        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            distance_mat = sess.run(distance_matrix_tensor)
            input_t = sess.run(input_tensor)
        
        return input_t, distance_mat

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
        graphormer_output = self.graphormer([input_tensor, distance_matrix_tensor], True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            t1 = time.time() * 1000
            graphormer_out = sess.run(graphormer_output)
            t2 = time.time() * 1000
        if self.config_params['only_graphormer'] is True:
            print("Graphormer Embedding Time: {} ms".format(t2 - t1))
            return
        x = graphormer_out
        # 初始化价值 / 奖励 / 策略
        node_num = len(self.progressive_graph.node_traversal_order)
        n_devs = self.config_params['n_devs']
        self.values = tf.random.uniform(shape=[node_num, n_devs])
        self.next_values = tf.random.uniform(shape=[node_num, n_devs])
        self.old_policy = tf.random.uniform(
            (node_num, n_devs), minval=0, maxval=1, dtype=tf.float32)
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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for e in range(18):
                # 前向：生成策略
                policy, value, device_indices = reinforce_model.train_step_forward(
                    input_data=x)
                sess.run(tf.global_variables_initializer())
                self.policy, self.values, device_indices = sess.run(
                    [policy, value, device_indices])
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
                    # 节点很多时会刷屏，需要完整映射时再打开下一行
                    # print("        placement:", placement)
                # 反向：更新策略
                train_op = reinforce_model.train_step_backward(
                    x, self.reward, self.old_policy)
                sess.run(tf.global_variables_initializer())
                sess.run(train_op)
                self.old_policy = self.policy
            if debug_placement and best_pl is not None:
                n_devs = self.config_params['n_devs']
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
               

    def place(self):
        """
        The body of aware placement (embedding and reinforcement learning)
        """
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True

        input_tensor = self.progressive_graph.get_embeddings()
        input_tensor = tf.reduce_sum(input_tensor, axis=1, keepdims=False)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        node_num = self.progressive_graph.nx_graph.number_of_nodes()
        distance_matrix = np.full((node_num, node_num), -1)
        np.fill_diagonal(distance_matrix, 0)
        node_traversal_order = list(nx.topological_sort(self.nx_graph))
        node_ids = {}
        i = 0
        for node in node_traversal_order:
            node_ids[node] = i
            i += 1
        
        for i, (u, v, weight) in enumerate(self.nx_graph.edges(data='weight')):
            distance_matrix[node_ids[u]][node_ids[v]] = weight  # 索引调整，假设节点标签从1开始
            distance_matrix[node_ids[u]][node_ids[v]] = weight  # 确保矩阵是对称的，因为这是一个无向图
        distance_matrix_tensor = tf.convert_to_tensor(distance_matrix, dtype=tf.float32)
        distance_matrix_tensor = tf.expand_dims(distance_matrix_tensor, axis=0)

        self.graphormer = Graphormer(2, 512, 4, 100)
        graphormer_output =  self.graphormer([input_tensor, distance_matrix_tensor], True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF


        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            graphormer_out = sess.run(graphormer_output)
        
        x = tf.squeeze(graphormer_out)

        num_groups = min(256, 20 * self.n_devs)


            
        lstm(x, x, x, 0.1, 0.1)
        

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # sess.run(self.model.init_global_step)
            rand_placement = self.progressive_graph.get_random_placement(seed=0)

            best_reward_runtime = 1e9
            best_placement = rand_placement
            best_runtime = 1e9
            for episode in range(self.config_params["n_eps"]):
                self.init_placement = rand_placement
                is_eval_episode = episode % self.config_params["eval_freq"] == 0
                is_save_episode = episode % self.config_params["save_freq"] == 0 and episode > 0

                (
                    _,
                    _,
                    _,
                    run_times,
                    memory_utils,
                    _,
                    _,
                    placements,
                ) = self.run_episode(sess, is_eval_episode)

                batched_cum_rewards = []
                reward_runtimes = []
                cum_rewards, reward_runtime = self.compute_rewards(run_times, memory_utils)
                batched_cum_rewards.append(cum_rewards)
                reward_runtimes.append(reward_runtime)

                (
                    ep_best_placement,
                    ep_best_reward_runtime,
                    ep_best_placement_runtime,
                    _,
                ) = self.identify_best_pl(reward_runtimes, run_times, memory_utils, placements)

                if ep_best_reward_runtime < best_reward_runtime and not is_eval_episode:
                    best_reward_runtime = ep_best_reward_runtime
                    best_placement = ep_best_placement
                    best_runtime = ep_best_placement

                self.update_best_pl(ep_best_reward_runtime, ep_best_placement, episode, is_eval_episode)
                log.debug("Episode best pl runtime %s", ep_best_placement_runtime)
                log.debug("Episode best pl (simplified) %s", [best_placement[node] for node in self.nx_graph.nodes()])

                self.best_placement = best_placement

                if is_save_episode:
                    self.save_model(sess, episode)

                batched_cum_rewards = np.float32(batched_cum_rewards)
                avg_cum_rewards = np.mean(batched_cum_rewards, axis=0)
                rnd_cum_rewards = [deque(maxlen=self.config_params["bl_n_rnds"]) for _ in range(self.max_rounds)]
                for i, avg_cum_reward in enumerate(avg_cum_rewards):
                    rnd_cum_rewards[i].append(avg_cum_reward)

                os.makedirs(os.path.join(self.config_params["output_save_path"], "summary"), exist_ok=True)
                with open(
                    self.config_params["output_save_path"] + "/summary/summary_{episode}.txt", "a+", encoding="utf-8"
                ) as f:
                    f.write("\nEpisode: " + str(episode))
                    f.write("\nep best runtime: " + str(ep_best_placement_runtime))
                    f.write("\nbest so far: " + str(best_runtime))
                    f.write("\nep best reward runtime: " + str(ep_best_reward_runtime))
                    f.write("\nbest reward runtime: " + str(best_reward_runtime))
                    f.write("\n------------\n")

    def run_episode(self, sess, is_eval_episode):
        """
        update the best results after every episode
        """
        self.progressive_graph.reset_placement(self.init_placement)  # 将所有的设备都设置成init_pl
        self.progressive_graph.new_episode()  # 将所有的节点和当前节点的flag标记为unseen

        start_time = np.array([[-1] * self.progressive_graph.nx_graph.number_of_nodes()])
        run_time, start_time, output_memory_peak_utils = self.eval_placement()
        self.progressive_graph.set_start_times(start_time)

        episode_best_time = run_time
        episode_best_placement_memory = output_memory_peak_utils
        episode_best_pl = self.init_placement
        run_times = []
        memory_utils = []
        states = []
        explor_acts = []
        placements = [self.init_placement]

        run_times.append(run_time)
        memory_utils.append(output_memory_peak_utils)

        nn_time = 0
        s1 = time.time()
        for i_round in range(self.max_rounds):
            nodes = self.progressive_graph.node_traversal_order
            node = nodes[i_round]
            self.progressive_graph.set_curr_node(node)

            s2 = time.time()
            device, lo, feed, expl, train_outs = self.get_improvement(sess, node, is_eval_episode)
            nn_time += time.time() - s2

            explor_acts.append(expl)
            self.progressive_graph.refresh(node, device[0])
            placements.append(self.progressive_graph.get_placement())

            run_time, start_time, output_memory_peak_utils = self.eval_placement()
            run_times.append(run_time)
            memory_utils.append(output_memory_peak_utils)
            states.append([feed, device, lo, train_outs])
            self.progressive_graph.update_start_time(start_time)
            self.progressive_graph.inc_done_node(node)

        for i, rnd_rt in enumerate(run_times):
            if episode_best_time > rnd_rt:
                episode_best_time = rnd_rt
                episode_best_placement_memory = memory_utils[i]
                episode_best_pl = placements[i]

        run_times = np.transpose(run_times)
        memory_utils = np.array(memory_utils)

        log.debug("Total time: %s   NN time: %s", time.time() - s1, nn_time)

        return (
            episode_best_pl,
            episode_best_time,
            episode_best_placement_memory,
            run_times,
            memory_utils,
            states,
            explor_acts,
            placements,
        )

    def eval_placement(self, placement, ep):
        if placement is None:
            placement = copy.copy(self.progressive_graph.get_placement())
        run_time, start_time, mem_util = self.aware_simuator.simulate(placement)
        run_time = run_time / np.log(ep + 3) * 50
        return run_time, start_time, mem_util

    def get_improvement(self, sess, node, is_eval_episode):
        """
        Try to get the better results
        """
        input_tensor = self.progressive_graph.get_embeddings()
        node_num = self.progressive_graph.nx_graph.number_of_nodes()

        print("input tensor: {}".format(input_tensor.shape))




        feed = self.model.get_feed_dict(self.progressive_graph, node)
        if is_eval_episode:
            feed[self.model.is_eval_ph] = 1.0

        print(feed)

        train_ops = [
            self.model.logprob_grad_outs,
            self.model.ent_grad_outs,
            self.model.log_probs,
            self.model.sample,
            self.model.pl_ent_loss,
            self.model.log_prob_loss,
            self.model.no_noise_classifier,
            self.model.entropy,
            self.model.ent_dec,
        ]

        s, lo, _, expl, *train_outs = sess.run(
            self.model.get_eval_ops() + [self.model.expl_act] + train_ops, feed_dict=feed
        )

        return s, lo, feed, expl, train_outs

    def compute_rewards(self, run_times, memory_utils):
        """
        compute the reward of every placements
        """
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
            # 累计奖励值，过去时间的奖励通过可定义的变量disc_factor放缩，衡量过去状态的重要性
            cum_rewards[i] += self.config_params["disc_factor"] * cum_rewards[i + 1][0]

        return cum_rewards[0]

    def identify_best_pl(self, reward_runtimes, run_times, memory_utils, placements):
        """
        get the best placements in all placements
        """
        best_rew_runtime = 1e20
        best_placement = None
        best_placement_runtime = []
        best_placement_memory = []

        min_reward = np.min(reward_runtimes)
        min_idx = np.argmin(reward_runtimes)

        if best_rew_runtime > min_reward:
            best_rew_runtime = min_reward
            best_placement = placements[min_idx]
            best_placement_runtime = run_times[min_idx]
            best_placement_memory = max(memory_utils[min_idx])

        return best_placement, best_rew_runtime, best_placement_runtime, best_placement_memory

    def update_best_pl(self, ep_best_reward_runtime, ep_best_placement, episode, is_eval_episode):
        if len(self.best_runtimes) < self.n_max_best_runtimes:
            self.epsiode_to_placement[episode] = [ep_best_placement, is_eval_episode]
        elif -self.best_runtimes[0][0] > ep_best_reward_runtime:
            _, del_episode = heappushpop(self.best_runtimes, (-ep_best_reward_runtime, episode))
            self.epsiode_to_placement.pop(del_episode, None)
            self.epsiode_to_placement[episode] = [ep_best_placement, is_eval_episode]

    def save_model(self, sess, episode):
        save_path = self.config_params["output_save_path"] + "/model"
        self.model_saver.save(sess, save_path, global_step=episode, write_meta_graph=False)
        log.debug("Saved model at %s", save_path)
