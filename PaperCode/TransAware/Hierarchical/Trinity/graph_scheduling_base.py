# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Graph Placer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import tensorflow as tf
import time
import os
import json
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import item as gitem
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import training
from Trinity import ColorRL_program
from Trinity import Trinity_program_norml
from Trinity import trinity_program_ppo
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Trinity import trinity_program_ppo
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops as tf_ops
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import cluster
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


def compute_rewards(run_times, mem_utils, com_utils, run_time_,isall=False):
    reward = 0
    print("本次调度时间： " + str(run_times))
    print("本次调度巅峰内存:" + str(max(mem_utils) / 1073741824))
    print("本次调度总通信:" + str(max(com_utils) / 1073741824))
    if isall:
        mem_excess = max(mem_utils) / 1073741824
        # 7600表示传输速度
        com_excess = max(com_utils) / 1073741824
        if mem_excess > 11:
            mem_excess = mem_excess * 10
        else:
            mem_excess = 0
        # else:
        #     mem_excess = mem_excess * 0.1
        reward += 0.0 * mem_excess + 0.0 * com_excess + run_times
        return reward
    else:
        return run_times


class GraphSchedulingBase(object):
    """将提供的MetaGraph进行放置.
                  Args:
                    MetaGraph: 需要进行放置的MetaGraph‘.
                    cluster: 一组可选的硬件资源，用于优化布局。
                      如果没有指定，那么将根据本地资源进行自动创建
                    allotted_time: 花费在优化放置上的最长时间（以秒作为单位）
                    hparams: 寻找最佳放置的超参数.
                    verbose: 如果为True那么将会输出调试信息.

                  Returns:
                    将会返回已经配置好放置后的MetaGraph
                  """

    # 度量使用原始位置可实现的运行时。
    def __init__(self, metagraph, cluster, hparams, verbose, op_perf=None, step_stats=None, step=500, issim=True, isbase=True):
        self.issim = issim
        self.isbase = isbase
        self.metagraph = metagraph
        self.cluster = cluster
        # self.allotted_time = allotted_time
        self.op_perf, self.step_stats = op_perf, step_stats
        # 超参数
        self.hparams = hparams
        self.ungrouped_pl = None
        self.ungrouped_pl_ = None
        # 打印日志
        self.verbose = verbose
        self.model = None
        self.step = step
        self.run_time_list = []
        self.comm_list = []
        self.best_com_list = []
        self.best_mem_list = []
        self.mem_list = []
        self.best_run_time_list = []
        self.all_time = 0
        self.train_time_list = []
        self.reward = 0
        self.run_time_sim_list = []
        self.real_time_list = []
        self.sim_time_list = []
        self.reward_list = []
        if cluster is None:
            cluster = gcluster.Cluster(allow_soft_placement=False, disable_detailed_stats=False, disable_timeline=False)
        # 优化MetaGraph使其加速训练
        config = config_pb2.ConfigProto()
        # 生成一个GraphDef
        optimized_graph = tf_optimizer.OptimizeGraph(
            config, metagraph, verbose=verbose, cluster=cluster)
        # 生成一个空的MetaGraphDef
        optimized_metagraph = meta_graph_pb2.MetaGraphDef()
        # 将metagraph复制进来
        optimized_metagraph.CopyFrom(metagraph)
        # 将OptimizeGraph放到optimized_metagraph的graph_def中
        optimized_metagraph.graph_def.CopyFrom(optimized_graph)

        self.config = config
        self.optimized_graph = optimized_graph
        self.optimized_metagraph = optimized_metagraph
        # item = gitem.Item(optimized_metagraph)
        item = gitem.Item(self.optimized_metagraph)
        self.item = item
        # 如果没有启用模拟器
        if not self.op_perf and not self.step_stats and not issim:
            try:
                """返回指定item的训练成本
                    Args:
                      item: 需要衡量运行成本的item
                    Returns: 返回三元组，分别是op_perfs, runtime, step_stats.
                    """
                # 在此处op_perfs和step_stats后续没用到
                # tf.import_graph_def(optimized_graph, name="default")
                # saver = tf.train.import_meta_graph(metagraph)
                # with tf.Session() as sess:
                #     options = tf.RunOptions()
                #     saver.restore(sess, metagraph)
                #     metadata = tf.RunMetadata()
                #     sess.run(optimized_metagraph, options=options, run_metadata=metadata)
                #     print(metadata.step_stats)
                self.op_perf, self.original_run_time, self.step_stats = self.cluster.MeasureCosts(self.item)
                # self.original_run_time = self.hparams.failing_signal
                if self.verbose:
                    print("TF中MeasureCosts测量得到的原始运行时间: " + str(self.original_run_time))
            except errors.OpError as e:
                if self.verbose:
                    print("原始调度策略不可用: " + str(e))
                self.original_run_time = int(hparams.failing_signal)
                self.op_perf, self.step_stats = None, None
            self.run_time = self.original_run_time
            self.run_time_sim = self.original_run_time
            self.comm = self.hparams.failing_signal
            self.mem = self.hparams.failing_signal
            self.best_run_time = self.original_run_time
        else:
            self.original_run_time = self.hparams.failing_signal
            self.best_run_time = self.original_run_time

        # self.run_time_list.append(self.run_time)

    def schedule_graph(self, isall=False, issim=False, output_dir="./output/", isbase=True):
        if self.hparams is None:
            if isbase:
                self.hparams = ColorRL_program.colorrl_mian_hparams()
            else:
                self.hparams = Trinity_program_norml.trinity_mian_hparams()
        # We run with a single child
        # todo 这里的child是什么意思需要解决

        with tf_ops.Graph().as_default():
            # 将所有节点都放置在CPU上，我们不想让他们和模型的放置优化做竞争
            # Place all the nodes of the controller on the CPU. We don't want them to
            # fight for accelerator memory with the model to optimize.
            # 为了防止影响其他GPU对模型性能的测评，将有关策略搜索的程序都运行在CPU上
            with tf_ops.device("/device:CPU:0"):
                # Trinity分层架构的程序
                # self.model = trinity_program.TrinityProgram(
                #     self.hparams, self.item, self.cluster, self.op_perf, self.step_stats)
                if isbase:
                    self.model = ColorRL_program.ColorRLProgram(
                        self.hparams, self.item, self.cluster)
                else:
                    self.model = Trinity_program_norml.TrinityProgram(
                        self.hparams, self.item, self.cluster, self.op_perf, self.step_stats)
                ops = self.model.build_program()
                config_proto = config_pb2.ConfigProto()
                off = rewriter_config_pb2.RewriterConfig.OFF
                config_proto.graph_options.rewrite_options.arithmetic_optimization = off
                session_creator = training.ChiefSessionCreator(config=config_proto)
                with training.MonitoredSession(session_creator=session_creator) as sess:
                    writer = tf.summary.FileWriter(output_dir + "logs/", sess.graph)
                    # while current_time - start_time < self.allotted_time:
                    current_step = 0
                    while current_step < self.step:
                        # 首先对神经网络进行切分操作
                        partitioning_actions = self.model.partitioning(sess)
                        input_to_seq2seq = self.model.create_group_embeddings(
                            partitioning_actions, verbose=False)
                        # 再对神经网络进行分组操作
                        self.model.scheduling(input_to_seq2seq, sess)
                        # 采用模拟执行引擎
                        if issim:
                            self.run_time, self.mem, self.comm, sim_time, ungrouped_pl, ungrouped_pl_ = self.model.eval_grouping_and_scheduling_sim(sess,
                                                                                                             verbose=self.verbose,
                                                                                                             op_perfs=self.op_perf,
                                                                                                             step_stats=self.step_stats)
                            # self.real_time_list.append(real_time)
                            self.run_time_sim = self.run_time
                            self.sim_time_list.append(sim_time)
                            self.run_time_sim_list.append(self.run_time)
                            self.comm_list.append(str(max(self.comm) / 1e9))
                            self.mem_list.append(str(max(self.mem) / 1e9))
                            # print("真实环境（执行时间）：", real_time)
                            print("模拟环境（执行时间）：", sim_time)
                            # 将时间转换为秒metagraph, n_devs, op_perf, step_stats
                            # self.run_time = sim_eva(self.cluster, self.metagraph, n_devs, op_perf, step_stats)[
                            #                     0] / 1e6
                            #
                            # self.run_time_list.append(self.run_time)
                        else:
                            try:
                                start_time_real = time.time()
                                # 使用原本Google的分层方法
                                self.run_time, self.mem, self.comm = self.model.eval_placement(
                                    sess,
                                    verbose=self.verbose)
                                start_time_sim = time.time()
                                real_time = start_time_sim - start_time_real
                                if self.op_perf and self.step_stats:
                                    self.run_time_sim, _, _, sim_time, ungrouped_pl = self.model.eval_grouping_and_scheduling_sim(sess,
                                                                                                          verbose=self.verbose,
                                                                                                          op_perfs=self.op_perf,
                                                                                                          step_stats=self.step_stats)
                                self.real_time_list.append(real_time)
                                self.sim_time_list.append(sim_time)
                                self.run_time_list.append(self.run_time)
                                self.run_time_sim_list.append(self.run_time_sim)
                                self.comm_list.append(str(max(self.comm) / 1e9))
                                self.mem_list.append(str(max(self.mem) / 1e9))
                                print("TF虚拟反馈值（反馈值）：", real_time)
                                print("ios模拟环境（执行时间）：", self.run_time_sim)
                            except errors.OpError as e:
                                if self.verbose:
                                    print("运行计算图出错:" + str(e))
                                self.run_time = self.hparams.failing_signal
                                self.run_time_list.append(self.run_time)
                        reward = compute_rewards(self.run_time, self.mem, self.comm, self.run_time_sim, isall)
                        self.reward_list.append(reward)
                        updated = self.model.update_reward(sess, reward, verbose=self.verbose)
                        if updated and self.run_time < self.original_run_time:
                            if self.verbose:
                                self.best_run_time = self.run_time
                                print("--------------------------搜索到最佳的调度策略---------------------------")
                                print("最佳调度时间： " + str(self.run_time))
                                print("此时调度巅峰内存:" + str(max(self.mem) / 1e9))
                                print("此时调度总通信:" + str(max(self.comm) / 1e9))
                                print("-----------------------------------------------------------------------")
                            self.model.export_placement(self.metagraph)
                            self.best_run_time_list.append(self.best_run_time)
                            self.best_com_list.append(self.comm)
                            self.best_mem_list.append(self.mem)
                            self.ungrouped_pl = ungrouped_pl
                            self.ungrouped_pl_ = ungrouped_pl_
                        if not isbase:
                            self.model.process_reward(sess, self.model.METHOD)
                        else:
                            self.model.process_reward(sess)
                        # writer.add_summary(summ, current_step)
                        current_step = current_step + 1
                        if current_step % 10 == 0:
                            file_path = output_dir
                            if isall:
                                file_path += "all/"
                            else:
                                file_path += "sin/"
                            if not os.path.exists(file_path):
                                os.makedirs(file_path)
                            output = self.get_output()
                            with open(file_path + "output_" + str(current_step) + ".json", mode="a") as f:
                                json.dump(output, f)
                            with open(file_path + "output_" + str(current_step) + "_ungrouped_pl.json", mode="a") as f:
                                json.dump(self.ungrouped_pl_, f)
                            with open(file_path + "output_" + str(current_step) + "_ungrouped_pl.pkl", "wb") as file:
                                pickle.dump(self.ungrouped_pl, file)
                            # np.savez(file_path + "/output.npz",
                            #          graph_placer.get_output())
                            # print(graph_placer.get_output())
                            print("------------------记录一次中间执行结果----------------------")
        return self.metagraph

    def get_output(self):
        return {
            "original_run_time": self.original_run_time,
            "run_time_list": self.run_time_list,
            "comm_list": self.comm_list,
            "mem_list": self.mem_list,
            "best_run_time": self.best_run_time,
            "best_run_time_list": self.best_run_time_list,
            "best_mem_list": self.best_mem_list,
            "best_com_list": self.best_com_list,
            "steps": self.step,
            "run_time_sim_list": self.run_time_sim_list,
            "real_time_list": self.real_time_list,
            "sim_time_list": self.sim_time_list,
            "reward_list": self.reward_list
        }

    def get_step_stats(self):
        return self.step_stats

    def get_op_perfs(self):
        return self.op_perfs

    def get_original_run_time(self):
        return self.original_run_time
