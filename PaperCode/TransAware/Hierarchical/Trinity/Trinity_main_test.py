import sys
import random

# noinspection PyProtectedMember

sys.path.append('./')
sys.path.append('progressive_placers/')
sys.path.append('sim/')
sys.path.append('model/')
sys.path.append('Trinity/')
import os
import json
import argparse
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.grappler import cluster
from readfile import ReadPickleFile
from graph_scheduling import GraphScheduling
from graph_scheduling_base import GraphSchedulingBase
from ColorRL_program import colorrl_mian_hparams
from Trinity_program_norml import trinity_mian_hparams
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.grappler import cluster
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class TrinityControllerTest(object):

    def __init__(self, num_gpus, num_cpus=1, start_virtual_GPU=True, allow_soft_placement=False, disable_detailed_stats=True, disable_timeline=True):
        self.cluster = self._buildCluster(num_cpus, num_gpus, allow_soft_placement, disable_detailed_stats, disable_timeline, start_virtual_GPU)
        pass

    @staticmethod
    def _buildCluster(num_cpus=0, num_gpus=1, allow_soft_placement=True, disable_detailed_stats=True, disable_timeline=True, start_virtual_GPU=True):
        devices = []
        # 配置PGU
        if num_gpus > 0:
            device_properties = device_properties_pb2.DeviceProperties(
                type='GPU',
                vendor='NVidia',
                model='GeForce GTX TITAN X',
                frequency=1076,
                num_cores=24,
                environment={'architecture': '5.2',
                             'cuda': '8000',
                             'cudnn': '6021'},
                num_registers=65536,
                l1_cache_size=24576,
                l2_cache_size=3145728,
                shared_memory_size_per_multiprocessor=98304,
                memory_size=12783648768,
                bandwidth=336480000)
            for i in range(num_gpus):
                devices.append(
                    device_properties_pb2.NamedDevice(
                        properties=device_properties, name='/GPU:' + str(i)))
        # 配置CPU
        if num_cpus > 0:
            device_properties = device_properties_pb2.DeviceProperties(
                type='CPU',
                frequency=2000,
                num_cores=4,
                l1_cache_size=32768,
                l2_cache_size=262144,
                l3_cache_size=12582912)
            for i in range(num_cpus):
                devices.append(
                    device_properties_pb2.NamedDevice(
                        properties=device_properties, name='/CPU:' + str(i)))
        if start_virtual_GPU:
            return cluster.Cluster(devices=devices)
        else:
            return cluster.Cluster(allow_soft_placement, disable_detailed_stats, disable_timeline)

    def getCluster(self):
        return self.cluster


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 该参数指定被训练对象目录，被训练对象以pkl文件形式存储
    parser.add_argument('--pickled-inp-file', '-i', type=str, default="./datasets/GNMT/gnmt_baseline.pkl", nargs='+')
    # 是否使用虚拟化GPU
    parser.add_argument('--virtual', type=bool, default=True)
    # 该参数指定放置设备数量
    parser.add_argument('--n-devs', type=int, default=4)
    # 是否开启日志打印，默认开启
    parser.add_argument('--verbose', type=bool, default=True)
    # 是否在开始时候重置计算图分配设备
    parser.add_argument('--reset-devs', type=bool, default=True)
    # 是否使用仿真模拟器
    parser.add_argument('--is-sim', type=bool, default=True)
    # 是否使用综合模型衡量指标
    parser.add_argument('--is-all', type=bool, default=True)
    parser.add_argument('--is-base', type=bool, default=True)
    # 总共的迭代次数是为多少轮
    parser.add_argument('--all-steps', type=int, default=120)
    # 使用多少块CPU，如果使用模拟器请设置为0，需要同构设备
    parser.add_argument('--num-cpus', type=int, default=0)
    parser.add_argument('--model-dir', type=str, default="../datasets/ResNet/cifar10.pkl")
    # 总共用于优化计算图的时间是为多少时间，和reset-devs选一个即可，这里采用reset-devs，弃用
    # parser.add_argument('--allotted-time', type=int, default=500f)
    # 指定结果输出文件夹地址，如果当前不存在，那么久创建
    parser.add_argument('--output-dir', type=str, default="../output/GNMT/")
    args, unknown = parser.parse_known_args()
    # 获取集群配置信息
    print("----------------获取集群配置信息--------------")
    gcluster = TrinityControllerTest(args.n_devs, args.num_cpus, start_virtual_GPU=args.virtual, allow_soft_placement=True, disable_detailed_stats=False, disable_timeline=False).getCluster()
    # pkl文件需要包含完整的图和日志信息
    print("----------------读取文件信息------------------")
    rf = ReadPickleFile(pickled_inp_file=args.pickled_inp_file, gcluster=gcluster, reset_devs=args.reset_devs, devices_num=args.n_devs)
    mg, G, ungroup_map = rf.get_triple()
    # f = open(args.model_dir, 'rb')
    # mg = pickle.load(f)
    # 初始化设备图的放置
    if args.reset_devs:
        available_devices = [device.name for device in gcluster.ListDevices()]
        for node in mg.graph_def.node:
            if node.device is not None:
                # node.device = available_devices[random.randint(0, devices_num)]
                # 全都初始化为第一个GPU
                node.device = available_devices[0]
    if args.is_base:
        hparams = colorrl_mian_hparams()
    else:
        hparams = trinity_mian_hparams()
    # save = tf.train.import_meta_graph(mg, clear_devices=True)

    # saver.restore(sess, model_path + 'model.cptk')
    # with tf.Session() as sess:
    #      saver = tf.train.import_meta_graph('../InceptionV3/my_graph.meta', clear_devices=True)
    #    saver.restore(sess, "../save/model.ckpt.data-00000-of-00001")
    #      mg = tf.train.export_meta_graph(mg, 'name.meta')
    # op_perf, step_stats = rf.get_outperform()
    op_perf, step_stats = rf.get_outperform()

    print("-----------------生成调度器对象---------------")
    if args.is_base:
        graph_placer = GraphSchedulingBase(mg, cluster=gcluster, op_perf=op_perf, step_stats=step_stats,
                                       hparams=hparams,
                                       verbose=args.verbose, step=args.all_steps, issim=args.is_sim,
                                       isbase=args.is_base)
    else:
        graph_placer = GraphScheduling(mg, cluster=gcluster, op_perf=op_perf, step_stats=step_stats,
                                       hparams=hparams,
                                       verbose=args.verbose, step=args.all_steps, issim=args.is_sim,
                                       isbase=args.is_base)

    # op_perf, step_stats = rf.get_outperform()
    # graph_placer = GraphPlacer(rf.getMg(), cluster=gcluster, hparams=hierarchical_controller_hparams(), verbose=args.verbose, step=args.all_steps)
    print("-----------------开始调度---------------")
    
    # [新增] 开始计时
    import time
    start_timestamp = time.time()

    placed_mg = graph_placer.schedule_graph(args.is_all, args.is_sim, output_dir=args.output_dir + str(args.n_devs) + "/", isbase=args.is_base)
    
    # [新增] 停止计时并打印
    end_timestamp = time.time()
    duration = end_timestamp - start_timestamp
    
    print("\n" + "="*40)
    print(f"【Trinity 策略搜索统计】")
    print(f"总耗时 (Total Search Time): {duration:.4f} 秒")

    # placed_mg = graph_placer.place_graph(args.is_all, args.is_sim, n_devs=args.n_devs, ungroup_map=ungroup_map, output_dir=args.output_dir)
    file_path = args.output_dir + str(args.n_devs) + "/"
    if args.is_all:
        file_path += "all/"
    else:
        file_path += "sin/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    output = graph_placer.get_output()
    with open(file_path + "output_final.json", mode="a") as f:
        json.dump(output, f)
    # np.savez(file_path + "/output.npz",
    #          graph_placer.get_output())
    # print(graph_placer.get_output())
    print("------------------使用Trinity方法搜索最优模型并行策略结束----------------------")
