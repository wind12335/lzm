import sys

sys.path.append('sim/')
sys.path.append('../sim/')
sys.path.append('../')
# from important_ops_simulator import ImportantOpsSimulator
# from grouper.group_pruner import neighbor_merge_pruner
import pickle
import networkx as nx
from sim.tf_sim.tf_pl_simulator import ImportantOpsSimulator
import random


class ReadPickleFile(object):
    #def __init__(self, pickled_inp_file, gcluster=None, reset_devs=False, devices_num=1):
    #    pickled_inp_file = pickled_inp_file[0]
    #    with open(pickled_inp_file, 'rb') as f:
    #        j = pickle.load(f)
    #        mg, G, ungroup_map = j['optim_mg'], j['G'], j['ungrouped_mapping']
    def __init__(self, pickled_inp_file, gcluster=None, reset_devs=False, devices_num=1):
        pickled_inp_file = pickled_inp_file[0]
        with open(pickled_inp_file, 'rb') as f:
            j = pickle.load(f)
            
            # --- 修改开始：兼容不同键名并自动补全 ---
            
            # 1. 尝试读取 MetaGraph (Trinity用 'optim_mg', TransAware用 'Meta_graph_def')
            if 'optim_mg' in j:
                mg = j['optim_mg']
            elif 'Meta_graph_def' in j:
                mg = j['Meta_graph_def']
            else:
                raise KeyError("Pickle文件中缺少 MetaGraph 定义 (找不到 'optim_mg' 或 'Meta_graph_def')")

            # 2. 尝试读取 NetworkX图 (Trinity用 'G', TransAware用 'nx_graph')
            if 'G' in j:
                G = j['G']
            elif 'nx_graph' in j:
                G = j['nx_graph']
            else:
                raise KeyError("Pickle文件中缺少图结构 (找不到 'G' 或 'nx_graph')")

            # 3. 尝试读取映射表，如果缺失则自动生成恒等映射
            if 'ungrouped_mapping' in j:
                ungroup_map = j['ungrouped_mapping']
            else:
                print("Warning: 'ungrouped_mapping' not found. Generating identity mapping.")
                # 生成一个 key 和 value 相同的字典，对应 TransAware 的逻辑
                ungroup_map = {node: node for node in G.nodes()}

            # --- 修改结束 ---

        # if reset_devs and gcluster:
        #     available_devices = [device.name for device in gcluster.ListDevices()]
        #     for node in mg.graph_def.node:
        #         if node.device is not None:
        #             # node.device = available_devices[random.randint(0, devices_num)]
        #             node.device = available_devices[random.randint(0, devices_num-1)]
        # else:
        #     for node in mg.graph_def.node:
        #         if node.device == '/job:localhost/replica:0/task:0/device:GPU:1' or node.device == '/job:localhost/replica:0/task:0/device:GPU:2' or node.device == '/job:localhost/replica:0/task:0/device:GPU:3':
        #             node.device = '/job:localhost/replica:0/task:0/device:GPU:0'
        self.mg, self.G, self.ungroup_map = mg, G, ungroup_map
        self.j = j
        # if reset_devs and gcluster:
        #     available_devices = [device.name for device in gcluster.ListDevices()]
        #     for node in mg.graph_def.node:
        #         if node.device is not None:
        #             # node.device = available_devices[random.randint(0, devices_num)]
        #             node.device = available_devices[random.randint(0, devices_num-1)]
        # else:
        #     for node in mg.graph_def.node:
        #         if node.device == '/job:localhost/replica:0/task:0/device:GPU:1' or node.device == '/job:localhost/replica:0/task:0/device:GPU:2' or node.device == '/job:localhost/replica:0/task:0/device:GPU:3':
        #             node.device = '/job:localhost/replica:0/task:0/device:GPU:0'
        self.mg, self.G, self.ungroup_map = mg, G, ungroup_map
        self.j = j

    def getMg(self):
        return self.mg

    def get_triple(self):
        return self.mg, self.G, self.ungroup_map

    def get_stats(self):
        return self.j['step_stats']

    def get_outperform(self):
        return self.j['op_perf'], self.j['step_stats']
