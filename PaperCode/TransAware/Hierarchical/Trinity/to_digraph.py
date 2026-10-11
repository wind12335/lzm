import collections
import functools
import operator
import networkx as nx
from numpy.lib._iotools import str2bool
from future.utils import bytes_to_native_str
from tensorflow.core.protobuf import device_properties_pb2
import os
import argparse
Cost = collections.namedtuple(
    'Cost',
    ['compute_cost', 'output_memory', 'temporary_memory', 'persistent_memory'])
import tensorflow as tf
from tensorflow.python.grappler import cluster as cluster
from tensorflow.python.grappler import item as item
_PERSISTENT_OPS = ["Variable", "VariableV2", "AutoReloadVariable",
                   "VarHandleOp", "ReadVariableOp"]
import re
import pickle


def main_cost(step_stats):
    # 跳过以XX结尾的操作符
    # 这些操作符是与原始操作符类似的副本
    # 由于通信的原因
    def _is_skip_node(node_name):
        if node_name == '_SOURCE' or re.match(r".+/_\d+$", node_name) or 'ShapeN' in node_name:
            print("Skipping node name=%s" % node_name)
            return True
        return False

    def _calculate_memory_size(tensor_desc):
        return tensor_desc.allocation_description.allocated_bytes

    def _add_to_cost_dict(op_name, key, value, update_existing=False):
        if op_name in raw_cost_dict:
            stat_dict = raw_cost_dict[op_name]
            if key in stat_dict:
                assert update_existing
                stat_dict[key] += value
            else:
                stat_dict[key] = value
        else:
            raw_cost_dict[op_name] = {key: value}

    raw_cost_dict = {}
    # 获取step_stats和MetaGraphDef
    triple_div = "./triple_resnet_50.pkl"
    hardware_stats = None
    gpu_dev_stats_list = []
    for dev_stats in step_stats.dev_stats:
        device = dev_stats.device
        last_device_string = device.split('/')[-1]
        # 设备名称通常采用如下格式:
        #   "/device:GPU:0/stream:all" (hardware stats)
        #   "/job:localhost/replica:0/task:0/device:GPU:0" (normal stats)
        if last_device_string == 'stream:all':
            hardware_stats = dev_stats
        elif 'device:' in last_device_string:
            gpu_dev_stats_list.append(dev_stats)
        else:
            print("Skipping stats for device: %s", device)

    if hardware_stats is not None:
        for node_stats in hardware_stats.node_stats:
            # stream:all 只有计算代价
            name = node_stats.node_name.split(":")[0]
            if _is_skip_node(name):
                continue
            _add_to_cost_dict(
                name, 'compute_cost', node_stats.op_end_rel_micros,
                update_existing=True)
    for dev_stats in gpu_dev_stats_list:
        for node_stats in dev_stats.node_stats:
            name = node_stats.node_name
            if _is_skip_node(name):
                continue
            # 计算代价
            # 如果 hardware stats 可用, 跳过
            if name not in raw_cost_dict:
                _add_to_cost_dict(
                    name, 'compute_cost', node_stats.op_end_rel_micros)
            # output memory
            output_memory = [None] * len(node_stats.output)
            for output in node_stats.output:
                output_memory[output.slot] = \
                    _calculate_memory_size(output.tensor_description)
            _add_to_cost_dict(name, 'output_memory', output_memory)

            # memory stats
            memory_stats = node_stats.memory_stats
            _add_to_cost_dict(name, 'temporary_memory',
                              memory_stats.temp_memory_size)
            _add_to_cost_dict(name, 'persistent_memory',
                              memory_stats.persistent_memory_size)
            # convert raw cost dict to immutable cost dict

    return {name: Cost(**raw_cost) for name, raw_cost in raw_cost_dict.items()
            if len(raw_cost) == 4}


def _add_nodes(op_graph, _stats, _tf_graph, ungrouped_mapping, _is_cost_dict=True):
    def _get_raw_colocation_group(tf_op):
        """Returns a raw string-typed co-location group of the given TF op."""
        return [bytes_to_native_str(colocation_group)
                for colocation_group in tf_op.colocation_groups()][0]

    def _is_persistent_op(tf_op):
        """Returns whether the given tf.Operation is a persistent op."""
        op_type = tf_op.type
        return op_type in _PERSISTENT_OPS

    def _get_estimated_memory(tensor):
        """Returns estimated memory bytes from tf.Tensor()."""
        if not tensor.shape:
            # unknown shape. TODO: fix
            shape_list = [1]
        else:
            shape_list = tensor.shape.as_list()
            if len(shape_list) == 0:
                shape_list = [1]
            # replace None to 1. TODO: fix.
            shape_list = [shape if shape else 1 for shape in shape_list]
        return tensor.dtype.size * functools.reduce(operator.mul, shape_list)

    """Adds operator nodes to the given op graph.

            Returns:
                a dict that maps from a operator name to an index in the graph.
            """
    # pylint: disable=too-many-locals
    # the weight of each node is the computation time
    tf_only_ops = set()
    num_fixed_cost = 0
    num_fixed_out_size = 0
    stats = dict(_stats)  # generate a copy

    for i, tf_op in enumerate(_tf_graph.get_operations()):
        if tf_op.name in stats:
            data = stats[tf_op.name]
            if _is_cost_dict:
                cost = data.compute_cost
                out_size = list(data.output_memory)
                mem = sum(out_size)
            else:
                cost = data["dur"]
                out_size = [output["requested_bytes"]
                            for output in data["outputs"]]
                # TODO: fix this. all output memory is not persistent
                mem = sum(out_size)

            # each operator should have computer cost > 0
            if cost <= 0:
                print("Change duration of op=%s from %d to 1", tf_op.name, cost)
                cost = 1
                num_fixed_cost += 1

            # output memory should be zero or a positive value
            for port, out_memory in enumerate(out_size):
                if out_memory <= 0:
                    print("Change output memory of %s:%d from %d to 0", tf_op.name, port, out_size[port])
                    out_size[port] = 0
                    num_fixed_out_size += 1

            if len(tf_op.outputs) != len(out_size):
                raise RuntimeError(
                    "Different # outputs for {}. tf#={} stats#={}".format(
                        tf_op.name, len(tf_op.outputs),
                        len(out_size)))

            del stats[tf_op.name]
        else:
            print("op: %s does not have stats", tf_op.name)
            tf_only_ops.add(tf_op.name)
            # TODO: is this okay?
            cost = 1
            out_size = [_get_estimated_memory(output)
                        for output in tf_op.outputs]
            if _is_persistent_op(tf_op):
                mem = sum(out_size)
            else:
                mem = sum(out_size)
        ungrouped_mapping[str(tf_op.name)] = str(tf_op.name)
        op_graph.add_node(str(tf_op.name), name=str(tf_op.name), cost=cost, mem=mem, out_memory=out_size,
                          out_size=sum(out_size), device=str(tf_op.device),
                          aggregated_nodes=[str(tf_op.name)], colocation_group=_get_raw_colocation_group(tf_op))


def _add_edges(_tf_graph, op_graph):
    """Adds edges between operators to the graph."""
    i = 0
    for tf_op in _tf_graph.get_operations():
        for input_tensor in tf_op.inputs:
            input_op_name, index = input_tensor.name.split(":")
            index = int(index)
            out_size = op_graph.nodes[input_op_name]["out_memory"][index]

            # the weight is the communication time.
            # weight = self._comm_cost_fn(out_size)

            edge_data = {
                "id": i,
                # "weight": weight,
                "tensor": [{
                    "name": input_tensor.name,
                    # "weight": weight,
                    "num_bytes": out_size,
                }],
            }

            if op_graph.has_edge(input_op_name, tf_op.name):
                # merge edge data only if a new tensor is transferred.
                prev_edge_data = op_graph[input_op_name][tf_op.name]
                exists = False
                for prev_edge_tensor in prev_edge_data['tensor']:
                    if prev_edge_tensor['name'] == input_tensor.name:
                        exists = True
                        break
                if not exists:
                    # prev_edge_data['weight'] += edge_data['weight']
                    prev_edge_data['tensor'] += edge_data['tensor']
            else:
                op_graph.add_edge(input_op_name, tf_op.name, **edge_data)
                print("New edge added. [%s] -> [%s] %s",
                      input_op_name,
                      input_tensor.name,
                      tf_op.name,
                      edge_data
                      )
                i += 1

        # use 4 bytes for control edges
        # weight = self._comm_cost_fn(4)
        for control_input in tf_op.control_inputs:
            # control_input_op_id = op_index[control_input.name]
            edge_data = {
                "id": i,
                # "weight": weight,
                "tensor": [{
                    # control tensor starts with ^
                    "name": "^%s" % control_input.name,
                    # "weight": weight,
                    "num_bytes": 4,
                }],
            }

            assert not op_graph.has_edge(control_input.name, tf_op.name)

            op_graph.add_edge(control_input.name, tf_op.name, **edge_data)
            # _LOGGER.debug("New edge added. %d[%s] -> %d[%s] %s",
            #                 control_input_op_id,
            #                 control_input.name,
            #                 op_id,
            #                 tf_op.name,
            #                 edge_data)
            i += 1

    assert i == op_graph.number_of_edges()


def initialize(_tf_graph, mg, virtual=False):
        if virtual:
            devices = []
            device_properties = device_properties_pb2.DeviceProperties(
                type='GPU',
                vendor='NVIDIA',
                model='Tesla P100-PCIE-12GB',
                frequency=1328,
                num_cores=56,
                environment={'architecture': '6.0',
                             'cuda': '10010',
                             'cudnn': '7604'},
                num_registers=65536,
                l1_cache_size=24576,
                l2_cache_size=3145728,
                shared_memory_size_per_multiprocessor=65536,
                memory_size=12790923264,
                bandwidth=549120000)
            grappler_item = item.Item(mg)
            gcluster = cluster.Cluster(allow_soft_placement=False, disable_detailed_stats=False, disable_timeline=False,
                                       devices=devices)
            for i in range(4):
                devices.append(
                    device_properties_pb2.NamedDevice(
                        properties=device_properties, name='/GPU:' + str(i)))
        else:
            grappler_item = item.Item(mg)
            gcluster = cluster.Cluster(disable_timeline=False, disable_detailed_stats=False)

        op_perf, original_run_time, step_stats = gcluster.MeasureCosts(grappler_item)
        return _tf_graph, mg, op_perf, step_stats


def to_digraph(_tf_graph, Meta_graph_def, op_perf, step_stats, out_file):
    ungrouped_mapping = {}
    G = nx.DiGraph()
    _add_nodes(G, main_cost(step_stats), _tf_graph, ungrouped_mapping, True)
    _add_edges(_tf_graph, G)
    with open(out_file, 'wb') as file:
        pickle.dump({"optim_mg": Meta_graph_def, "op_perf": op_perf, "step_stats": step_stats, "G": G,
                     "ungrouped_mapping": ungrouped_mapping}, file)
    print("Finish！！")
    return Meta_graph_def, op_perf, step_stats, G, ungrouped_mapping


def to_digraph_main(_tf_graph, mg, out_file):
    _tf_graph, Meta_graph_def, op_perf, step_stats = initialize(_tf_graph, mg, virtual=True)
    return to_digraph(_tf_graph, Meta_graph_def, op_perf, step_stats, out_file)






