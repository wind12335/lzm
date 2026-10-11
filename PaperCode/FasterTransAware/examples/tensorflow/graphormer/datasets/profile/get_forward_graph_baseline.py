import re
import pickle
import collections
import functools
import operator
import tensorflow as tf
import networkx as nx
import numpy as np

from utils import logger
from cv.networks import nets_factory
from nlp import model_factory
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import item as gitem
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging
from future.utils import bytes_to_native_str

_CLASSIFICATION_NUM = {
    'cifarnet': 10,
    'inception_v3': 10,
    'resnet_v1_50': 10,
    'resnet_v1_152': 10,
    'resnet_v1_101': 10,
    'resnet_v1_200': 10,
    'nasnet': 10,
    'nasnet_cifar': 10,
    'vgg_19': 10,
}

_PERSISTENT_OPS = ["Variable", "VariableV2", "AutoReloadVariable",
                   "VarHandleOp", "ReadVariableOp"]

Cost = collections.namedtuple(
    'Cost',
    ['compute_cost', 'output_memory', 'temporary_memory', 'persistent_memory'])

# row_graph
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Logging device placement.')
tf.app.flags.DEFINE_string('eval_graph_dest', './cifarnet.pkl', 'Logging device placement.')
tf.app.flags.DEFINE_string('dest_graph_output', './cifarnet.pkl', 'Logging device placement.')
tf.app.flags.DEFINE_boolean('colocate_grads_with_ops', True, 'Colocate gradient with ops.')
tf.app.flags.DEFINE_enum(
    'optimizer', 'adam',
    ['adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'sgd', 'rmsprop'],
    'The name of the optimizer')
tf.app.flags.DEFINE_string('model_name', 'resnet_v1_152', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer('batch_size', 1, 'The number of samples in each batch.')
# 初始值 32
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_string('logdir', '', 'Path to log dir.')
tf.app.flags.DEFINE_string('cost_path', '/tmp/cost_xyk.pkl', 'Path to the cost file.')
tf.app.flags.DEFINE_string('place_path', './tmp/best_place.json', 'Path to the place file.')
tf.app.flags.DEFINE_boolean('costgen', False, 'Generate cost dict.')
tf.app.flags.DEFINE_boolean('only_forward', True, 'Consider only forward ops.')
tf.app.flags.DEFINE_float('memory_fraction', 1.0, 'GPU memory fraction')
tf.app.flags.DEFINE_string(
    'comm_cost_coeffs', '0.0001754,134',
    'Comma-separated linear communication cost function coefficients')
tf.app.flags.DEFINE_float(
    'comm_cost_factor', 1.0, 'Communication cost function factor.')
tf.app.flags.DEFINE_float(
    'cost_factor', 1.0, 'Factor that applies to all costs')
###### Image classifier ######
tf.app.flags.DEFINE_enum(
    'data_format', 'NHWC', ['NHWC', 'NCHW'], 'Image data format')
tf.app.flags.DEFINE_enum(
    'exe_pattern', 'aware_forward', ['aware_forward', 'eval', 'ColorRL', 'aware_back'], 'Image data format')
##### NMT ######
tf.app.flags.DEFINE_integer('vocab_size', 5000, 'Vocabulary size.')
tf.app.flags.DEFINE_integer('max_seq_length', 30, 'Max. sequence length.')
tf.app.flags.DEFINE_integer('rnn_units', 1024, 'RNN units.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'RNN # layers.')
tf.app.flags.DEFINE_enum(
    'rnn_unit_type', 'lstm', ['lstm', 'gru'], 'RNN unit type.')
tf.app.flags.DEFINE_enum(
    'encoder_type', 'bi', ['bi', 'uni', 'gnmt'], 'Encoder type.')
tf.app.flags.DEFINE_boolean(
    'residual', False, 'Add residual connections to RNN.')
tf.app.flags.DEFINE_integer('num_gpus', 80, 'Number of gpus for NMT.')
tf.app.flags.DEFINE_boolean('disable_nmt_colocation', False,
                            'Disable the NMT ops colocation.')
##### Grappler ######
tf.app.flags.DEFINE_boolean('ColorRL', False, 'Use Grappler.')
tf.app.flags.DEFINE_integer(
    'grappler_time', 3600, 'Allotted time in seconds for Grappler.')

_LOGGER = logger.get_logger(__file__)

def _get_gpu_devices(sess_config):
    with tf.Session(config=sess_config) as sess:
        return [
            {"name": device.name,
             "memory_size": device.memory_limit_bytes,
             "type": device.device_type}
            for device in sess.list_devices()
            if device.device_type == 'GPU']

def _build_nlp_inputs(batch_size, max_seq_length):
    input_shape = (batch_size, max_seq_length)

    src_input = np.ones(input_shape, dtype=np.int32)
    target_input = np.ones(input_shape, dtype=np.int32)
    target_output = np.ones(input_shape, dtype=np.int32)

    element = (src_input, target_input, target_output)

    with tf.variable_scope('dataset'):
        dataset = tf.data.Dataset.from_tensors(element).repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

def _build_cv_inputs(model_name, batch_size, data_format):
    num_classes = _CLASSIFICATION_NUM[model_name]
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes)

    if data_format == 'NHWC':
        input_shape = (batch_size,
                       network_fn.default_image_size,
                       network_fn.default_image_size,
                       3)
    else:
        input_shape = (batch_size,
                       3,
                       network_fn.default_image_size,
                       network_fn.default_image_size)

    images = np.ones(input_shape, dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.int32)

    element = (images, labels)

    with tf.variable_scope('dataset'):
        dataset = tf.data.Dataset.from_tensors(element).repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

def build_inputs(model_name, batch_size, data_format, max_seq_length):
    if model_name in _CLASSIFICATION_NUM:
        return _build_cv_inputs(
            model_name, batch_size, data_format)
    return _build_nlp_inputs(batch_size, max_seq_length)

def build_cv_model(inputs, model_name, data_format):
    images, labels = inputs
    num_classes = _CLASSIFICATION_NUM[model_name]

    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes)
    logits, _ = network_fn(images)
    
    with tf.variable_scope('loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_sum(losses) / tf.to_float(images.shape[0])
    return loss

def build_nlp_networks(inputs, model_name, **kwargs):
    """Builds NMT with the given specs."""
    src_input, target_input, target_output = inputs
    vocab_size = kwargs.pop('vocab_size')

    # replicate vocab size
    kwargs['src_vocab_size'] = vocab_size
    kwargs['tgt_vocab_size'] = vocab_size

    model_fn = model_factory.get_model_fn(model_name, **kwargs)
    _, loss = model_fn(src_input, target_input, target_output)

    return loss

def build_deep_learning_network(inputs, model_name, data_format, **kwargs):
    """Builds a model with the given specs."""
    if model_name in _CLASSIFICATION_NUM:
        return build_cv_model(inputs, model_name, data_format)

    return build_nlp_networks(inputs, model_name, **kwargs)

def parse_comm_cost_coeffs(coeffs_str, factor=1.0):
    comm_cost_coeffs = coeffs_str.split(',')
    assert len(comm_cost_coeffs) == 2

    comm_cost_coeffs[0] = float(comm_cost_coeffs[0])
    comm_cost_coeffs[1] = int(comm_cost_coeffs[1])

    if factor != 1.0:
        _LOGGER.info('通信代价影响因子: %s', str(factor))
        comm_cost_coeffs = tuple(
            [value * factor for value in comm_cost_coeffs])

    return comm_cost_coeffs

def generate_aware_graph(target_op, allotted_time, comm_cost_coeffs, logdir, verbose, sess_config):
    graph = tf.get_default_graph()
    devices = _get_gpu_devices(sess_config)
    cluster = gcluster.Cluster(disable_detailed_stats=False, disable_timeline=False)
    metagraph = tf.train.export_meta_graph(graph=graph,
                                           clear_extraneous_savers=True)
    item = gitem.Item(metagraph, ignore_colocation=False)
    op_perf, _, step_stats = cluster.MeasureCosts(item)

    return graph, metagraph, op_perf, step_stats

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
        op_graph.add_node(str(tf_op.name), 
                          name=str(tf_op.name), 
                          cost=cost, 
                          mem=mem, 
                          out_memory=out_size,
                          out_size=sum(out_size), device=str(tf_op.device),
                          aggregated_nodes=[str(tf_op.name)], 
                          colocation_group=_get_raw_colocation_group(tf_op))

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
            i += 1

    assert i == op_graph.number_of_edges()

def to_digraph(_tf_graph, Meta_graph_def, op_perf, step_stats):
    ungrouped_mapping = {}
    G = nx.DiGraph()
    _add_nodes(G, main_cost(op_perf, step_stats), _tf_graph, ungrouped_mapping, True)
    _add_edges(_tf_graph, G)
    # Get nodes
    nodes = G.nodes()
    print("Nodes num:", len(nodes))

    # Get edges
    edges = G.edges()
    print("Edges:", len(edges))
    
    return G, ungrouped_mapping

def main_cost(op_perf, step_stats):
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

    # 解析硬件数据
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

def main(unparsed_args):
    FLAGS = tf.app.flags.FLAGS

    sess_config = tf.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=FLAGS.log_device_placement)

    if FLAGS.memory_fraction != 1.0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = \
            FLAGS.memory_fraction
    sess_config.graph_options.optimizer_options.opt_level = -1
    _LOGGER.debug('Session config: %s', str(sess_config))

    inputs = build_inputs(
        model_name=FLAGS.model_name,
        batch_size=FLAGS.batch_size,
        # image classifier
        data_format=FLAGS.data_format,
        # NMT
        max_seq_length=FLAGS.max_seq_length,
    )

    print("inputs dir {}".format(type(inputs)))

    target_op = build_deep_learning_network(
        inputs=inputs,
        model_name=FLAGS.model_name,
        # image classifier
        data_format=FLAGS.data_format,
        # NMT
        vocab_size=FLAGS.vocab_size,
        rnn_units=FLAGS.rnn_units,
        num_layers=FLAGS.num_layers,
        rnn_unit_type=FLAGS.rnn_unit_type,
        encoder_type=FLAGS.encoder_type,
        residual=FLAGS.residual,
        num_gpus=FLAGS.num_gpus,
        colocation=not FLAGS.disable_nmt_colocation)

    comm_cost_coeffs = parse_comm_cost_coeffs(
        FLAGS.comm_cost_coeffs, FLAGS.comm_cost_factor)

    tf.add_to_collection(tf.GraphKeys.TRAIN_OP, target_op)

    graph, metagraph, op_perf, step_stats = generate_aware_graph(target_op,
                                            allotted_time=FLAGS.grappler_time,
                                            comm_cost_coeffs=comm_cost_coeffs,
                                            logdir=FLAGS.logdir, verbose=True, sess_config=sess_config)

    G, ungrouped_mapping = to_digraph(graph, metagraph, op_perf, step_stats)

    # print("Node and Value:")
    # for node in G.nodes():
    #     print(G.nodes[node])

    with open(FLAGS.dest_graph_output, 'wb') as file:
        pickle.dump({"G": G,                    # 保存计算图
                     "optim_mg": metagraph,      # 保存metagraph
                     "ungrouped_mapping": ungrouped_mapping,
                     "op_perf": op_perf,               # 这里面包含节点的profile信息
                     "step_stats": step_stats}, file)  # 这里面包含了设备信息

if __name__ == "__main__":
    tf.app.run(main)
