import argparse
import copy
import numpy as np
import tensorflow as tf
import threading
import os
import sys
import time
import json
import pickle
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
# from examples.tensorflow.bert.utils.common import TransformerArgument
# from examples.tensorflow.bert.utils.common import time_test
from examples.tensorflow.graphormer.utils.common import cross_check
from examples.tensorflow.graphormer.tensorflow_graphormer.graphormer import run_tf_graphormer
# from examples.tensorflow.bert.utils.bert import ft_bert
# from examples.tensorflow.bert.utils.bert import build_sequence_mask

from examples.tensorflow.graphormer.utils.graphormer import FasterGraphormer

# tf.compat.v1.enable_eager_execution()

def graphormer_example(args_dict):
    print("\n=============== Argument ===============")
    for key in args_dict:
        print("{}: {}".format(key, args_dict[key]))
    print("========================================")

    np.random.seed(1)
    tf.set_random_seed(1)

    batch_size = args_dict['batch_size']
    num_layer = args_dict['num_layer']
    head_num = args_dict['head_number']
    size_per_head = args_dict['size_per_head']
    hidden_dim = head_num * size_per_head
    input_vocab_size = args_dict['input_vocab_size']
    inter_size = args_dict['inter_size']
    sp_enc = args_dict['sp_enc']
    max_num_nodes = args_dict['max_num_nodes']
    max_degree = args_dict['max_degree']
    model_head = args_dict['model_head']
    thread_num = args_dict['thread_num']
    data_type = args_dict['data_type']
    test_time = args_dict['test_time']

    if args_dict['model_path'] != "None":
        name = ""
        if args_dict['model_path'] == "GNMT":
            name = "gnmt"
        if args_dict['model_path'] == "NasNet":
            name = "nasnet"
        if args_dict['model_path'] == "PnasNet":
            name = "pnasnet"
        if args_dict['model_path'] == "NMT_4layers":
            name = "nmt"
        if args_dict['model_path'] == "InceptionV3":
            name = "inceptionv3"

        graph_path = "/home/FasterTransformer/examples/tensorflow/graphormer/datasets/" + args_dict['model_path'] + \
            "/" + name + "_32.pkl"

        with open(graph_path, 'rb') as file:
            data = pickle.load(file)

        nx_graph = data['nx_graph']
        max_num_nodes = len(nx_graph.nodes())


    tf_datatype = tf.float32
    np_datatype = np.float32
    atol_threshold = 3e-5        
    if data_type == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
        atol_threshold = 3e-2

    sequence_length = np.ones(batch_size) * max_num_nodes
    sequence_length = sequence_length.astype(np.int32)
    
    node_features = np.random.randn(batch_size, max_num_nodes)
    node_features = tf.convert_to_tensor(node_features, dtype=tf_datatype)
    distance_matrix = np.random.randn(batch_size, max_num_nodes, 1)
    distance_matrix = tf.convert_to_tensor(distance_matrix, dtype=tf_datatype)

    tf_graphormer_result = run_tf_graphormer(node_features, 
        distance_matrix, num_layer, hidden_dim, head_num, input_vocab_size, 
        max_num_nodes, dff=inter_size, d_sp_enc=sp_enc, float_type=tf_datatype)
    
    graphormer_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
    graphormer_variables_dict = {}
    for v in graphormer_variables:
        # print(v)
        graphormer_variables_dict[v.name] = v

    # print("#################################################################")
    # print("Graphormer Params:")
    # for v in graphormer_variables_dict:
    #     print(graphormer_variables_dict[v])
    # print("#################################################################")

    faster_graphormer = FasterGraphormer(num_layer, 
                                         hidden_dim, 
                                         head_num, 
                                         input_vocab_size, 
                                         graphormer_variables_dict, 
                                         size_per_head,
                                         dff=inter_size, 
                                         float_type=tf_datatype)
    faster_graphormer_result = faster_graphormer(node_features, distance_matrix)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    
    ite = 10
    a = 0
    b = 0    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        

        m_1 = time.time()
        for i in range(ite):
            m_a = time.time() * 1000
            tf_graphormer_result_val = sess.run(tf_graphormer_result)
            print(tf_graphormer_result_val.shape)
            m_b = time.time() * 1000
            if i == ite - 2:
                # print("Time {:6.2f}ms".format(m_b - m_a))
                a = m_b - m_a
        m_2 = time.time()
    
        # a = m_2 - m_1


    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        m_3 = round(time.time() * 1000)
        for i in range(ite):
            m_c = round(time.time() * 1000, 2)
            faster_graphormer_result_val = sess.run(faster_graphormer_result)
            m_d = round(time.time() * 1000, 2)
            if i == ite - 2:
                # print("Time {:6.2f}ms".format(m_d - m_c))
                b = m_d - m_c
        m_4 = round(time.time() * 1000)

    print("[INFO] batch_size {} max_seq_len {} precision {} {} layer TF-while-time     {:6.2f} ms ( {} iterations)".format(batch_size, max_num_nodes, data_type.upper(), num_layer, a, ite))
    print("[INFO] batch_size {} max_seq_len {} precision {} {} layer FT-OP-while-time  {:6.2f} ms ( {} iterations)".format(batch_size, max_num_nodes, data_type.upper(), num_layer, b * 3, ite))

    sys.stdout.flush()
    # return max_diff
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')   
    parser.add_argument('-l', '--num_layer', type=int, default=2, metavar='NUMBER',
                        help='number of layers (default: 12)')    
    parser.add_argument('-n', '--head_number', type=int, default=12, metavar='NUMBER',
                        help='head number (default: 12)')
    parser.add_argument('-size', '--size_per_head', type=int, default=64, metavar='NUMBER',
                        help='size per head (default: 64)')
    parser.add_argument('-vocab_size', '--input_vocab_size', type=int, default=100, metavar='NUMBER',
                        help='input vocab size (default: 100)')
    parser.add_argument('-inter_size', '--inter_size', type=int, default=512, metavar='NUMBER',
                        help='inter_size (default: 512)')
    parser.add_argument('-d_sp_enc', '--sp_enc', type=int, default=128, metavar='NUMBER',
                        help='Dimension of hidden layer to learn spatial encoding (default: 128)')
    parser.add_argument('-max_n', '--max_num_nodes', type=int, default=1990, metavar='NUMBER',
                        help='Maximum number of nodes per graph (default: 256)')
    parser.add_argument('-max_d', '--max_degree', type=int, default=256, metavar='NUMBER',
                        help='Maximum degree per graph (default: 256)')
    parser.add_argument('-model_h', '--model_head', type=str, default="VNode", metavar='STRING',
                        help='one of VNode, average. average pools the outputs over all positions, .\
                            VNode only selects the learnt output of the first position assumes a virtual node .\
                            as the first token, e.g. similar as in BERT')
    parser.add_argument('-thread_num', '--thread_num', type=int, default=1, metavar='int',
                        help='Testing multithread if thread_num > 1.') 
    parser.add_argument('-d', '--data_type', type=str, default="fp16", metavar='STRING',
                        help='data type (default: fp16)', choices=['fp32', 'fp16'])
    parser.add_argument('-time', '--test_time', type=int, default=0, metavar='BOOL',
                        help='test the time or not. (default: False (0)), True is 1.',
                        choices=[0, 1])
    parser.add_argument('--model_path', type=str, default="None", metavar='STRING',
                        help='model path')


    args = parser.parse_args()
    graphormer_example(vars(args))