import tensorflow as tf
import numpy as np
import torch
from torch_geometric.datasets import ZINC, pcqm4m
from torch_geometric.loader import DataLoader
from shell.accuracy.graphormer import Graphormer

tf.compat.v1.disable_eager_execution()
# 
# 定义数据集的路径
path = '/home/PaperCode/TransAware/PolicyExecTime/transaware/datasets/ZINC-300'

# def graph_sage_layer(node_features, adj_matrix, output_size):
#     """
#     GraphSAGE层的简单实现，使用均值聚合。
#     """
#     with tf.compat.v1.variable_scope("graph_sage", reuse=tf.compat.v1.AUTO_REUSE):
#         # 权重矩阵
#         W = tf.compat.v1.get_variable("W", shape=[node_features.shape[-1], output_size],
#                             initializer=tf.keras.initializers.glorot_uniform())

#         # 聚合邻居特征
#         node_features = node_features.reshape((node_features.shape[0], 1))
#         neighbor_features = tf.linalg.matmul(adj_matrix, node_features)
#         neighbor_features = tf.math.divide(neighbor_features, tf.reduce_sum(adj_matrix, axis=-1, keepdims=True) + 1e-7)
        
#         # 合并自身特征和邻居特征
#         total_features = tf.concat([node_features, neighbor_features], axis=-1)
        
#         # 应用权重矩阵
#         total_features = tf.reshape(total_features, (total_features.shape[1], total_features.shape[0]))
#         node_features = tf.linalg.matmul(total_features, W)
        
#         return node_features

class GraphSAGELayer(tf.keras.layers.Layer):
    """GraphSAGE层的简化实现，使用均值聚合"""
    def __init__(self, output_size, activation=None, **kwargs):
        super(GraphSAGELayer, self).__init__(**kwargs)
        self.output_size = output_size
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        node_feature_shape = input_shape[0][-1]  # 修改点：获取节点特征维度的大小
        # 注意：确保node_feature_shape是一个整数
        if isinstance(node_feature_shape, tf.TensorShape):
            node_feature_shape = node_feature_shape.value
        self.weight = self.add_weight(name='weight',
                                      shape=(node_feature_shape * 2, self.output_size),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(GraphSAGELayer, self).build(input_shape)

    def call(self, inputs):
        node_features, adj_matrix = inputs

        # 聚合邻居特征
        neighbor_features = tf.matmul(adj_matrix, node_features)
        neighbor_features = neighbor_features / (tf.reduce_sum(adj_matrix, axis=-1, keepdims=True) + 1e-7)

        # 合并自身特征和邻居特征
        total_features = tf.concat([node_features, neighbor_features], axis=-1)
        
        # 应用权重矩阵并使用激活函数
        node_features = tf.matmul(total_features, self.weight)
        
        if self.activation is not None:
            node_features = self.activation(node_features)
        
        return node_features


class GraphSAGEModel(tf.keras.Model):
    """GraphSAGE模型，适用于处理图结构数据"""
    def __init__(self, num_layers, output_size, activation='relu', **kwargs):
        super(GraphSAGEModel, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.graphsage_layers = [GraphSAGELayer(output_size, activation=activation) for _ in range(num_layers)]

    def call(self, x, adj_matrix):
        # x, adj_matrix = inputs
        for layer in self.graphsage_layers:
            x = layer([x, adj_matrix])
        return x

# 加载数据集
dataset = pcqm4m(root=path, subset=True)

# 分割数据集为训练、验证、测试集
train_dataset = dataset[:10000]
val_dataset = dataset[10000:12000]
test_dataset = dataset[12000:]

# PyTorch DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 示例转换函数，将 PyTorch 张量转换为 NumPy 数组，不直接创建 TensorFlow 张量
def convert_to_numpy(data):
    # 转换节点特征为 NumPy 数组
    x = data.x.numpy()
    # 转换边缘索引为 NumPy 数组
    edge_index = data.edge_index.numpy()
    return x, edge_index

# 修改后的生成器函数，使用 NumPy 数组
def dataset_generator():
    for data in dataset:
        yield convert_to_numpy(data)

# 使用修改后的生成器函数创建 TensorFlow 数据集
tf_dataset = tf.data.Dataset.from_generator(
    dataset_generator,
    output_types=(tf.float32, tf.int64),
    # 需要根据你的数据调整这里的形状
    output_shapes=((None, None), (2, None))
)

# 创建 TensorFlow 数据集的迭代器
iterator = tf_dataset.make_one_shot_iterator()
next_element = iterator.get_next()
scale_factor = np.sqrt(300)
bias = 0.13

x_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, None, 1])
y_true_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, None, 64])
edge_index_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None])

model = GraphSAGEModel(num_layers=2, output_size=64)
out = model(x_placeholder, edge_index_placeholder)
model_1 = GraphSAGEModel(num_layers=4, output_size=64)
out_1 = model(x_placeholder, edge_index_placeholder)

# with tf.Session() as sess:
#     sess.run(init_op)
#     while True:
#         # 获取数据
#         numpy_x, numpy_edge_index = sess.run(next_element)
#         # print(numpy_x.shape)
#         # print(numpy_edge_index)
#         # 处理数据集
#         i = numpy_x.reshape((numpy_x.shape[1], numpy_x.shape[0]))
#         j = np.full((1, i.shape[1], i.shape[1]), -1)
#         for idx in range(numpy_edge_index.shape[1]):
#             j[0][numpy_edge_index[0][idx] - 1][numpy_edge_index[1][idx] - 1] = 1
#         # 执行模型
#         # print(i.shape)
#         # print(j.shape)
#         feed_dict = {
#             x_placeholder: i,
#             edge_index_placeholder: j
#         }

#         # x_output = sess.run(out, feed_dict=feed_dict)
#         y_output = sess.run(out_1, feed_dict=feed_dict)

#         loss_val = sess.run(loss, feed_dict={
#             x_placeholder: i, 
#             edge_index_placeholder: j, # 如果你的模型需要边缘索引信息
#             y_true_placeholder: y_output
#         })
        
#         print(loss_val)

# out = graph_sage_layer(x_placeholder, edge_index_placeholder, 512)
# out_1 = graph_sage_layer(x_placeholder, edge_index_placeholder, 512)
loss = tf.reduce_mean(tf.abs(out - y_true_placeholder))

# 假设模型的调用方式如下
# out = graphormer(x_placeholder, edge_index_placeholder)
init_op = tf.global_variables_initializer()
total_val = 0
# 使用 TensorFlow session 来迭代数据集并执行模型
epoch = 100
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(epoch):
        # 获取数据
        numpy_x, numpy_edge_index = sess.run(next_element)
        # print(numpy_x.shape)
        # print(numpy_edge_index)
        # 处理数据集
        i = numpy_x.reshape((1, numpy_x.shape[0], 1))
        i_r = np.random.uniform(size=(1, numpy_x.shape[0], 1))
        j = np.full((i.shape[1], i.shape[1]), -1)
        j_r = np.eye(i.shape[1])
        for idx in range(numpy_edge_index.shape[1]):
            j[numpy_edge_index[0][idx] - 1][numpy_edge_index[1][idx] - 1] = 1
        j = j.astype(np.float32)
        feed_dict = {x_placeholder : i, edge_index_placeholder : j}
        y_output = sess.run(out_1, feed_dict=feed_dict)
        # 执行模型
        # print(i.shape)
        # print(j.shape)
        
        
        
        # output = sess.run(out, feed_dict=feed_dict)
        # print(output.shape)

        x_output = sess.run(out, feed_dict=feed_dict)
        

        loss_val = sess.run(loss, feed_dict={
            x_placeholder: i_r, 
            edge_index_placeholder: j_r, # 如果你的模型需要边缘索引信息
            y_true_placeholder: y_output
        })

        total_val += loss_val
        
print(total_val / epoch)
