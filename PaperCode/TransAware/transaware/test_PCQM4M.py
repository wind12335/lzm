import tensorflow as tf
import numpy as np
import torch
from torch_geometric.datasets import ZINC, pcqm4m
from torch_geometric.loader import DataLoader
from shell.accuracy.graphormer import Graphormer

tf.compat.v1.disable_eager_execution()

# 定义数据集的路径
path = '/home/PaperCode/TransAware/PolicyExecTime/transaware/datasets/PCQM4M'

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

x_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, None])
y_true_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, None, 512])
edge_index_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, None, None])

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

graphormer = Graphormer(2, 512, 4, 100)
out = graphormer(x_placeholder, edge_index_placeholder, False)
graphormer = Graphormer(4, 512, 4, 100)
out_1 = graphormer(x_placeholder, edge_index_placeholder, False)
loss = tf.reduce_mean(tf.abs(out - y_true_placeholder))

# 假设模型的调用方式如下
# out = graphormer(x_placeholder, edge_index_placeholder)
init_op = tf.global_variables_initializer()
# 使用 TensorFlow session 来迭代数据集并执行模型
with tf.Session() as sess:
    sess.run(init_op)
    while True:
        # 获取数据
        numpy_x, numpy_edge_index = sess.run(next_element)
        print(numpy_x.shape)
        print(numpy_edge_index.shape)
        # print(numpy_x.shape)
        # print(numpy_edge_index)
        # 处理数据集
        i = numpy_x.reshape((numpy_x.shape[1], numpy_x.shape[0]))
        j = np.full((1, i.shape[1], i.shape[1]), -1)
        for idx in range(numpy_edge_index.shape[1]):
            j[0][numpy_edge_index[0][idx] - 1][numpy_edge_index[1][idx] - 1] = 1
        # 执行模型
        # print(i.shape)
        # print(j.shape)
        feed_dict = {
            x_placeholder: i,
            edge_index_placeholder: j
        }

        # x_output = sess.run(out, feed_dict=feed_dict)
        y_output = sess.run(out_1, feed_dict=feed_dict)

        loss_val = sess.run(loss, feed_dict={
            x_placeholder: i, 
            edge_index_placeholder: j, # 如果你的模型需要边缘索引信息
            y_true_placeholder: y_output
        })
        
        print(loss_val)