import tensorflow as tf
import numpy as np
import torch
from torch_geometric.datasets import ZINC, pcqm4m
from ogb.lsc import PygPCQM4MDataset
from torch_geometric.data import DataLoader
from graphormer import Graphormer

tf.compat.v1.disable_eager_execution()

# 定义数据集的路径
path = './PCQM4M'

# 加载数据集
dataset = PygPCQM4MDataset(root=path)

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

# x_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None])
# edge_index_placeholder = tf.placeholder(dtype=tf.float32, shape=[2, None])

# graphormer = Graphormer(2, 512, 4, 100)
# out = graphormer(x_placeholder, edge_index_placeholder, False)


# 假设模型的调用方式如下
# out = graphormer(x_placeholder, edge_index_placeholder)

# 使用 TensorFlow session 来迭代数据集并执行模型
with tf.Session() as sess:
    try:
        while True:
            # 获取数据
            numpy_x, numpy_edge_index = sess.run(next_element)
            print(numpy_x.shape)
            print(numpy_edge_index)
            # 处理数据集

            # 执行模型
            # feed_dict = {
            #     x_placeholder: numpy_x,
            #     edge_index_placeholder: numpy_edge_index
            # }
            # output = sess.run(out, feed_dict=feed_dict)
            # print(output)

    except tf.errors.OutOfRangeError:
        print("数据集遍历完成。")