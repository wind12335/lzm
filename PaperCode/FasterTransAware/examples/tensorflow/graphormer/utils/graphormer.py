import tensorflow as tf
import os
import numpy as np
import time

class CentralityEncoding(tf.keras.layers.Layer):
    """CentralityEncoding layer.
    The Centrality Encoding measures the importance of each node in the graph.
    Here, importance is determined by the degree centrality, i.e. the degree of
    each node. Degrees are projected to the hidden size of the model via learnt 
    embeddings.

    Args:
        max_degree: Maximum degree of the input graphs
        d_model: Size of the hidden dimension of the model
        
    Call arguments:
        distances: `Tensor` of shape `(B, N, N)`, where `B` is the batch_size and `N` is
        number of nodes. Entries of 1 (or -1) indicate direct neighbors and 
        contribute to the centrality encoding of that node. 
    Returns:
        centrality_encoding: The result of the computation, of shape `(B, N, D)`,
        where `N` is for the number of nodes and `D` is the hidden dimension size 
        of the model. 
    """

    def __init__(self, max_degree, d_model, float_type=tf.float16):
        super(CentralityEncoding, self).__init__()
        self.float_type = float_type
        self.center_embedding = tf.keras.layers.Embedding(max_degree, d_model, dtype=self.float_type)
        
    def centrality(self, distances):
        centrality = tf.cast(tf.math.equal(tf.math.abs(distances), 1), dtype=self.float_type) # abs --> connections to vnode count
        centrality = tf.math.reduce_sum(centrality, axis=-1, keepdims=False)
        return tf.cast(centrality, dtype=self.float_type)

    def call(self, distances):
        centrality = self.centrality(distances)
        centrality_encoding = self.center_embedding(centrality)
        return centrality_encoding

class FasterGraphormer(tf.keras.Model):
    """Graphormer model.
    Implements a Graphormer model following Ying et al. (2021). *Do Transformers Really
    Perform Bad for Graph Representation?* arXiv:2106.05234. Inputs represent undirected
    graphs. Inputs all need to be padded to equal lengths, node_id 0 is assumed as the
    padding token.

    Args:
        num_layers: Number of Graphormer blocks.
        d_model: Size of the hidden dimension of the model.
        num_heads: Number of attention heads. d_model must be divisible by num_heads.
        input_vocab_size: Maximum number of unique node ids across all inputs, should be 
        set large enough.
        dropout: Dropout rate in the Graphormer blocks. Default: 0.1.
        dff: Intermediate size of the feedforward networks in the Graphormer blocks. Default: 512.
        d_sp_enc: Dimension of hidden layer to learn spatial encoding. Default: 128.
        sp_enc_activation: Keras activation function for spatial encoding. Default: 'relu'.
        max_num_nodes: Maximum number of nodes per graph. Default: 256.
        max_degree: Maximum degree per graph. Default: 256.
        model_head: one of 'VNode', 'average'. 'average' pools the outputs over all positions, 
        'VNode' only selects the learnt output of the first position (assumes a virtual node
        as the first token, e.g. similar as in BERT)
        top_dropout: Dropout rate in the classification head. Default: 0.1.
        d_top: Hidden dimension of the classification head. Default: 64.
        top_activation: Activation function in the classification head. Default: 'relu'.
        initial_bias: Initial bias of the classifier. Default: 0.
        concat_n_layers: Int, Number of Graphormer layers of which outputs to concatenate to
        feed into the classification head. Default: 1.
        
    Call arguments:
        inputs: tuple/list of node_features, distance_matrix
            node_features: (batch_size, max_num_nodes) tensor of node features
            distance_matrix: (batch_size, max_num_nodes, max_num_nodes) pairwise minimum distances
            between nodes
    Returns:
        centrality_encoding: The result of the computation, of shape `(B, N, D)`,
        where `N` is for the number of nodes and `D` is the hidden dimension size 
        of the model. 
    """
  
    def __init__(self, num_layers, d_model, num_heads, input_vocab_size, graphormer_variables_dict,
                size_per_head, dropout=0.1, dff=512, d_sp_enc=128, sp_enc_activation='gelu', 
                max_num_nodes=256, max_degree=256, model_head='VNode', top_dropout=0.1, d_top=64,
                top_activation='relu', initial_bias=0, concat_n_layers=1, float_type=tf.float16):
        super(FasterGraphormer, self).__init__()

        self.d_model = d_model
        self.max_num_nodes = max_num_nodes
        self.num_layers = num_layers
        self.float_type = float_type
        self.transformer_op_module = tf.load_op_library(os.path.join('./lib/libtf_graphormer.so'))
        self.graphormer_variables_dict = graphormer_variables_dict
        self.size_per_head = size_per_head
        self.dff = dff
        self.head_num = num_heads

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, dtype=self.float_type)
        self.center_encoding = CentralityEncoding(max_degree, d_model, self.float_type)

        # self.graphormer_layers = [GraphormerBlock(d_model, num_heads, dff, dropout, d_sp_enc, sp_enc_activation)
        #                 for _ in range(num_layers)]

        # self.concat_n_layers = concat_n_layers
        # self.model_head = model_head
        # self.top_dropout = tf.keras.layers.Dropout(top_dropout)
        # self.d_top = d_top
        # self.top_activation = top_activation
        # self.top_dense = tf.keras.layers.Dense(self.d_top, activation=self.top_activation, name='top_dense_1')
        # self.output_bias = tf.keras.initializers.Constant(initial_bias)
        # self.head = tf.keras.layers.Dense(1, activation='sigmoid', name='binary_classifier' 
        #                             ,bias_initializer=self.output_bias)
        
    def create_padding_mask(self, nodes):
        return tf.cast(tf.math.equal(nodes, 0), dtype=self.float_type)

    def call(self, node_features, distance_matrix):
        # m_1 = round(time.time() * 1000)
        # # Generate masks
        mask = self.create_padding_mask(node_features)
        
        # # mask transformation to add extra dimensions to add the padding to the attention logits.
        attention_mask = mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
        # # attention_mask = mask[:, tf.newaxis, :]

        # # Embedding
        embed = self.embedding(node_features)  # (batch_size, input_seq_len, d_model)
        embed *= tf.math.sqrt(tf.cast(self.d_model, dtype=self.float_type))
        embed += self.center_encoding(distance_matrix)

        max_num_nodes = np.array([self.max_num_nodes]).astype(np.int32)
        output = self.transformer_op_module.graphormer(
                    embed,
                    embed,
                    max_num_nodes,
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/query/kernel:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/query/bias:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/key/kernel:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/key/bias:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/value/kernel:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/value/bias:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/output/dense/kernel:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/output/dense/bias:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/AttentionLayerNorm/beta:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/AttentionLayerNorm/gamma:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/FFN/dense1/kernel:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/FFN/dense1/bias:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/FFN/dense2/kernel:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/FFN/dense2/bias:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/FFNLayerNorm/beta:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/FFNLayerNorm/gamma:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/spatial_encoding/dense1/kernel:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/spatial_encoding/dense1/bias:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/spatial_encoding/dense2/kernel:0' % id] for id in range(self.num_layers)],
                    [self.graphormer_variables_dict['graphormer/layer_%d/graphormer_block/Attention/spatial_encoding/dense2/bias:0' % id] for id in range(self.num_layers)],
                    head_num = self.head_num, size_per_head = self.size_per_head,
                    inter_size = self.dff, num_layer = self.num_layers, q_scaling = 1.0)
        
        # print(output.shape)
        # m_2 = round(time.time() * 1000)
        
        # print("FasterGraphormerTime: {}ms/iter".format(m_2 - m_1))
        # hidden_states = [] # List to hold hidden states per graphormer block

        # # Graphormer layers
        # for i in range(self.num_layers):
        #     out = self.graphormer_layers[i](out, training, attention_mask, distance_matrix)  # (batch_size, inp_seq_len, d_model)
        # if self.concat_n_layers > 1:
        #     hidden_states.append(out)

        # # Model head
        # if self.model_head not in ['VNode', 'average']:
        #     raise NameError('model_head must be in ["VNode", "average"]')

        # # Concatenate hidden states
        # if self.concat_n_layers > 1:
        #     out = tf.keras.layers.Concatenate()(tuple([hidden_states[i] for i in range(-self.concat_n_layers, 0)]))
        
        # # Virtual Node
        # if self.model_head == 'VNode':
        #     out = out[:, 0, :]

        # # Average Pooling
        # if self.model_head == 'average':
        #     out = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(out, mask)
        
        # out = self.top_dropout(out)
        # out = self.top_dense(out)
        
        # out = self.head(out)
        
        return output