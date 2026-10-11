import tensorflow as tf
import numpy as np
import time
import pickle
# from layers import CentralityEncoding, GraphormerBlock


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

    def __init__(self, max_degree, d_model, name="centrality_encoding", float_type=tf.float16):
        super(CentralityEncoding, self).__init__(name=name, dtype=float_type)
        self.float_type = float_type
        self.center_embedding = tf.keras.layers.Embedding(max_degree, d_model, name="embedding", dtype=self.float_type)

    def centrality(self, distances):
        centrality = tf.cast(tf.math.equal(tf.math.abs(distances), 1), dtype=self.float_type) # abs --> connections to vnode count
        centrality = tf.math.reduce_sum(centrality, axis=-1, keepdims=False)
        return tf.cast(centrality, dtype=self.float_type)

    def call(self, distances):
        centrality = self.centrality(distances)
        centrality_encoding = self.center_embedding(centrality)
        return centrality_encoding


class SpatialEncoding(tf.keras.layers.Layer):
    """SpatialEncoding layer.
    The Spatial Encoding assigns each pairwise distance between nodes a learnable scalar
    which serve as a bias term in the self-attention module.

    Args:
        d_sp_enc: Degree of the intermediate layer to learn scalars for each distance. Default: 16.
        activation: Activation function for the mapping from distances to bias terms. Default: 'relu'.
        
    Call arguments:
        distances: `Tensor` of shape `(B, N, N)`, where `B` is the batch_size and `N` is
        number of nodes. Entries represent the pairwise mininum distances of nodes in the
        graph. Special values should be assigned to nodes which are not connected and to
        connections to the virtual node. 
    Returns:
        spatial_encoding: The result of the computation, of shape `(B, N, N)`,
        where `N` is for the number of nodes. 
    """
    def __init__(self, d_sp_enc=16, activation='relu', name="spatial_encoding", float_type=tf.float16):
        super(SpatialEncoding, self).__init__(name=name, dtype=float_type)
        self.float_type = float_type
        self.d_sp_enc = d_sp_enc
        self.activation = activation
        self.dense1 = tf.keras.layers.Dense(d_sp_enc, activation=activation, name="dense1", dtype=self.float_type)
        self.dense2 = tf.keras.layers.Dense(1, activation=activation, name="dense2", dtype=self.float_type)

    def call(self, distances):
        expanded_inputs = tf.expand_dims(distances, axis=-1)
        outputs = self.dense1(expanded_inputs)
        outputs = self.dense2(outputs)
        spatial_encoding = tf.squeeze(outputs, axis=-1)
        return spatial_encoding

# class ScaledDotProductAttention(tf.keras.layers.Layer):
#     def __init__():


def scaled_dot_product_attention(q, k, v, min_distance_matrix, spatial_encoding, num_heads, mask=None, d_sp_enc=16, sp_enc_activation='relu', float_type=tf.float16):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask must be broadcastable for addition. Spatial encoding bias is added
    to the query-key product matrix.

    Args:
        q: query shape == (..., seq_len_q, hidden_dim)
        k: key shape == (..., seq_len_k, hidden_dim)
        v: value shape == (..., seq_len_v, hidden_dim)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
        min_distance_matrix: Float tensor of pairwise minimum distances, 
            shape == (..., seq_len, seq_len)
        d_sp_enc: Dimension of hidden layer to learn spatial encoding
        sp_enc_activation: Keras activation function for spatial encoding

    Returns:
        output, attention_weights
    """

    # print(q.shape)
    # print(k.shape)
    # t1 = time.time() * 1000
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # t2 = time.time() * 1000

    # print(t2 - t1)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], dtype=float_type)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Spatial encodings
    spatial_encoding_bias = spatial_encoding(min_distance_matrix)
    spatial_encoding_bias = tf.expand_dims(spatial_encoding_bias, axis=1)
    spatial_encoding_bias = tf.repeat(spatial_encoding_bias, repeats=num_heads, axis=1)
    scaled_attention_logits += spatial_encoding_bias

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention layer.
    The implementation largely follows the implementation of the original transformer
    architecture with the addition of spatial encodings.

    Args:
        d_model: Size of the hidden dimension of the model
        num_heads: Number of attention heads. d_model must be divisible by num_heads.
        d_sp_enc: Dimension of hidden layer to learn spatial encoding.
        sp_enc_activation: Keras activation function for spatial encoding.
        
    Call arguments:
        q: query shape == (..., seq_len_q, hidden_dim)
        k: key shape == (..., seq_len_k, hidden_dim)
        v: value shape == (..., seq_len_v, hidden_dim)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
        min_distance_matrix: Float tensor of pairwise minimum distances, 
            shape == (..., seq_len, seq_len)

    Returns:
        output, attention_weights
    """
  
    def __init__(self, d_model, num_heads, d_sp_enc, sp_enc_activation, name="Attention", float_type=tf.float16):
        super(MultiHeadAttention, self).__init__(name=name, dtype=float_type)
        self.float_type = float_type
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.spatial_encoding = SpatialEncoding(self.d_sp_enc, self.sp_enc_activation, float_type=self.float_type)

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name="query", dtype=self.float_type)
        self.wk = tf.keras.layers.Dense(d_model, name="key", dtype=self.float_type)
        self.wv = tf.keras.layers.Dense(d_model, name="value", dtype=self.float_type)

        self.dense = tf.keras.layers.Dense(d_model, name="dense", dtype=self.float_type)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, min_distance_matrix, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, min_distance_matrix, self.spatial_encoding, self.num_heads, mask, self.d_sp_enc, self.sp_enc_activation, float_type=self.float_type)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        with tf.variable_scope("output"):
            output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class FeedforwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, name="FFN"):
        super(FeedforwardNetwork, self).__init__(name=name)
        self.ffn = [
            tf.keras.layers.Dense(dff, activation='relu', name="dense1"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model, name="dense2")  # (batch_size, seq_len, d_model)
        ]
    
    def call(self, x):
        for network in self.ffn:
            x = network(x)
        return x

class GraphormerBlock(tf.keras.layers.Layer):
    """Graphormer layer.
    Corresponds to a transformer block. Pre-layer normalization is used in 
    accordance with the paper, i.e. before the attention module and the 
    feed-forwards network respectively.

    Args:
        d_model: Size of the hidden dimension of the model.
        num_heads: Number of attention heads. d_model must be divisible by num_heads.
        rate: Dropout rate. Default: 0.1.
        d_sp_enc: Dimension of hidden layer to learn spatial encoding. Default: 128.
        sp_enc_activation: Keras activation function for spatial encoding. Default: 'relu'
        
    Call arguments:
        training: Bool indicating if training or inference.
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
        min_distance_matrix: Float tensor of pairwise minimum distances, 
            shape == (..., seq_len, seq_len)
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1, d_sp_enc=128, sp_enc_activation='relu', name="graphormer_block", float_type=tf.float16):
        super(GraphormerBlock, self).__init__(name=name, dtype=float_type)

        self.float_type = float_type
        self.mha = MultiHeadAttention(d_model, num_heads, d_sp_enc, sp_enc_activation, float_type=self.float_type)
        self.ffn = FeedforwardNetwork(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="AttentionLayerNorm")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="FFNLayerNorm")

    def call(self, x, mask, min_distance_matrix):
        residual = x
        attn_output, _ = self.mha(x, x, x, min_distance_matrix, mask)  # (batch_size, input_seq_len, d_model)
        x_norm = self.layernorm1(attn_output)        
        out1 = residual + x_norm  # (batch_size, input_seq_len, d_model)

        residual = out1
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        out1_norm = self.layernorm2(ffn_output)
        out2 = residual + out1_norm  # (batch_size, input_seq_len, d_model)

        return out2

class Graphormer(tf.keras.Model):
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
  
    def __init__(self, num_layers, d_model, num_heads, input_vocab_size,
                dropout=0.1, dff=512, d_sp_enc=128, sp_enc_activation='relu', 
                max_num_nodes=256, max_degree=256, model_head='VNode', top_dropout=0.1, d_top=64,
                top_activation='relu', initial_bias=0, concat_n_layers=1, name="graphormer",float_type=tf.float16):
        super(Graphormer, self).__init__(name=name, dtype=float_type)

        self.d_model = d_model
        self.max_num_nodes = max_num_nodes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.float_type = float_type

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, dtype=self.float_type)
        self.center_encoding = CentralityEncoding(max_degree, d_model, float_type=self.float_type)
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.graphormer_layers = [GraphormerBlock(d_model, num_heads, dff, dropout, d_sp_enc, sp_enc_activation, float_type=self.float_type)
                        for _ in range(num_layers)]

        self.concat_n_layers = concat_n_layers
        self.model_head = model_head
        self.top_dropout = tf.keras.layers.Dropout(top_dropout)
        self.d_top = d_top
        self.top_activation = top_activation
        self.top_dense = tf.keras.layers.Dense(self.d_top, activation=self.top_activation, name='top_dense_1')
        self.output_bias = tf.keras.initializers.Constant(initial_bias)
        self.head = tf.keras.layers.Dense(1, activation='sigmoid', name='binary_classifier' 
                                    ,bias_initializer=self.output_bias)
        
    def create_padding_mask(self, nodes):
        return tf.cast(tf.math.equal(nodes, 0), dtype=self.float_type)

    def call(self, node_features, distance_matrix):
        
        # Generate masks
        mask = self.create_padding_mask(node_features)
        
        # mask transformation to add extra dimensions to add the padding to the attention logits.
        attention_mask = mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

        # Embedding
        embed = self.embedding(node_features)  # (batch_size, input_seq_len, d_model)
        embed *= tf.math.sqrt(tf.cast(self.d_model, dtype=self.float_type))
        embed += self.center_encoding(distance_matrix) 

        # hidden_states = [] # List to hold hidden states per graphormer block

        # Graphormer layers
        out = embed
        
        # m_1 = 0
        # m_2 = 0
        for i in range(self.num_layers):
            # m_1 = round(time.time() * 1000)
            with tf.variable_scope("layer_%d" % i, reuse=tf.AUTO_REUSE):
                out = self.graphormer_layers[i](out, attention_mask, distance_matrix)  # (batch_size, inp_seq_len, d_model)
            # m_2 = round(time.time() * 1000)
            # print("GraphormerTime: {}ms/iter".format(m_2 - m_1))
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
        # print(out.shape)
        return out

def run_tf_graphormer(node_features,
                      distance_matrix,
                      num_layers, 
                      d_model, 
                      num_heads, 
                      input_vocab_size,
                      num_nodes,
                      dropout=0.1, 
                      dff=512, 
                      d_sp_enc=128, 
                      sp_enc_activation='relu',
                      max_degree=256, 
                      model_head='VNode', 
                      top_dropout=0.1, 
                      d_top=64,
                      top_activation='relu', 
                      initial_bias=0, 
                      concat_n_layers=1,
                      float_type=tf.float16):
    test = Graphormer(num_layers, d_model, num_heads, input_vocab_size, float_type=float_type)

    # # 提取模型参数
    # weights = test.get_weights()  # 返回模型参数的numpy数组列表

    # # 使用numpy保存参数
    # np.save('/home/FasterTransformer/examples/tensorflow/graphormer/params/model_weights.npy', weights)
    # graphormer_variables_dict = {v.name: v.numpy() for v in test.weights}

    # with open('/home/FasterTransformer/examples/tensorflow/graphormer/params/params-16fp-192nodes.pkl', 'wb') as f:
    #     pickle.dump(graphormer_variables_dict, f)

    result = test(node_features, distance_matrix)

    return result
