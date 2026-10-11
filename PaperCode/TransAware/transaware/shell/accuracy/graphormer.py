import random
import datetime

try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
import numpy as np
import networkx as nx

# __doc__ = "Implementation of GraphSAGE-like algorithm for embedding to be used in the RL policy."

INIT_SCALE = 1.0

class FNN:
    """
    hidden_layer_sizes: list of hidden layer sizes
    out_size: Size of the last softmax layer
    """

    def __init__(self, inp_size, hidden_layer_sizes, out_size, name, dtype=tf.float32):
        layers = []
        sizes = [inp_size] + hidden_layer_sizes + [out_size]
        for i in range(len(sizes) - 1):
            w = self.glorot(shape=(sizes[i], sizes[i + 1]), scope=name, dtype=dtype)
            b = self.zero_init(shape=(1, sizes[i + 1]), scope=name, dtype=dtype)
            layers.append([w, b])

        self.layers = layers

    def build(self, inp_tensor):
      out = inp_tensor
      for idx, [w, b] in enumerate(self.layers):
        out = tf.matmul(out, w) + b
        if idx != len(self.layers) - 1:
          out = tf.nn.relu(out)
      return out

    def zero_init(self, shape, scope="default", dtype=tf.float32):
        with tf.variable_scope(scope):
            init = np.zeros(shape)
        return tf.Variable(init, dtype=dtype)

    def glorot(self, shape, scope="default", dtype=tf.float32):
        # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
        with tf.variable_scope(scope):
            init_range = np.sqrt(float(6.0 * INIT_SCALE) / int(shape[0] + shape[1]))
            init = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=dtype)
        return tf.Variable(init, dtype=dtype)

class Classifier:
    def __init__(self, input_size, hidden_layer_sizes, out_size, dtype=tf.float32):
        self.nn = FNN(input_size, hidden_layer_sizes, out_size, "classifier", dtype=dtype)

    def build(self, input_tensor):
        return self.nn.build(input_tensor)

class Aggregator:
    """
    N is the max number of children to be aggregated
    d is the degree of embeddings
    d1 is the degree of embedding transformation
    d2 is degree of aggregation
    """

    def __init__(self, d, d1=None, d2=None, use_mask=True, normalize_aggs=False, dtype=tf.float32):
        self.d = d
        self.d1 = d1
        self.d2 = d2
        self.normalize_aggs = normalize_aggs

        if d1 is None:
            d1 = self.d1 = d
        if d2 is None:
            d2 = self.d2 = d

        self.use_mask = use_mask

        if self.use_mask:
            self.Mask = tf.placeholder(dtype, shape=(None, None))

        self.f = FNN(self.d, [self.d], self.d1, "f", dtype=dtype)
        self.g = FNN(self.d1, [self.d1], self.d2, "g", dtype=dtype)

    def build(self, input_tensor, mask=None):
        """
        Build the Aggregator.
        """
        f_out = tf.nn.relu(self.f.build(input_tensor))
        if self.use_mask or mask is not None:
            if mask is None:
                mask = self.Mask
            g = tf.matmul(mask, f_out)
            if self.normalize_aggs:
                d = tf.cond(tf.reduce_sum(mask) > 0, lambda: tf.reduce_sum(mask), lambda: 1.0)
                g /= d
        else:
            g = tf.reduce_sum(f_out, 0, keepdims=True)
        g = tf.nn.relu(self.g.build(g))
        return g

class Messenger(object):
  def __init__(self, embedding_size):
    # forward pass
    with tf.name_scope('FPA'):
      # self.fpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
      self.fpa = Aggregator(embedding_size, 
                            embedding_size, 
                            embedding_size, 
                            False,
                            False)
    with tf.name_scope('BPA'):
      self.bpa = Aggregator(embedding_size, 
                            embedding_size, 
                            embedding_size, 
                            False, 
                            False)
      # self.bpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
    with tf.name_scope('node_transform'):
      self.node_transform = FNN(embedding_size, 
                                [embedding_size, embedding_size],
                                embedding_size, 'fnn', dtype=tf.float32)

  def build(self, G, node_order, input_tensor, bs=1):
    try:
      self_trans = self.node_transform.build(input_tensor)
      def message_pass(nodes, messages_from, agg):
        node2emb = {}

        for n in nodes:
          msgs = [node2emb[pred] for pred in messages_from(n)]

          node2emb[n] = tf.expand_dims(self_trans[G.get_idx(n), :],
                                        axis=0)

          if len(msgs) > 0:
            t = tf.concat(msgs, axis=0)
            inp = agg.build(t)
            node2emb[n] += inp

        return tf.concat([node2emb[n] for n in G.nodes()], 
                            axis=0)

      out_fpa = message_pass(node_order, G.predecessors, self.fpa)
      out_bpa = message_pass(reversed(node_order), G.neighbors, self.bpa)

      out = tf.concat([out_fpa, out_bpa], axis=-1)
      
      return out
    except Exception:
      import my_utils; my_utils.PrintException()
      import pdb; pdb.set_trace()

class RadioMessenger(Messenger):
    def __init__(self, config_params, embedding_size, progressive_graph):
        Messenger.__init__(self, embedding_size)
        self.config_params = config_params
        self.k = config_params['radial_mp']
        self.progressive_graph = progressive_graph
        self.nx_nodes = progressive_graph.node_traversal_order
        self.nx_graph = progressive_graph.nx_graph
        self.bs = 1
        self.dtype = tf.float32

    def build(self):
        f_adj = self.progressive_graph.fadj
        b_adj = self.progressive_graph.badj

        print(type(self.progressive_graph))

        input_tensor = tf.placeholder(tf.float32)
        input_tensor = tf.cast(input_tensor, dtype=tf.float32)
        input_tensor = tf.reshape(input_tensor, [-1, tf.shape(input_tensor)[-1]])
        self_trans = self.node_transform.build(input_tensor)

        def f(adj):
            n = adj.shape[0]
            t = np.zeros([self.bs * n] * 2, dtype=np.float32)
            for i in range(self.bs):
                t[i * n: (i + 1) * n, i * n: (i + 1) * n] = adj

            return t

        f_adj = f(f_adj)
        b_adj = f(b_adj)

        with tf.variable_scope('Forward_pass'):
            # out_fpa = message_pass(f_adj, self.fpa)
            out_fpa = self.message_pass(f_adj, self_trans, self.fpa)
        with tf.variable_scope('Backward_pass'):
            out_bpa = self.message_pass(b_adj, self_trans, self.bpa)

        out = tf.concat([out_fpa, out_bpa], axis=-1)
        out = tf.cast(out, tf.float32)

        return out

        # self_trans = tf.Print(self_trans, [self_trans], message='self_trans: ', summarize=100000000)

        # def message_pass(adj, agg):
        #     sink_mask = (np.sum(adj, axis=-1) > 0)
        #     # sink_mask = np.float32(sink_mask)
        #     # sink_mask = np.float16(sink_mask)
        #     # adj = np.float16(adj)
        #     sink_mask = tf.cast(sink_mask, self.dtype)
        #     adj = tf.cast(adj, self.dtype)

        #     x = self_trans
        #     for _ in range(self.k):
        #         # x = tf.Print(x, [x], message='pre agg: x', summarize=1000)
        #         x = agg.build(x, mask=adj)
        #         # x = tf.Print(x, [x], message='x', summarize=1000)
        #         x = sink_mask * tf.transpose(x)
        #         x = tf.transpose(x)
        #         x += self_trans

        #     return x

    def message_pass(self, adj, self_trans, agg):
      """
      Message pass.
      """
      sink_mask = np.sum(adj, axis=-1) > 0
      sink_mask = tf.cast(sink_mask, tf.float32)
      adj = tf.cast(adj, tf.float32)
      x = self_trans

      for _ in range(self.config_params["radial_mp"]):
        x = agg.build(x, mask=adj)
        x = sink_mask * tf.transpose(x)
        x = tf.transpose(x)
        x += self_trans

      return x

    def mess_build(self, input_tensor):
        return Messenger.build(self, self.nx_graph, self.nx_nodes, input_tensor)

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

    def __init__(self, max_degree, d_model):
        super(CentralityEncoding, self).__init__()
        self.centr_embedding = tf.keras.layers.Embedding(max_degree, d_model)

    def centrality(self, distances):
        centrality = tf.cast(tf.math.equal(tf.math.abs(distances), 1), tf.float32) # abs --> connections to vnode count
        centrality = tf.math.reduce_sum(centrality, axis=-1, keepdims=False)
        return tf.cast(centrality, tf.float32)

    def call(self, distances):
        centrality = self.centrality(distances)
        centrality_encoding = self.centr_embedding(centrality)
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
    def __init__(self, d_sp_enc=16, activation='relu'):
        super(SpatialEncoding, self).__init__()
        self.d_sp_enc = d_sp_enc
        self.activation = activation
        self.dense1 = tf.keras.layers.Dense(d_sp_enc, activation=activation)
        self.dense2 = tf.keras.layers.Dense(1, activation=activation)

    def call(self, distances):
        expanded_inputs = tf.expand_dims(distances, axis=-1)
        outputs = self.dense1(expanded_inputs)
        outputs = self.dense2(outputs)
        spatial_encoding = tf.squeeze(outputs, axis=-1)
        return spatial_encoding


def scaled_dot_product_attention(q, k, v, min_distance_matrix, spatial_encoding, num_heads, mask=None, d_sp_enc=16, sp_enc_activation='relu'):
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

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
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
  
    def __init__(self, d_model, num_heads, d_sp_enc, sp_enc_activation):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.spatial_encoding = SpatialEncoding(self.d_sp_enc, self.sp_enc_activation)

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

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
            q, k, v, min_distance_matrix, self.spatial_encoding, self.num_heads, mask, self.d_sp_enc, self.sp_enc_activation)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """Feedforward Network.

    Args:
        d_model: Size of the hidden dimension of the model.
        dff: Intermediate size of the model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


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
    def __init__(self, d_model, num_heads, dff, rate=0.1, d_sp_enc=128, sp_enc_activation='relu'):
        super(GraphormerBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, d_sp_enc, sp_enc_activation)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, min_distance_matrix):

        residual = x
        x_norm = self.layernorm1(x) # NEW: Pre-LayerNorm
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, min_distance_matrix, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = residual + attn_output  # (batch_size, input_seq_len, d_model)

        residual = out1
        out1_norm = self.layernorm2(out1) # NEW: Pre-LayerNorm
        ffn_output = self.ffn(out1_norm)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = residual + ffn_output  # (batch_size, input_seq_len, d_model)

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
                top_activation='relu', initial_bias=0, concat_n_layers=1):
        super(Graphormer, self).__init__()

        self.d_model = d_model
        self.max_num_nodes = max_num_nodes
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.centr_encoding = CentralityEncoding(max_degree, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.graphormer_layers = [GraphormerBlock(d_model, num_heads, dff, dropout, d_sp_enc, sp_enc_activation)
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
        return tf.cast(tf.math.equal(nodes, 0), tf.float32)

    def call(self, node_features, distance_matrix, training):
        # All inputs passed in the first argument (tupel or list)
        # node_features, distance_matrix = inputs

        # Generate masks
        mask = self.create_padding_mask(node_features)
        
        # mask transformation to add extra dimensions to add the padding to the attention logits.
        attention_mask = mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
        # attention_mask = mask[:, tf.newaxis, :]

        # Embedding
        embed = self.embedding(node_features)  # (batch_size, input_seq_len, d_model)
        embed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embed += self.centr_encoding(distance_matrix)
        out = embed
        # out = self.dropout(embed, training=training)

        hidden_states = [] # List to hold hidden states per graphormer block

        # Graphormer layers
        for i in range(self.num_layers):
            out = self.graphormer_layers[i](out, training, attention_mask, distance_matrix)  # (batch_size, inp_seq_len, d_model)
        if self.concat_n_layers > 1:
            hidden_states.append(out)

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
        
        return out
