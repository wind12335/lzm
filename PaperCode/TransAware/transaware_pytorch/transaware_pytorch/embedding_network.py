"""TransAware 的 Graphormer 节点编码网络 —— PyTorch 移植版。

对应原 TF 实现 ``embedding_network.py``（Graphormer, Ying et al. 2021），
作为 RL 放置策略的节点编码主干。本文件只移植 ``search_policy()`` 主链路
用到的 Graphormer 编码栈；原文件顶部与 ``progressive_nn`` 配套的
``FNN/Classifier/Aggregator/Messenger/RadioMessenger`` 未使用，故不移植。

与 TF 版的逐层对应关系见项目 README。
"""

import math

import torch
import torch.nn as nn


class CentralityEncoding(nn.Module):
    """中心性编码层。

    节点的重要性由度中心性衡量（一跳邻居数量），度被投影到隐藏维度。
    ``max_degree`` 为度嵌入表大小，``d_model`` 为隐藏维度。
    输入 ``distances``: (B, N, N)，``|距离|==1`` 记为直接邻居。
    """

    def __init__(self, max_degree, d_model):
        super().__init__()
        self.centr_embedding = nn.Embedding(max_degree, d_model)

    @staticmethod
    def centrality(distances):
        # abs()==1 --> 直接邻居计数（含反向边）
        centrality = (distances.abs() == 1).float().sum(dim=-1)
        return centrality

    def forward(self, distances):
        centrality = self.centrality(distances).long()
        # 原 TF 实现的"距离矩阵"把非边填成 -1，导致 abs==1 的计数（即"度"）在
        # 真实大图上远超 max_degree，TF Keras Embedding 会越界报错。为保持可运行，
        # 将度索引 clamp 到嵌入表范围内（语义对应原实现的退化行为，详见 README）。
        centrality = centrality.clamp(0, self.centr_embedding.num_embeddings - 1)
        centrality_encoding = self.centr_embedding(centrality)
        return centrality_encoding


class SpatialEncoding(nn.Module):
    """空间编码层。

    为每对节点间的距离学习一个可学习的标量，作为自注意力的偏置项。
    TF: ``Dense(16, relu) -> Dense(1, relu)``，作用于最后一个维度。
    """

    def __init__(self, d_sp_enc=16, activation='relu'):
        super().__init__()
        self.d_sp_enc = d_sp_enc
        self.activation = activation
        self.dense1 = nn.Linear(1, d_sp_enc)
        self.dense2 = nn.Linear(d_sp_enc, 1)

    def forward(self, distances):
        expanded_inputs = distances.unsqueeze(-1)  # (B, N, N, 1)
        outputs = torch.relu(self.dense1(expanded_inputs))
        outputs = torch.relu(self.dense2(outputs))
        spatial_encoding = outputs.squeeze(-1)  # (B, N, N)
        return spatial_encoding


def scaled_dot_product_attention(q, k, v, min_distance_matrix, spatial_encoding,
                                 num_heads, mask=None, d_sp_enc=16,
                                 sp_enc_activation='relu'):
    """缩放点积注意力 + 空间编码偏置 + padding mask。

    与原 TF 实现逐行对应：``q@k^T / sqrt(dk)``，加上按 head 复制的空间编码
    偏置，加上 ``mask * -1e9``，softmax 后加权 v。
    """
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (B, H, seq_q, seq_k)

    # scale matmul_qk
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    # 空间编码
    spatial_encoding_bias = spatial_encoding(min_distance_matrix)  # (B, N, N)
    spatial_encoding_bias = spatial_encoding_bias.unsqueeze(1)  # (B, 1, N, N)
    spatial_encoding_bias = spatial_encoding_bias.expand(
        -1, num_heads, -1, -1)  # (B, H, N, N)
    scaled_attention_logits += spatial_encoding_bias

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力层。

    在原始 Transformer 多头注意力的基础上加入空间编码。``d_model`` 需能被
    ``num_heads`` 整除。``d_sp_enc`` 为空间编码中间层维度。
    """

    def __init__(self, d_model, num_heads, d_sp_enc, sp_enc_activation):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.spatial_encoding = SpatialEncoding(d_sp_enc, sp_enc_activation)

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """将最后一维切分为 (num_heads, depth) 并转置为 (B, H, seq, depth)。"""
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, v, k, q, min_distance_matrix, mask=None):
        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_attention (B, H, seq_q, depth), attention_weights (B, H, seq_q, seq_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, min_distance_matrix, self.spatial_encoding,
            self.num_heads, mask, self.d_sp_enc, self.sp_enc_activation)

        scaled_attention = scaled_attention.transpose(1, 2)  # (B, seq_q, H, depth)
        concat_attention = scaled_attention.reshape(
            batch_size, -1, self.d_model)  # (B, seq_q, d_model)

        output = self.dense(concat_attention)  # (B, seq_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """前馈网络：TF ``Sequential([Dense(dff, relu), Dense(d_model)])``。"""
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model),
    )


class GraphormerBlock(nn.Module):
    """Graphormer 层，对应 Transformer 编码器块。

    与原实现一致采用 Pre-LayerNorm：先归一化再做注意力 / 前馈，最后残差。
    LayerNorm epsilon 保持 ``1e-6``。
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1, d_sp_enc=128,
                 sp_enc_activation='relu'):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, d_sp_enc, sp_enc_activation)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, training, mask, min_distance_matrix):
        residual = x
        x_norm = self.layernorm1(x)  # Pre-LayerNorm
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, min_distance_matrix, mask)
        attn_output = self.dropout1(attn_output) if training else attn_output
        out1 = residual + attn_output

        residual = out1
        out1_norm = self.layernorm2(out1)  # Pre-LayerNorm
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output) if training else ffn_output
        out2 = residual + ffn_output

        return out2


class Graphormer(nn.Module):
    """Graphormer 模型。

    遵循 Ying et al. (2021) 的结构。输入为 ``(node_features, distance_matrix)``。
    与原 TF 实现一致，返回原始序列 ``out``（分类 head 在原实现中已被注释，
    本移植同样省略）。

    已知怪癖：``node_features`` 实际是浮点硬编码特征（``alpha*cost + beta*mem``），
    但经过 ``nn.Embedding`` 需要整数索引。TF Keras Embedding 会把浮点输入
    强制 cast 成 int；PyTorch 用 ``.long().clamp(0, vocab-1)`` 复现相同语义并
    保证可运行（否则真实数据上索引会越界）。
    """

    def __init__(self, num_layers, d_model, num_heads, input_vocab_size,
                 dropout=0.1, dff=512, d_sp_enc=128, sp_enc_activation='relu',
                 max_num_nodes=256, max_degree=256, model_head='VNode',
                 top_dropout=0.1, d_top=64, top_activation='relu',
                 initial_bias=0, concat_n_layers=1):
        super().__init__()

        self.d_model = d_model
        self.max_num_nodes = max_num_nodes
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.centr_encoding = CentralityEncoding(max_degree, d_model)
        self.dropout = nn.Dropout(dropout)

        self.graphormer_layers = nn.ModuleList([
            GraphormerBlock(d_model, num_heads, dff, dropout, d_sp_enc,
                            sp_enc_activation)
            for _ in range(num_layers)
        ])

        # 保留 concat_n_layers 以维持与原类签名一致；分类 head 未移植
        self.concat_n_layers = concat_n_layers
        self.model_head = model_head

    @staticmethod
    def create_padding_mask(nodes):
        return (nodes == 0).float()

    def forward(self, inputs, training):
        # 所有输入以 tuple/list 形式传入第一个参数
        node_features, distance_matrix = inputs

        # 生成 mask
        mask = self.create_padding_mask(node_features)
        attention_mask = mask[:, None, None, :]  # (batch_size, 1, 1, seq_len)

        # Embedding
        # node_features 为浮点硬编码特征，nn.Embedding 需要长整型索引
        embed_indices = node_features.long().clamp(0, self.embedding.num_embeddings - 1)
        embed = self.embedding(embed_indices)  # (batch_size, seq_len, d_model)
        embed *= math.sqrt(float(self.d_model))
        embed += self.centr_encoding(distance_matrix)
        out = embed
        # out = self.dropout(embed)  # 原实现中已注释，保持一致

        hidden_states = []  # 记录每个 block 的隐状态（原实现保留但未用于分类 head）

        # Graphormer layers
        for i in range(self.num_layers):
            out = self.graphormer_layers[i](out, training, attention_mask, distance_matrix)
        if self.concat_n_layers > 1:
            hidden_states.append(out)

        # 分类 head 原实现中已被注释，直接返回序列
        return out
