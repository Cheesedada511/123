import sys
sys.path.append('..')
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.utils import xavier_normal_small_init_, xavier_uniform_small_init_

def make_model(d_atom, d_edge, N=2, d_model=128, h=8, dropout=0.1, attenuation_lambda=0.1, 
               max_length=100, N_dense=2, leaky_relu_slope=0.0, dense_output_nonlinearity='relu', 
               distance_matrix_kernel='softmax', n_output=1, scale_norm=True, 
               init_type='uniform', n_generator_layers=1, aggregation_type='mean'):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, leaky_relu_slope, dropout, attenuation_lambda, distance_matrix_kernel)
    ff = PositionwiseFeedForward(d_model, N_dense, dropout, leaky_relu_slope, dense_output_nonlinearity)
    model = GraphTransformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, scale_norm), N, scale_norm),
        Node_Embeddings(d_atom, d_model, dropout),
        Edge_Embeddings(d_edge, d_model, dropout),
        Position_Encoding(max_length + 1, d_model, dropout),
        Generator(d_model, n_output, n_generator_layers, leaky_relu_slope, dropout, scale_norm, aggregation_type)
    )

    for p in model.parameters():
        if p.dim() > 1:
            if init_type == 'uniform':
                nn.init.xavier_uniform_(p)
            elif init_type == 'normal':
                nn.init.xavier_normal_(p)
            elif init_type == 'small_normal_init':
                xavier_normal_small_init_(p)
            elif init_type == 'small_uniform_init':
                xavier_uniform_small_init_(p)
    return model

class GraphTransformer(nn.Module):
    def __init__(self, encoder, node_embed, edge_embed, pos_embed, generator):
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.pos_embed = pos_embed
        self.generator = generator

    def forward(self, node_features, node_mask, adj_matrix, edge_features):
        return self.predict(self.encode(node_features, edge_features, adj_matrix, node_mask), node_mask)

    def encode(self, node_features, edge_features, adj_matrix, node_mask):
        node_initial = self.node_embed(node_features[:, :, :-1]) + self.pos_embed(node_features[:, :, -1].squeeze(-1).long())
        edge_initial = self.edge_embed(edge_features)
        return self.encoder(node_initial, edge_initial, adj_matrix, node_mask)

    def predict(self, out, out_mask):
        return self.generator(out, out_mask)

def make_pretrain_model(d_atom, d_edge, N=2, d_model=128, h=8, dropout=0.1, attenuation_lambda=0.1, 
               max_length=100, N_dense=2, leaky_relu_slope=0.0, dense_output_nonlinearity='relu', 
               distance_matrix_kernel='softmax', scale_norm=True, 
               init_type='uniform'):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, leaky_relu_slope, dropout, attenuation_lambda, distance_matrix_kernel)
    ff = PositionwiseFeedForward(d_model, N_dense, dropout, leaky_relu_slope, dense_output_nonlinearity)
    model = PretrainGraphTransformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, scale_norm), N, scale_norm),
        Node_Embeddings(d_atom, d_model, dropout),
        Edge_Embeddings(d_edge, d_model, dropout),
        Position_Encoding(max_length + 1, d_model, dropout),
        d_model,dropout,d_atom,num_graphs=3
    )

    for p in model.parameters():
        if p.dim() > 1:
            if init_type == 'uniform':
                nn.init.xavier_uniform_(p)
            elif init_type == 'normal':
                nn.init.xavier_normal_(p)
            elif init_type == 'small_normal_init':
                xavier_normal_small_init_(p)
            elif init_type == 'small_uniform_init':
                xavier_uniform_small_init_(p)
    return model

class PretrainGraphTransformer(nn.Module):
    def __init__(self, encoder, node_embed, edge_embed, pos_embed, d_model,dropout,d_atom,num_graphs=3):
        super(PretrainGraphTransformer, self).__init__()
        self.encoder = encoder
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.pos_embed = pos_embed
        self.num_graphs = num_graphs
        self.fusion_layer = nn.Linear(d_model * num_graphs, d_model)
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dropout=dropout
            ),
            num_layers=6
        )
        self.output_layer = nn.Linear(d_model, d_atom + 1)

    def forward(self, node_features, node_mask, adj_matrix, edge_features):
        graph_embeddings = []
        for idx in range(self.num_graphs):
            x = self.encode(node_features[idx+1], edge_features[idx+1], adj_matrix[idx+1], node_mask[idx+1])
            graph_embeddings.append(x)
        fused_features = torch.cat(graph_embeddings, dim=-1)
        fused_features = self.fusion_layer(fused_features)
        batch_mask = node_mask[0]
        src_key_padding_mask = ~batch_mask
        fused_features = fused_features.permute(1, 0, 2)
        transformer_output = self.transformer(fused_features, src_key_padding_mask=src_key_padding_mask)
        transformer_output = transformer_output.permute(1, 0, 2)

        rec_node_features = self.output_layer(transformer_output)
        return rec_node_features


    def encode(self, node_features, edge_features, adj_matrix, node_mask):
        node_initial = self.node_embed(node_features[:, :, :-1]) + self.pos_embed(node_features[:, :, -1].squeeze(-1).long())
        edge_initial = self.edge_embed(edge_features)
        return self.encoder(node_initial, edge_initial, adj_matrix, node_mask)

class Node_Embeddings(nn.Module):
    def __init__(self, d_atom, d_emb, dropout):
        super(Node_Embeddings, self).__init__()
        self.lut = nn.Linear(d_atom, d_emb) # 
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb

    def forward(self, x):
        return self.dropout(self.lut(x)) * math.sqrt(self.d_emb)

class Edge_Embeddings(nn.Module):
    def __init__(self, d_edge, d_emb, dropout):
        super(Edge_Embeddings, self).__init__()
        self.lut = nn.Linear(d_edge, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb

    def forward(self, x):
        return self.dropout(self.lut(x)) * math.sqrt(self.d_emb)

class Position_Encoding(nn.Module):
    def __init__(self, max_length, d_emb, dropout):
        super(Position_Encoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_length + 1, d_emb, padding_idx=0)

    def forward(self, x):
        return self.dropout(self.pe(x))

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def swish_function(x):
    return x * torch.sigmoid(x)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def mish_function(x):
    return x * torch.tanh(F.softplus(x))


class Generator(nn.Module):
    def __init__(self, d_model, n_output=1, n_layers=1,
                 leaky_relu_slope=0.01, dropout=0.0, scale_norm=False, aggregation_type='mean'):
        super(Generator, self).__init__()
        if n_layers == 1:
            self.proj = nn.Linear(d_model, n_output)
        else:
            self.proj = []
            for i in range(n_layers - 1):
                self.proj.append(nn.Linear(d_model, d_model))
                self.proj.append(Mish())
                self.proj.append(ScaleNorm(d_model) if scale_norm else LayerNorm(d_model))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_model, n_output))
            self.proj = torch.nn.Sequential(*self.proj)

        self.aggregation_type = aggregation_type
        self.leaky_relu_slope = leaky_relu_slope

        if self.aggregation_type == 'gru':
            self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(2 * d_model, d_model)
            self.bias = nn.Parameter(torch.Tensor(d_model))
            self.bias.data.uniform_(-1.0 / math.sqrt(d_model), 1.0 / math.sqrt(d_model))

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask
        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=1)
            out_pooling = out_sum / mask_sum
        elif self.aggregation_type == 'sum':
            out_sum = out_masked.sum(dim=1)
            out_pooling = out_sum
        elif self.aggregation_type == 'summax':
            out_sum = torch.sum(out_masked, dim=1)
            out_max = torch.max(out_masked, dim=1)[0]
            out_pooling = out_sum * out_max
        elif self.aggregation_type == 'gru':
            out_hidden = mish_function(out_masked + self.bias)
            out_hidden = torch.max(out_hidden, dim=1)[0].unsqueeze(0)
            out_hidden = out_hidden.repeat(2, 1, 1)
            cur_message, cur_hidden = self.gru(out_masked, out_hidden)
            cur_message = mish_function(self.linear(cur_message))
            store_message = cur_message * mask
            out_sum = cur_message.sum(dim=1)
            mask_sum = mask.sum(dim=1)
            out_pooling = out_sum / mask_sum
        else:
            out_pooling = out_masked

        projected = self.proj(out_pooling)
        return projected,out_pooling


try:
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v)
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )
        self.dct_norm = nn.LayerNorm(channel, eps=1e-6)

    def forward(self, x):
        b, l, c = x.size()
        x_dct = torch.zeros_like(x)
        for i in range(b):
            x_dct[i] = dct(x[i], norm='ortho')
        lr_weight = self.dct_norm(x_dct)
        lr_weight = self.fc(lr_weight)
        lr_weight = self.dct_norm(lr_weight)
        return x * lr_weight


class Encoder(nn.Module):
    def __init__(self, layer, N, scale_norm):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = ScaleNorm(layer.size) if scale_norm else LayerNorm(layer.size)

    def forward(self, node_hidden, edge_hidden, adj_matrix, mask):
        for layer in self.layers:
            node_hidden, edge_hidden = layer(
                node_hidden, edge_hidden, adj_matrix, mask
            )
        return self.norm(node_hidden)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.norm1 = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.norm2 = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.norm3 = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.block = dct_channel_block(size)

    def forward(self, node_hidden, edge_hidden, adj_matrix, mask):
        node_hidden_norm = self.norm1(node_hidden)
        node_hidden_attn, edge_hidden_temp = self.self_attn(
            node_hidden_norm, node_hidden_norm, edge_hidden, adj_matrix, mask
        )
        node_hidden_attn = self.block(node_hidden_attn)
        node_hidden = node_hidden + self.dropout(node_hidden_attn)
        node_hidden_norm = self.norm2(node_hidden)

        high_freq_features = self.self_attn.edge_distribution_high(edge_hidden, node_hidden_norm, tau=0.1)
        low_freq_features = self.self_attn.edge_distribution_low(edge_hidden)
        node_hidden_norm = node_hidden_norm + high_freq_features + low_freq_features

        node_hidden_ffn = self.feed_forward(node_hidden_norm)
        node_hidden_ffn = self.block(node_hidden_ffn)

        node_hidden = node_hidden + self.dropout(node_hidden_ffn)
        node_hidden = self.norm3(node_hidden)
        return self.norm3(node_hidden), edge_hidden
    


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.1, dense_output_nonlinearity='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'gelu':
            self.dense_output_nonlinearity = lambda x: F.gelu(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
        elif dense_output_nonlinearity == 'swish':
            self.dense_output_nonlinearity = lambda x: x * torch.sigmoid(x)
        elif dense_output_nonlinearity == 'mish':
            self.dense_output_nonlinearity = lambda x: x * torch.tanh(F.softplus(x))

    def forward(self, node_hidden):
        if self.N_dense == 0:
            return node_hidden

        for i in range(self.N_dense - 1):
            node_hidden = self.dropout[i](mish_function(self.linears[i](node_hidden)))

        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](node_hidden)))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


def attention(query, key, value, adj_matrix, mask=None, dropout=None, laplacian_order=2):
    d_k = query.size(-1)

    out_scores = torch.einsum('bhmd,bhmnd->bhmn', query, key) / math.sqrt(d_k)
    in_scores = torch.einsum('bhnd,bhmnd->bhnm', query, key) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1)
        out_scores = out_scores.masked_fill(mask == 0, -np.inf)
        in_scores = in_scores.masked_fill(mask == 0, -np.inf)

    out_attn = F.softmax(out_scores, dim=-1)
    in_attn = F.softmax(in_scores, dim=-1)

    diag_attn = torch.diag_embed(torch.diagonal(out_attn, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

    message = out_attn + in_attn - diag_attn

    laplacian_features = compute_laplacian_high_order(adj_matrix, laplacian_order)
    message = message + laplacian_features.unsqueeze(1)

    message = message * adj_matrix.unsqueeze(1)

    if dropout is not None:
        message = dropout(message)

    node_hidden = torch.einsum('bhmn,bhnd->bhmd', message, value)

    edge_hidden = message.unsqueeze(-1) * key

    return node_hidden, edge_hidden, message


def compute_laplacian_high_order(adj_matrix, order=2):
    degree_matrix = torch.diag_embed(torch.sum(adj_matrix, dim=-1))

    laplacian = degree_matrix - adj_matrix

    laplacian_high_order = laplacian
    for _ in range(order - 1):
        laplacian_high_order = torch.matmul(laplacian_high_order, laplacian)

    laplacian_high_order = F.softmax(-torch.abs(laplacian_high_order), dim=-1)

    return laplacian_high_order

    

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, leaky_relu_slope=0.1, dropout=0.1, attenuation_lambda=0.1, distance_matrix_kernel='softmax'):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.attenuation_lambda = torch.nn.Parameter(torch.tensor(attenuation_lambda, requires_grad=True))
        self.linears = clones(nn.Linear(d_model, d_model), 5)
        self.high_freq_module = nn.Linear(d_model, d_model)
        self.low_freq_module = nn.Linear(d_model, d_model)
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = nn.Dropout(p=dropout)

    def edge_distribution_high(self, edge_hidden, node_hidden, tau):
        src = edge_hidden.mean(dim=2)
        dst = edge_hidden.mean(dim=1)
        feats_abs = torch.abs(node_hidden - src - dst)
        high_freq = F.softmax(feats_abs / tau, dim=-1)
        return self.high_freq_module(high_freq)

    def edge_distribution_low(self, edge_hidden):
        aggregated_edges = edge_hidden.mean(dim=(1, 2))
        low_freq_features = aggregated_edges.unsqueeze(1).expand(-1, edge_hidden.size(1), -1)
        return self.low_freq_module(low_freq_features)

    def forward(self, query_node, value_node, key_edge, adj_matrix, mask=None):
        mask = mask.unsqueeze(1) if mask is not None else mask
        n_batches, max_length, d_model = query_node.shape

        torch.clamp(self.attenuation_lambda, min=0, max=1)
        adj_matrix = self.attenuation_lambda * adj_matrix
        adj_matrix = adj_matrix.masked_fill(mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
        adj_matrix = F.softmax(-adj_matrix, dim=-1)

        query = self.linears[0](query_node).view(n_batches, max_length, self.h, self.d_k).transpose(1, 2)
        key = self.linears[1](key_edge).view(n_batches, max_length, max_length, self.h, self.d_k).permute(0, 3, 1, 2, 4)
        value = self.linears[2](value_node).view(n_batches, max_length, self.h, self.d_k).transpose(1, 2)

        high_freq = self.edge_distribution_high(key_edge, query_node, tau=0.1)
        low_freq = self.edge_distribution_low(key_edge)

        high_freq = high_freq.view(n_batches, max_length, self.h, self.d_k).transpose(1, 2)
        low_freq = low_freq.view(n_batches, max_length, self.h, self.d_k).transpose(1, 2)
        query = query + high_freq
        key = key + low_freq.unsqueeze(3)

        node_hidden, edge_hidden, self.message = attention(
            query, key, value, adj_matrix, mask=mask, dropout=self.dropout
        )

        node_hidden = node_hidden.transpose(1, 2).contiguous().view(n_batches, max_length, self.h * self.d_k)
        edge_hidden = edge_hidden.permute(0, 2, 3, 1, 4).contiguous().view(n_batches, max_length, max_length, self.h * self.d_k)
        return mish_function(self.linears[3](node_hidden)), mish_function(self.linears[4](edge_hidden))