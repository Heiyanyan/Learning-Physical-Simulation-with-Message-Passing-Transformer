import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from utils.utils import decompose_graph
from torch_geometric.data import Data
import numpy as np


# class MultiHeadAttentionLayer(nn.Module):
#     def __init__(self, feature_dim, seq_len, num_heads=4, dropout=0.1):
#         super(MultiHeadAttentionLayer, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = feature_dim // num_heads
#
#         self.norm = nn.LayerNorm(feature_dim)
#         self.query = nn.Linear(feature_dim, feature_dim)
#         self.key = nn.Linear(feature_dim, feature_dim)
#         self.value = nn.Linear(feature_dim, feature_dim)
#
#         self.output_linear = nn.Linear(feature_dim * seq_len, feature_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, query, key, value, i):
#         seq_len, num_nodes, _ = key.shape
#
#         input_query = query
#         query = self.query(query)
#         key = self.key(key)
#         value = self.value(value)
#
#         key = key.transpose(0, 1)  # [num_nodes, seq_len, feature_dim]
#         value = value.transpose(0, 1)  # [num_nodes, seq_len, feature_dim]
#
#         query = query.view(num_nodes, self.num_heads, self.head_dim)
#         key = key.view(num_nodes, seq_len, self.num_heads, self.head_dim)
#         value = value.view(num_nodes, seq_len, self.num_heads, self.head_dim)
#
#         mask = torch.zeros((1, seq_len, 1, 1), device=query.device, dtype=torch.float)
#         mask[:, i + 1:, :, :] = -1e9
#
#         query_expanded = query.unsqueeze(1)
#         scores = query_expanded * key
#         scores = scores / math.sqrt(self.head_dim)
#         # scores = torch.einsum('bhd,bshd->bshd', query, key) / math.sqrt(self.head_dim)
#         # orig MHA: scores = torch.einsum('nhd,nshd->nsh', query, key) / math.sqrt(self.head_dim)
#
#         scores = scores + mask
#         attention = F.softmax(scores, dim=-1)
#         # orig MHA: attention = F.softmax(scores, dim=1)
#         attention = self.dropout(attention)
#
#         weighted_value = attention * value
#         # weighted_value = torch.einsum('bshd,bshd->bshd', attention, value)
#         # orig MHA: weighted_value = torch.einsum('nsh,nshd->nhd', attention, value)
#         weighted_value = weighted_value.view(num_nodes, -1)
#         output = self.output_linear(weighted_value)
#         output = input_query + self.norm(output)
#
#         return output

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.norm = nn.LayerNorm(feature_dim)
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.output_linear = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask_idx=None):
        seq_len, num_nodes, _ = key.shape

        input_query = query
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        key = key.transpose(0, 1)  # [num_nodes, seq_len, feature_dim]
        value = value.transpose(0, 1)  # [num_nodes, seq_len, feature_dim]

        query = query.view(num_nodes, self.num_heads, self.head_dim)
        key = key.view(num_nodes, seq_len, self.num_heads, self.head_dim)
        value = value.view(num_nodes, seq_len, self.num_heads, self.head_dim)

        scores = torch.einsum('nhd,nshd->nsh', query, key) / math.sqrt(self.head_dim)
        # print(scores[0])

        if mask_idx != None:
            index_tensor = torch.arange(seq_len, device=query.device).unsqueeze(0).expand(num_nodes, -1)
            mask = (index_tensor >= mask_idx.unsqueeze(1))
            mask = mask.unsqueeze(-1).to(torch.float32) * -1e9
            scores = scores + mask

        attention = F.softmax(scores, dim=1)
        # print(attention)
        attention = self.dropout(attention)

        weighted_value = torch.einsum('nsh,nshd->nhd', attention, value)
        weighted_value = weighted_value.view(num_nodes, -1)  # 恢复原始查询向量的形状
        output = self.output_linear(weighted_value)
        output = input_query + self.norm(output)

        return output


class EdgeBlock(nn.Module):
    def __init__(self, custom_func=None):
        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        node_attr, edge_index, edge_attr, _ = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)
        collected_edges = torch.cat(edges_to_collect, dim=1)
        # collected_edges = torch.stack(edges_to_collect, dim=0)

        # edge_attr = self.MHA(edge_attr, collected_edges, collected_edges)

        edge_attr_ = edge_attr + self.net(collected_edges)

        return Data(x=node_attr, edge_attr=edge_attr_, edge_index=edge_index)


class NodeBlock(nn.Module):
    def __init__(self, custom_func=None):
        super(NodeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        edge_attr = graph.edge_attr
        nodes_to_collect = []

        _, receivers_idx = graph.edge_index
        num_nodes = graph.num_nodes
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)

        x = graph.x + self.net(collected_nodes)

        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)

#
# class NodeBlock(nn.Module):
#     def __init__(self, custom_func=None):
#         super(NodeBlock, self).__init__()
#         self.net = custom_func
#
#     def forward(self, graph):
#         edge_attr = graph.edge_attr
#         _, receivers_idx = graph.edge_index
#         num_nodes = graph.num_nodes
#         nodes_to_collect = []
#
#         agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)
#
#         nodes_to_collect.append(graph.x)
#         nodes_to_collect.append(agg_received_edges)
#         # collected_nodes = torch.stack(nodes_to_collect, dim=0)
#         #
#         # x = self.MHA(graph.x, collected_nodes, collected_nodes)
#         collected_nodes = torch.cat(nodes_to_collect, dim=-1)
#
#         x = graph.x + self.net(collected_nodes)
#
#         return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)
