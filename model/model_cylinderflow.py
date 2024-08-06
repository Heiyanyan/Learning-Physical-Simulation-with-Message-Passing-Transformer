import torch.nn as nn
import torch
from .blocks import EdgeBlock, NodeBlock, MultiHeadAttentionLayer
from utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data
from torch_scatter import scatter_add


def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    module = nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                           nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size))
    if lay_norm: return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Module):
    def __init__(self,
                 edge_input_size=3,
                 node_input_size=11,
                 hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, graph):
        node_attr, _, edge_attr, _ = decompose_graph(graph)

        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)


class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):
        super(GnBlock, self).__init__()

        #MPT
        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        # self.nb_custom_func0 = build_mlp(nb_input_dim, hidden_size, hidden_size)
        self.nb_custom_func = build_mlp(hidden_size, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        # node_update = build_mlp(nb_input_dim, hidden_size)

        #GFL
        # eb_input_dim = 3 * hidden_size
        # nb_input_dim = 2 * hidden_size
        # self.nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        # # self.nb_custom_func = build_mlp(hidden_size, hidden_size, hidden_size)
        # eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        # # node_update = build_mlp(nb_input_dim, hidden_size)

        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=self.nb_custom_func)

    def forward(self, graph, graph_last, attention_layer_node, i):

        graph = self.eb_module(graph)
        # graph = self.nb_module(graph)


        edge_attr = graph.edge_attr
        nodes_to_collect = []
        _, receivers_idx = graph.edge_index
        num_nodes = graph.num_nodes
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)
        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.stack(nodes_to_collect)

        graph_last[2 * i: 2 * i + 2] = collected_nodes
        # graph_last[i: i + 1] = graph.x
        new_graph_last = graph_last.clone()
        # print(new_graph_last[i-1 if i > 1 else 0])
        # print(new_graph_last[i])
        # print(new_graph_last[i+1])
        # print("///////////////////////////////////////////////////////")
        # print(i)
        i = torch.full((graph.num_nodes,), 2 * i + 2, device=torch.device('cuda'), dtype=torch.int64)
        # graph.x = attention_layer_node(graph.x, new_graph_last, new_graph_last, i)
        # graph.x = attention_layer_node(graph.x, new_graph_last, new_graph_last, 2 * i + 2)
        # graph = self.nb_module(graph)
        node_ = attention_layer_node(graph.x, new_graph_last, new_graph_last, 2 * i + 2)
        graph.x = self.nb_custom_func(node_)

        return Data(x=graph.x, edge_attr=graph.edge_attr, edge_index=graph.edge_index)


class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()

        # nb_input_dim = 2 * hidden_size
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)
        # self.nb_module = NodeBlock4(custom_func=self.decode_module)

    def forward(self, graph):

        # graph = self.nb_module(graph)
        return self.decode_module(graph.x)
        # return graph.x


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, output_size, hidden_size=128):

        super(EncoderProcesserDecoder, self).__init__()

        self.message_passing_num = message_passing_num
        self.hidden_size = hidden_size

        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size,
                               hidden_size=hidden_size)

        # self.attention_layer_node = MultiHeadAttentionLayer(hidden_size)
        self.attention_layer_node = MultiHeadAttentionLayer(hidden_size)

        self.processer_list = nn.ModuleList([GnBlock(hidden_size) for _ in range(message_passing_num)])

        # self.decoder_list = nn.ModuleList([Decoder(hidden_size=hidden_size, output_size=1) for _ in range(output_size)])
        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size)

    def forward(self, graph):
        graph_last = torch.zeros(self.message_passing_num * 2, graph.x.size(0), self.hidden_size,
                                 device=torch.device('cuda'))

        graph = self.encoder(graph)

        for i, model in enumerate(self.processer_list):
            graph = model(graph, graph_last, self.attention_layer_node, i)

        # decoded_list = []
        # for i, model in enumerate(self.decoder_list):
        #     decoded_list.append(model(graph))
        #
        # decoded = torch.cat(decoded_list, dim=-1)
        decoded = self.decoder(graph)

        return decoded
