import torch


import torch
from torch_geometric.utils import to_dense_adj

def build_adjacency_matrix(edge_index, num_nodes):
    """使用edge_index构建邻接矩阵"""
    # edge_index: [2, num_edges]
    # num_nodes: 图中节点的数量
    # 返回一个[num_nodes, num_nodes]的邻接矩阵
    adjacency_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    return adjacency_matrix

def compute_laplacian(adjacency_matrix):
    """计算图的拉普拉斯矩阵"""
    degree_matrix = torch.diag(adjacency_matrix.sum(dim=0))
    laplacian_matrix = degree_matrix - adjacency_matrix
    return laplacian_matrix


def graph_fourier_transform(signal, laplacian_matrix):
    """进行图傅里叶变换。"""
    # 计算拉普拉斯矩阵的特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix)
    # 将信号投影到特征向量上
    transformed_signal = torch.matmul(eigenvectors.T, signal)
    return transformed_signal, eigenvalues, eigenvectors

def inverse_graph_fourier_transform(transformed_signal, eigenvectors):
    """进行逆图傅里叶变换。"""
    # 使用特征向量的线性组合恢复原始信号
    original_signal = torch.matmul(eigenvectors, transformed_signal)
    return original_signal