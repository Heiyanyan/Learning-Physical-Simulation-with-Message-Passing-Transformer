import torch
from utils.utils import NodeType, decompose_graph

def get_velocity_noise(graph, noise_std, device):
    velocity_sequence = graph.x[:, 1:3]
    # _, edge_index, edge_attr, _ = decompose_graph(graph)
    # senders_idx, receivers_idx = edge_index
    type = graph.x[:, 0]
    noise = torch.normal(std=noise_std, mean=0.0, size=velocity_sequence.shape).to(device)
    # edge_noise = torch.normal(std=noise_std, mean=0.0, size=edge_attr.shape).to(device)
    mask = type!=NodeType.NORMAL
    noise[mask]=0

    # sender_types = type[senders_idx]
    # receiver_types = type[receivers_idx]
    # edge_mask = (sender_types != NodeType.NORMAL) | (receiver_types != NodeType.NORMAL)
    # edge_noise[edge_mask] = 0
    # return noise.to(device), edge_noise.to(device)
    return noise.to(device)


def get_velocity_noise_3D(graph, noise_std, device):
    velocity_sequence = graph.x[:, 1:4]
    type = graph.x[:, 0]
    noise = torch.normal(std=noise_std, mean=0.0, size=velocity_sequence.shape).to(device)
    mask = type != NodeType.NORMAL
    noise[mask] = 0

    return noise.to(device)


def get_position_noise_3D(graph, noise_std, device):
    velocity_sequence = graph.pos[:, 2:]
    type = graph.x[:, 0]
    noise = torch.normal(std=noise_std, mean=0.0, size=velocity_sequence.shape).to(device)
    mask = type != NodeType.NORMAL
    noise[mask] = 0

    return noise.to(device)


def get_velocitydensity_noise(graph, noise_std, device):
    velocity_sequence = graph.x[:, 1:3]
    density_sequence = graph.x[:, 3].unsqueeze(1)
    type = graph.x[:, 0]
    noise_vel = torch.normal(std=noise_std, mean=0.0, size=velocity_sequence.shape).to(device)
    noise_den = torch.normal(std=noise_std / 500, mean=0.0, size=density_sequence.shape).to(device)
    noise = torch.cat((noise_vel, noise_den), dim=1)
    mask = type!=NodeType.NORMAL
    noise[mask]=0
    return noise.to(device)
