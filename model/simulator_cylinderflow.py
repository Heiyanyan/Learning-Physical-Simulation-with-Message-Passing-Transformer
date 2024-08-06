# from .model_cylinderflow_attention import EncoderProcesserDecoder
from .model_cylinderflow import EncoderProcesserDecoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os



class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, output_size, device, model_dir='checkpoint/cylinderflow/simulator') -> None:
        super(Simulator, self).__init__()
        self.lambda_val = nn.Parameter(torch.tensor([0.005], device=device))
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num, node_input_size=node_input_size,
                                             output_size=output_size, edge_input_size=edge_input_size).to(device)
        self._output_normalizer = normalization.Normalizer(size=2, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer', device=device)
        # self._edge_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer', device=device)

        print('Simulator model initialized')

    # 将输入帧（速度）和节点类型（one-hot编码）合并为节点特征
    def update_node_attr(self, frames, types:torch.Tensor):
        node_feature = []

        node_feature.append(frames) #velocity
        node_type = torch.squeeze(types.long())
        one_hot = torch.nn.functional.one_hot(node_type, 9)
        node_feature.append(one_hot)
        node_feats = torch.cat(node_feature, dim=1)
        attr = self._node_normalizer(node_feats, self.training)

        return attr

    # 计算下一个时间步的速度与当前速度的差，得到加速度
    def velocity_to_accelation(self, noised_frames, next_velocity):
        acc_next = next_velocity - noised_frames
        return acc_next

    def get_lambda_val(self):
        return self.lambda_val

    def forward(self, graph:Data, velocity_sequence_noise):

        # 在训练模式下，计算目标加速度并进行标准化。
        if self.training:

            node_type = graph.x[:, 0:1]
            frames = graph.x[:, 1:3]
            target = graph.y

            # 加入噪声提高鲁棒性
            noised_frames = frames + velocity_sequence_noise
            node_attr = self.update_node_attr(noised_frames, node_type)
            graph.x = node_attr
            # k1 = self.model(graph)
            # update_vel_1 = noised_frames + 0.5 * k1
            # node_attr = self.update_node_attr(update_vel_1, node_type)
            # graph.x = node_attr
            # k2 = self.model(graph)
            # update_vel_2 = noised_frames + 0.5 * k2
            # node_attr = self.update_node_attr(update_vel_2, node_type)
            # graph.x = node_attr
            # k3 = self.model(graph)
            # update_vel_3 = noised_frames + k3
            # node_attr = self.update_node_attr(update_vel_3, node_type)
            # graph.x = node_attr
            # k4 = self.model(graph)
            # predicted = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            predicted = self.model(graph)

            target_acceration = self.velocity_to_accelation(noised_frames, target)
            target_acceration_normalized = self._output_normalizer(target_acceration, self.training)

            return predicted, target_acceration_normalized

        # 在评估模式下，使用标准化的逆变换来更新速度，并计算预测速度。不再添加noise
        else:

            node_type = graph.x[:, 0:1]
            frames = graph.x[:, 1:3]
            node_attr = self.update_node_attr(frames, node_type)
            graph.x = node_attr
            # k1 = self.model(graph)
            # update_vel_1 = frames + 0.5 * k1
            # node_attr = self.update_node_attr(update_vel_1, node_type)
            # graph.x = node_attr
            # k2 = self.model(graph)
            # update_vel_2 = frames + 0.5 * k2
            # node_attr = self.update_node_attr(update_vel_2, node_type)
            # graph.x = node_attr
            # k3 = self.model(graph)
            # update_vel_3 = frames + k3
            # node_attr = self.update_node_attr(update_vel_3, node_type)
            # graph.x = node_attr
            # k4 = self.model(graph)
            # predicted = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            predicted = self.model(graph)

            # 用标准化的逆变换，加上加速度以返回速度
            velocity_update = self._output_normalizer.inverse(predicted)
            predicted_velocity = frames + velocity_update

            return predicted_velocity

    def load_checkpoint(self, ckpdir=None):

        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir)
        self.load_state_dict(dicts['model'])

        keys = list(dicts.keys())
        keys.remove('model')
        keys.remove('optimizer')

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.' + k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s" % ckpdir)

    def save_checkpoint(self, optimizer, batch_index, savedir=None, message_passing_num=15, mode='ADE'):
        if savedir is None:
            savedir = self.model_dir
        savedir = f'{savedir}_step={batch_index}_MP={message_passing_num}_{mode}.pth'

        os.makedirs(os.path.dirname(savedir), exist_ok=True)

        model_state = self.state_dict()
        optimizer_state = optimizer.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer = self._node_normalizer.get_variable()
        # _edge_normalizer = self._edge_normalizer.get_variable() 如果有边缘归一化器，也可以保存

        to_save = {
            'model': model_state,
            '_output_normalizer': _output_normalizer,
            '_node_normalizer': _node_normalizer,
            'optimizer': optimizer_state
        }

        torch.save(to_save, savedir)
        print(f'Simulator model saved at {savedir}')

