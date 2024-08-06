import math
import os
import os.path as osp
import time

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data, Dataset, Batch, DataLoader


# from torch_geometric.loader import DataLoader


# import torch_geometric.transforms as T


# FPC Fluid Process Computation 用于处理特定格式的流体动力学数据，将其转换为图形格式，以便在图神经网络中使用。
class FPCBase():
    def __init__(self, max_epochs=1, files=None):
        # 初始化数据集的基本属性，如处理的轨迹数、文件句柄、数据键名等。
        super().__init__()
        self.eigenvectors_cache = {}
        self.open_tra_num = 10  # 控制同时打开的轨迹数
        self.file_handle = files
        self.shuffle_file()  # 随机化文件顺序

        self.data_keys = ("pos", "node_type", "velocity", "cells", "laplacian_matrix", "eigenvectors")
        self.out_keys = list(self.data_keys) + ['time']

        # 内部状态管理
        self.tra_index = 0  # 轨迹索引，用于追踪当前处理的轨迹
        self.epcho_num = 1  # 记录当前的epoch数
        self.tra_readed_index = -1  # 跟踪当前已读取的轨迹索引
        # dataset attr, 指定轨迹的长度和时间间隔
        self.tra_len = 600
        self.time_iterval = 0.01

        self.opened_tra = []  # 存储当前打开的轨迹列表
        self.opened_tra_readed_index = {}  # 一个字典，用于存储每个打开轨迹的已读取索引
        self.opened_tra_readed_random_index = {}  # 存储每个打开轨迹的随机读取索引
        self.tra_data = {}  # 存储实际的轨迹数据
        self.max_epochs = max_epochs

    def __len__(self):
        ...

    # 打开指定数量的轨迹（数据文件）以供读取。
    def open_tra(self):
        while (len(self.opened_tra) < self.open_tra_num):

            tra_index = self.datasets[self.tra_index]

            # 如果这个轨迹 (tra_index) 还没有被打开（即不在 self.opened_tra 列表中），则将其添加到已打开轨迹的列表中，并初始化相关的读取索引和随机索引。
            if tra_index not in self.opened_tra:
                self.opened_tra.append(tra_index)
                self.opened_tra_readed_index[tra_index] = -1
                # 使用随机排列来选择轨迹帧
                self.opened_tra_readed_random_index[tra_index] = np.random.permutation(self.tra_len - 2)

            self.tra_index += 1

            # 检查是否所有轨迹都已被处理（即一个epoch结束），如果是，则重置轨迹索引并随机化文件顺序。
            if self.check_if_epcho_end():
                self.epcho_end()
                print('Epcho Finished')

    # 检查并关闭已读完的轨迹
    def check_and_close_tra(self):
        to_del = []
        for tra in self.opened_tra:
            if self.opened_tra_readed_index[tra] >= (self.tra_len - 3):
                to_del.append(tra)
        for tra in to_del:
            self.opened_tra.remove(tra)
            try:
                del self.opened_tra_readed_index[tra]
                del self.opened_tra_readed_random_index[tra]
                del self.tra_data[tra]
            except Exception as e:
                print(e)

    # 对数据集中的文件进行随机化
    def shuffle_file(self):
        datasets = list(self.file_handle.keys())
        np.random.shuffle(datasets)
        self.datasets = datasets

    # 检查是否所有轨迹都已读完，如果是，则重新开始一个新的 epoch。
    def epcho_end(self):
        self.tra_index = 0
        self.shuffle_file()
        self.epcho_num = self.epcho_num + 1

    def check_if_epcho_end(self):
        if self.tra_index >= len(self.file_handle):
            return True
        return False

    # def collate(self, graph_list):
    #     return Batch.from_data_list(graph_list)

    # 将读取的数据转换为 PyTorch Geometric 的 Data 对象，以便用于图神经网络。
    @staticmethod
    def datas_to_graph(datas, tra_index=None):

        time_vector = np.ones((datas[0].shape[0], 1)) * datas[4 + 2]
        node_attr = np.hstack((datas[1], datas[2][0], time_vector))
        "node_type, cur_v, time"
        # "pos", "node_type", "velocity", "cells", "pressure"
        crds = torch.as_tensor(datas[0], dtype=torch.float)
        # crds = torch.as_tensor(np.hstack((datas[0], datas[1], datas[2][0])), dtype=torch.float)  # changed
        # senders = edge_index[0].numpy()
        # receivers = edge_index[1].numpy()
        # crds_diff = crds[senders] - crds[receivers]
        # crds_norm = np.linalg.norm(crds_diff, axis=1, keepdims=True)
        # edge_attr = np.concatenate((crds_diff, crds_norm), axis=1)

        target = datas[2][1]
        # node_type, cur_v, pressure, time
        node_attr = torch.as_tensor(node_attr, dtype=torch.float32)
        # edge_attr = torch.from_numpy(edge_attr)
        target = torch.from_numpy(target)
        face = torch.as_tensor(datas[3].T, dtype=torch.long)

        # laplacian_matrix = torch.from_numpy(datas[4][0])
        eigenvectors = torch.from_numpy(datas[5])
        g = Data(x=node_attr, face=face, y=target, pos=crds, eigenv=eigenvectors, tra=tra_index)
        # g = Data(x=node_attr, face=face, y=target, pos=crds, tra=tra_index)
        return g
        # print(node_attr.shape, face.shape, target.shape, crds.shape, eigenvectors.shape, tra_index)

        # for collate2 only
        # return g, eigenvectors

        # for collate only
        # return [node_attr, face, target, crds, eigenvectors, torch.Tensor(tra_index)]
        # here

    @staticmethod
    def rollout_datas_to_graph(datas):

        time_vector = np.ones((datas[0].shape[0], 1)) * datas[4]
        node_attr = np.hstack((datas[1], datas[2][0], time_vector))
        "node_type, cur_v, time"
        # "pos", "node_type", "velocity", "cells", "pressure"
        crds = torch.as_tensor(datas[0], dtype=torch.float)
        # crds = torch.as_tensor(np.hstack((datas[0], datas[1], datas[2][0])), dtype=torch.float)  # changed
        # senders = edge_index[0].numpy()
        # receivers = edge_index[1].numpy()
        # crds_diff = crds[senders] - crds[receivers]
        # crds_norm = np.linalg.norm(crds_diff, axis=1, keepdims=True)
        # edge_attr = np.concatenate((crds_diff, crds_norm), axis=1)

        target = datas[2][1]
        # node_type, cur_v, pressure, time
        node_attr = torch.as_tensor(node_attr, dtype=torch.float32)
        # edge_attr = torch.from_numpy(edge_attr)
        target = torch.from_numpy(target)
        face = torch.as_tensor(datas[3].T, dtype=torch.long)

        # laplacian_matrix = torch.from_numpy(datas[4][0])
        # eigenvectors = torch.from_numpy(datas[5])
        g = Data(x=node_attr, face=face, y=target, pos=crds)

        # g = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=target, pos=crds)

        return g

    # 实现迭代逻辑，选择并读取轨迹数据，转换为图形格式，处理epoch结束
    def __next__(self):

        self.check_and_close_tra()
        self.open_tra()

        if self.epcho_num > self.max_epochs:
            raise StopIteration

        selected_tra = np.random.choice(self.opened_tra)

        data = self.tra_data.get(selected_tra, None)
        if data is None:
            data = self.file_handle[selected_tra]
            self.tra_data[selected_tra] = data

        selected_tra_readed_index = self.opened_tra_readed_index[selected_tra]
        selected_frame = self.opened_tra_readed_random_index[selected_tra][selected_tra_readed_index + 1]
        self.opened_tra_readed_index[selected_tra] += 1

        datas = []
        for k in self.data_keys:
            if k in ["velocity"]:
                r = np.array((data[k][selected_frame], data[k][selected_frame + 1]), dtype=np.float32)
            elif k in ["laplacian_matrix", "eigenvectors"]:
                r = np.array(data[k].astype(np.float32))
            else:
                r = data[k][selected_frame]
                if k in ["node_type", "cells"]:
                    r = r.astype(np.int32)
            datas.append(r)
        datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))
        # ("pos", "node_type", "velocity", "cells", "pressure", "time") 2+3+2+2+1+1
        # formatted_tra = f"{selected_tra:03}"
        # print(selected_tra)
        g = self.datas_to_graph(datas, int(selected_tra))
        # here,
        return g

    def __iter__(self):
        return self


# 专门用于训练数据
class FPC(IterableDataset):
    def __init__(self, max_epochs, dataset_dir, split='test') -> None:

        super().__init__()

        # 处理dateset
        dataset_dir = osp.join(dataset_dir, split + '.h5')
        self.max_epochs = max_epochs
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        print('Dataset ' + self.dataset_dir + ' Initilized')

    # 根据工作进程信息划分数据集，并返回 FPCBase 的迭代器。
    def __iter__(self):
        # 这个函数通常在自定义数据集的 __getitem__ 方法内部使用，以实现每个工作进程特有的数据加载逻辑，比如为每个工作进程设定不同的随机种子，以确保数据加载时的随机性。
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_handle)
        else:
            per_worker = int(math.ceil(len(self.file_handle) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_handle))

        keys = list(self.file_handle.keys())
        keys = keys[iter_start:iter_end]
        files = {k: self.file_handle[k] for k in keys}
        dataset = FPCBase(max_epochs=self.max_epochs, files=files)
        return dataset


# 可能用于测试或评估
class FPC_ROLLOUT(IterableDataset):
    # 初始化数据集，打开指定的 .h5 文件。
    def __init__(self, dataset_dir, split='test', name='flow pass a cylinder'):

        dataset_dir = osp.join(dataset_dir, split + '.h5')
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        self.data_keys = ("pos", "node_type", "velocity", "cells")
        self.time_iterval = 0.01
        self.load_dataset()

    # 加载 .h5 文件中的所有数据集键。
    def load_dataset(self):
        datasets = list(self.file_handle.keys())
        self.datasets = datasets

    # 切换到另一个轨迹文件。
    def change_file(self, file_index):

        file_index = self.datasets[file_index]
        self.cur_tra = self.file_handle[file_index]
        self.cur_targecity_length = self.cur_tra['velocity'].shape[0]
        self.cur_tragecity_index = 0
        self.edge_index = None

    # 逐帧迭代当前轨迹，将数据转换为图形。
    def __next__(self):
        if self.cur_tragecity_index == (self.cur_targecity_length - 1):
            raise StopIteration

        data = self.cur_tra
        selected_frame = self.cur_tragecity_index

        datas = []
        for k in self.data_keys:
            if k in ["velocity"]:
                r = np.array((data[k][selected_frame], data[k][selected_frame + 1]), dtype=np.float32)
            # elif k in ["laplacian_matrix", "eigenvectors"]:
            #     r = np.array(data[k].astype(np.float32))
            else:
                r = data[k][selected_frame]
                if k in ["node_type", "cells"]:
                    r = r.astype(np.int32)
            datas.append(r)
        datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))

        self.cur_tragecity_index += 1
        g = FPCBase.rollout_datas_to_graph(datas)
        # self.edge_index = g.edge_index
        return g

    def __iter__(self):
        return self


# add a function for batch construction: collate()
def collate(graph_list):
    # [node_attr, face, target, crds, eigenvectors, tra_index]
    node_attr, face, target, crds, eigenvectors, tra_index = (graph_list[0][i] for i in range(6))
    num_nodes = node_attr.shape[0]
    t = time.time()
    # 0.15s
    for i in range(1, len(graph_list)):
        node_attr = torch.cat((node_attr, graph_list[i][0]))
        face_to_add = graph_list[i][1] + num_nodes
        face = torch.cat((face, face_to_add), dim=1)
        # print(face.shape)
        # print(face[:, 0])
        # print(face[:, -1])

        target = torch.cat((target, graph_list[i][2]))
        crds = torch.cat((crds, graph_list[i][3]))
        eigenvectors = torch.block_diag(eigenvectors, graph_list[i][4])
        tra_index = torch.cat((tra_index, graph_list[i][5]))

        num_nodes += graph_list[i][0].shape[0]

    batch_g = Data(x=node_attr, face=face, y=target, pos=crds, eigenv=eigenvectors, tra=tra_index)
    return batch_g


def collate2(graph_list):
    batch_graph = Batch.from_data_list([g for g, _ in graph_list])
    batch_egv = torch.block_diag(*(egv for _, egv in graph_list))
    batch_graph.eigenv = batch_egv
    return batch_graph


if __name__ == '__main__':

    dataset_dir = "/media/heiyanyan/2/DataSet/MGNdataset/cylinderflow"
    dataset_fpc = FPC(dataset_dir=dataset_dir, split='train', max_epochs=50)

    train_loader = DataLoader(dataset=dataset_fpc, batch_size=8, pin_memory=True, collate_fn=collate)

    tic = time.time()
    for batch_index, graph in enumerate(train_loader):
        if batch_index == 10:
            toc = time.time() - tic
            print(toc)
            break
