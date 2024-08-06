# -*- encoding: utf-8 -*-
'''
@File    :   parse_tfrecord.py
@Author  :   jianglx 
@Version :   1.0
@Contact :   jianglx@whu.edu.cn
'''
#解析tfrecord解析数据，存为hdf5文件
import tensorflow as tf
import functools
import json
import os
import numpy as np
import scipy.sparse as sp
import torch
import h5py

def _parse(proto, meta):
  """Parses a trajectory from tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    print(key)
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])
    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')
    out[key] = data
  return out


def load_dataset(path, split):
  """Load dataset."""
  with open(os.path.join(path, 'meta.json'), 'r') as fp:
    meta = json.loads(fp.read())
  ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  ds = ds.prefetch(1)
  return ds


def compute_graph_features(pos, cells):
    # 取第一个时间步的位置和单元格数据
    pos_t0 = pos[0]  # shape: [num_nodes, feature_dim]
    cells_t0 = cells[0]  # shape: [num_triangles, 3]

    # 计算邻接矩阵
    num_nodes = pos_t0.shape[0]
    edges = np.vstack((cells_t0[:, [0, 1]], cells_t0[:, [1, 2]], cells_t0[:, [2, 0]]))  # 从三角形获得边
    edges = np.vstack((edges, edges[:, [1, 0]]))
    adj_matrix = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
    adjacency_matrix = adj_matrix.toarray()  # 转换为密集矩阵形式

    # 计算拉普拉斯矩阵
    deg_matrix = np.diag(adj_matrix.sum(axis=1).A1)  # 度矩阵
    laplacian_matrix = deg_matrix - adjacency_matrix  # 拉普拉斯矩阵

    # 将 NumPy 数组转换为 PyTorch 张量
    laplacian_matrix_torch = torch.from_numpy(laplacian_matrix).to(torch.float32)
    # 使用 PyTorch 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix_torch)
    # 为了后续的存储和使用，你可能需要将这些结果转换回 NumPy 数组
    eigenvalues = eigenvalues.numpy()
    eigenvectors = eigenvectors.numpy()
    return laplacian_matrix, adjacency_matrix, eigenvectors, eigenvalues


def compute_temporal_difference(data):
    # data的格式应该是[time steps, num of nodes, feature dim]
    # 计算t时刻与t-1时刻的差值
    temporal_diff = data[1:, :, :] - data[:-1, :, :]
    # 将首个时间步的差值设为0（因为没有前一个时间步与之比较）
    zero_diff = np.zeros_like(data[:1, :, :])
    # 将计算出的差值和0拼接起来
    temporal_diff = np.concatenate([zero_diff, temporal_diff], axis=0)
    return temporal_diff


if __name__ == '__main__':
    # tf.enable_resource_variables()
    # tf.enable_eager_execution()

    tf_datasetPath='/media/heiyanyan/2/DataSet/MGNdataset/deforming_plate'
    os.makedirs('/media/heiyanyan/2/DataSet/MGNdataset/deforming_plate/', exist_ok=True)

    for split in ['test']:
        ds = load_dataset(tf_datasetPath, split)
        save_path='/media/heiyanyan/2/DataSet/MGNdataset/deforming_plate/' + split  + '.h5'
        f = h5py.File(save_path, "w")
        print(save_path)

        for index, d in enumerate(ds):
            mesh_pos = d['mesh_pos'].numpy()
            world_pos = d['world_pos'].numpy()
            world_pos_diff = compute_temporal_difference(world_pos)
            pos = d['mesh_pos'].numpy()
            node_type = d['node_type'].numpy()
            # velocity = d['velocity'].numpy()
            cells = d['cells'].numpy()
            # pressure = d['pressure'].numpy()
            # density = d['density'].numpy()

            # laplacian_matrix, adjacency_matrix, eigenvectors, eigenvalues = compute_graph_features(mesh_pos, cells)
            data = ("mesh_pos", "world_pos", "world_pos_diff", "node_type", "cells")
            # data = ("pos", "node_type", "velocity", "cells", "pressure", "density")
            # d = f.create_dataset(str(index), (len(data), ), dtype=pos.dtype)
            g = f.create_group(str(index))
            for k in data:
             g[k] = eval(k)

            # 存储计算得到的图特征
            # g['laplacian_matrix'] = laplacian_matrix
            # g['adjacency_matrix'] = adjacency_matrix
            # g['eigenvectors'] = eigenvectors
            # g['eigenvalues'] = eigenvalues
            print(index)
        f.close()