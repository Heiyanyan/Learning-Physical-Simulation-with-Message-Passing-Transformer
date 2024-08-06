import matplotlib
import torch
import torch_geometric.transforms as T
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

from dataset.fpc_cylinderflow import FPC, collate, collate2
from model.simulator_cylinderflow import Simulator
from utils.noise import get_velocity_noise
from utils.utils import NodeType

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import time
import numpy as np
import model.GFT as GFT

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
dataset_dir = "/media/heiyanyan/2/DataSet/MGNdataset/cylinderflow"
# dataset_dir = "/media/heiyanyan/2/DataSet/MGNdataset/airfoil"
batch_size = 1
noise_std = 2e-2
lambda_val = 1
seg_rate = 0.5
mode = 'MPT'
# mode = 'cylinderflow_orig'
message_passing_num = 15

print_batch = 100
save_batch = 20000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=message_passing_num, node_input_size=11, edge_input_size=3, output_size=2,
                      device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
# checkpoint = torch.load('/home/heiyanyan/桌面/meshGraphNets_pytorch-master/checkpoint/cylinderflow/simulator_step=1540000_MP=15_cylinderflow_LearnableFourierLoss_seg=0.5+MPT.pth')
# simulator.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# bs = 8: 0.999983
# bs = 1: 0.999999078966387
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999078966387)
print('Optimizer initialized')


def save_to_txt(loss_values, batch_times, total_time, step, filename='training_data.txt', message_passing_num=15,
                mode='ADE'):
    filename = f'{filename.split(".")[0]}_step_{step}_MP={message_passing_num}_{mode}.txt'
    with open(filename, mode='w') as file:
        file.write('Batch Index, Loss, Batch Time\n')
        for i, (loss, batch_time) in enumerate(zip(loss_values, batch_times)):
            file.write(f'{i}, {loss}, {batch_time}\n')
        file.write(f'Total Time, , {total_time}\n')


def compute_graph_loss(predicted_acc, target_acc, edge_index, mask):
    preds_i = predicted_acc[edge_index[0]]
    preds_j = predicted_acc[edge_index[1]]
    targets_i = target_acc[edge_index[0]]
    targets_j = target_acc[edge_index[1]]
    mask_i = mask[edge_index[0]]
    mask_j = mask[edge_index[1]]
    valid_edges_mask = mask_i & mask_j
    diffs = (preds_i - targets_i) - (preds_j - targets_j)
    diffs_masked = diffs[valid_edges_mask]
    # diffs_masked = diffs
    loss = torch.mean(diffs_masked ** 2)

    return loss


def adjust_and_inverse(signal, adjustment_factors, original_indices):
    adjustment_factors_expanded = adjustment_factors.unsqueeze(1).expand(-1, signal.size(1))
    adjusted_signal = signal * adjustment_factors_expanded
    restored_signal = torch.zeros_like(adjusted_signal)
    restored_signal[original_indices] = adjusted_signal

    return restored_signal


def frequency_based_adjustment(transformed_signal, adjustment_factors=None, lambda_val=1, seg_rate=0.5):
    num_nodes = transformed_signal.size(0)
    split_point = int(num_nodes * seg_rate)

    energy = torch.abs(transformed_signal) ** 2
    energy_sum = torch.sum(energy, dim=1)  # 按特征求和以获得每个节点的总能量
    sorted_values, sorted_indices = torch.sort(energy_sum, descending=True)
    sorted_signal = transformed_signal[sorted_indices]

    if adjustment_factors == None:
        high_energy_mean = torch.mean(torch.abs(sorted_signal[:split_point]) ** 2)
        low_energy_mean = torch.mean(torch.abs(sorted_signal[split_point:]) ** 2)
        # print(f"lem: {low_energy_mean}")
        # print(f"hem: {high_energy_mean}")
        adjustment_factors = torch.ones(num_nodes, device='cuda')
        adjustment_factors[split_point:] = torch.sqrt(high_energy_mean / (low_energy_mean + 1e-8)) * lambda_val

        # print(f"low_freq_energy: {low_freq_energy}")
        # print(f"high_freq_energy: {high_freq_energy}")
        # print(f"adjustment_factors: {torch.sqrt(low_freq_energy_mean / (high_freq_energy_mean + 1e-8)) * lambda_val}")

    adjusted_resorted_signal = adjust_and_inverse(sorted_signal, adjustment_factors, sorted_indices)
    return adjusted_resorted_signal, adjustment_factors


def train(model: Simulator, dataloader, optimizer, lambda_val, seg_rate, message_passing_num, mode):
    # save: lambda_val, low_freq_energy, high_freq_energy, adjustment_factors, adjusted_resorted_signal
    collected_data = {
        'orig_signals':[],
        'lambda_val': [],
        'low_freq_energy': [],
        'high_freq_energy': [],
        'adjustment_factors': [],
        'adjusted_resorted_signals': [],
    }

    loss_values = []
    batch_times = []
    start_total_time = time.time()

    for batch_index, graph in enumerate(dataloader):
        # batch_index += 1540000
        start_time = time.time()
        # print(batch_index, graph)
        # t = time.time()
        graph = graph.cuda()
        node_type = graph.x[:, 0]  # "node_type, cur_v, time"

        graph = transformer(graph)
        eigenv = graph.eigenv

        lambda_val = model.get_lambda_val()

        velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph, velocity_sequence_noise)
        # print(f'model:{time.time() - t}')
        mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)

        # GFT
        # error = predicted_acc - target_acc
        # transformed_error = torch.matmul(eigenv.T, error)

        # print(eigenv.T.shape, target_acc.shape)
        target_transformed = torch.matmul(eigenv.T, target_acc)
        predicted_transformed = torch.matmul(eigenv.T, predicted_acc)
        temp = predicted_transformed

        collected_data['orig_signals'].append(predicted_transformed.detach().cpu().numpy())
        # # transformed_error, adjf = frequency_based_adjustment(transformed_error, lambda_val=lambda_val, seg_rate=seg_rate)
        target_transformed, adj_factors = frequency_based_adjustment(target_transformed, lambda_val=lambda_val,
                                                              seg_rate=seg_rate)
        predicted_transformed, _ = frequency_based_adjustment(predicted_transformed, adj_factors, lambda_val=lambda_val,
                                                              seg_rate=seg_rate)
        # print(target_transformed.size())
        collected_data['lambda_val'].append(lambda_val.item())
        high_energy = torch.mean(torch.abs(target_transformed[:int(target_transformed.size(0) * seg_rate)]) ** 2)
        low_energy = torch.mean(torch.abs(target_transformed[int(target_transformed.size(0) * seg_rate):]) ** 2)
        collected_data['high_freq_energy'].append(high_energy.item())
        collected_data['low_freq_energy'].append(low_energy.item())
        collected_data['adjustment_factors'].append(torch.sqrt((low_energy / (high_energy + 1e-8)) * lambda_val).item())
        collected_data['adjusted_resorted_signals'].append(predicted_transformed.detach().cpu().numpy())

        # errors = (error_adjusted_time ** 2)[mask]
        # errors = ((predicted_acc - target_acc) ** 2)[mask]
        errors = ((predicted_transformed - target_transformed) ** 2)[mask]
        mse_loss = torch.mean(errors)

        total_loss = mse_loss
        loss_values.append(total_loss.item())
        # t = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # print(f'opt:{time.time() - t}')

        if optimizer.param_groups[0]['lr'] > 1e-7:
            scheduler.step()

        end_time = time.time()
        batch_time = end_time - start_time
        batch_times.append(batch_time)

        if batch_index % print_batch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Batch {batch_index} [Loss: {total_loss.item():.2e}, '
                f'Time: {batch_time:.2f} seconds, Current LR: {current_lr:.2e}], '
                f'Lambda={lambda_val.item():.4f}, '
                f'High Energy={high_energy.item():.4f}, '
                f'Low Energy={low_energy.item():.4f}'
            )

        if batch_index % save_batch == 0:
            # print(f"lambda_val: {lambda_val}")
            # print(f"adjust factor: {adjf[-1]}")
            def save_data_with_pickle(data, filename):
                with open(filename, 'wb') as f:
                    pickle.dump(data, f)


            # 保存数据
            save_data_with_pickle(collected_data,
                                  f'collected_data_MP={message_passing_num}_{mode}.pkl')

            print("Saved collected data.")
            # Plot each collected data item
            for key, values in collected_data.items():
                if key != 'adjusted_resorted_signals' and key != 'orig_signals':
                    plt.figure(figsize=(10, 6))
                    plt.plot(values, label=f'{key}')
                    plt.xlabel('Batch Number')
                    plt.ylabel(key.replace('_', ' ').title())
                    plt.title(f'{key.replace("_", " ").title()} Over Batches')
                    plt.legend()
                    plt.savefig(f'{key}_batch_{batch_index}_MP={message_passing_num}_{mode}.png', dpi=256)
                    plt.close()

            model.save_checkpoint(optimizer, batch_index, message_passing_num=message_passing_num, mode=mode)
            plt.figure(figsize=(10, 6))
            plt.plot(loss_values, label='Training Loss')
            plt.xlabel('Batch Number')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.title(f'Training Loss - Up to Batch {batch_index}')
            plt.legend()
            plt.savefig(f'training_loss_batch_{batch_index}_MP={message_passing_num}_{mode}.png', dpi=256)  # 保存图表
            plt.close()
        if batch_index % save_batch == 0:
            end_total_time = time.time()
            total_time = end_total_time - start_total_time
            save_to_txt(loss_values, batch_times, total_time, batch_index, message_passing_num=message_passing_num,
                        mode=mode)

    return loss_values, batch_times, total_time


if __name__ == '__main__':
    dataset_fpc = FPC(dataset_dir=dataset_dir, split='train', max_epochs=50)
    train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10, pin_memory=True)
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
    loss_values, batch_times, total_time = train(simulator, train_loader, optimizer, lambda_val, seg_rate,
                                                 message_passing_num, mode)
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()
