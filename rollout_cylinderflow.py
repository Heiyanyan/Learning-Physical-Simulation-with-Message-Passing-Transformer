from dataset.fpc_cylinderflow import FPC_ROLLOUT
from torch_geometric.loader import DataLoader
import torch
import argparse
import pickle
import torch_geometric.transforms as T
from utils.utils import NodeType
import numpy as np
from model.simulator_cylinderflow import Simulator
from tqdm import tqdm
import os
import time


parser = argparse.ArgumentParser(description='Implementation of MeshGraphNets')
parser.add_argument("--gpu",
                    type=int,
                    default=0,
                    help="gpu number: 0 or 1")

parser.add_argument("--model_dir",
                    type=str,
                    default='checkpoint/simulator.pth')

parser.add_argument("--test_split", type=str, default='test')
parser.add_argument("--rollout_num", type=int, default=100)

args = parser.parse_args()

# gpu devices
torch.cuda.set_device(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rollout_error(predicteds, targets):

    start_time = time.time()
    number_len = targets.shape[0]
    squared_diff = np.square(predicteds - targets).reshape(number_len, -1)
    loss = np.sqrt(np.cumsum(np.mean(squared_diff, axis=1), axis=0)/np.arange(1, number_len+1))

    end_time = time.time()
    infer_time = end_time - start_time

    for show_step in range(0, 1000000, 50):
        if show_step < number_len:
            print('testing rmse  @ step %d loss: %.2e'%(show_step, loss[show_step]))
            print("Rollout inference time for index {}: {:.6f} seconds".format(show_step, infer_time))
        else: break

    return loss


@torch.no_grad()
def rollout(model, dataloader, rollout_index=1):

    dataset.change_file(rollout_index)

    predicted_velocity = None
    mask=None
    predicteds = []
    targets = []

    for graph in tqdm(dataloader, total=600):

        graph = graph.cuda()

        if mask is None:
            node_type = graph.x[:, 0]
            mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.OUTFLOW)
            mask = torch.logical_not(mask)

        if predicted_velocity is not None:
            # print(f'pre_graph.x:{graph.x[555, 1:3]}')
            graph.x[:, 1:3] = predicted_velocity.detach()
            # print(f'pro_graph.x:{graph.x[555, 1:3]}')
            # print(f'pre_graph.pos:{graph.pos[555, 3:5]}')
            # graph.pos[:, 3:5] = predicted_velocity.detach()
            # print(f'pro_graph.pos:{graph.pos[555, 3:5]}')
        graph = transformer(graph)

        next_v = graph.y
        predicted_velocity = model(graph, velocity_sequence_noise=None)

        predicted_velocity[mask] = next_v[mask]

        predicteds.append(predicted_velocity.detach().cpu().numpy())
        targets.append(next_v.detach().cpu().numpy())
        
    crds = graph.pos.cpu().numpy()
    result = [np.stack(predicteds), np.stack(targets)]

    os.makedirs(f'result/cylinderflow/cylinderflow_result_step={step}_MP={message_passing_num}_{mode}', exist_ok=True)
    with open(f'result/cylinderflow/cylinderflow_result_step={step}_MP={message_passing_num}_{mode}/cylinderflow_result_step={step}_MP={message_passing_num}_{mode}_' + str(rollout_index) + '.pkl', 'wb') as f:
        pickle.dump([result, crds], f)
    
    return result


if __name__ == '__main__':

    # simulator.load_checkpoint('checkpoint_TransEPD_v1/simulator.pth_step_280000.pth')
    # simulator.load_checkpoint('checkpoint_TransEPD_origAtt/simulator.pth_step_290000.pth')
    # simulator.load_checkpoint('checkpoint/simulator.pth_step_465000_MP=15_addloss.pth')
    mode = 'MHA'
    message_passing_num = 15
    step = 1770000
    simulator = Simulator(message_passing_num=message_passing_num, node_input_size=11, edge_input_size=3, output_size=2, device=device)
    # simulator.load_checkpoint(f'checkpoint/cylinderflow/simulator_step={step}_MP={message_passing_num}_{mode}.pth')
    # simulator.load_checkpoint(f'checkpoint/cylinderflow/MP=15_cylinderflow_LearnableFourierLoss_seg=0.9/simulator_step={step}_MP={message_passing_num}_{mode}.pth')
    simulator.load_checkpoint(f'checkpoint/cylinderflow/simulator_step={step}_MP={message_passing_num}_{mode}.pth')
    simulator.eval()

    dataset_dir = "/media/heiyanyan/2/DataSet/MGNdataset/cylinderflow"

    # dataset_dir = "/media/heiyanyan/2/DataSet/MGNdataset/airfoil"
    dataset = FPC_ROLLOUT(dataset_dir, split=args.test_split)
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
    test_loader = DataLoader(dataset=dataset, batch_size=1)
    losses_0 = []
    losses_50 = []
    losses_last = []
    losses_info = []
    all_step_losses = []

    total_time = 0

    for i in range(args.rollout_num):
        start_time = time.time()

        result = rollout(simulator, test_loader, rollout_index=i)
        loss = rollout_error(result[0], result[1])
        loss_0 = loss[0]
        loss_50= loss[49]
        loss_last= loss[-1]

        end_time = time.time()
        total_time += end_time - start_time

        losses_0.append(loss_0)
        losses_50.append(loss_50)
        losses_last.append(loss_last)
        losses_info.append((i, loss_0, loss_50, loss_last))
        all_step_losses.append(loss)

    average_time = total_time / args.rollout_num
    print('///////////////////////////////////////////////////////////////')
    print(f"Total time for all rollouts: {total_time:.2f} seconds")
    print(f"Average time per rollout: {average_time:.2f} seconds")

    # 分别对每个时间步的损失进行排序
    sorted_losses_0 = sorted(losses_info, key=lambda x: x[1], reverse=True)[:20]
    sorted_losses_50 = sorted(losses_info, key=lambda x: x[2], reverse=True)[:20]
    sorted_losses_last = sorted(losses_info, key=lambda x: x[3], reverse=True)[:20]

    # 提取rollout_index
    indexes_0 = {info[0] for info in sorted_losses_0}
    indexes_50 = {info[0] for info in sorted_losses_50}
    indexes_last = {info[0] for info in sorted_losses_last}

    # 找出同时出现在三个列表中的rollout_index
    common_indexes = indexes_0 & indexes_50 & indexes_last

    # 打印结果
    print('///////////////////////////////////////////////////////////////')
    print("\nIndexes appearing in top 10 of all three steps:")
    for index in common_indexes:
        print(f"Rollout Index: {index}")

    print("Top 10 rollouts at step 0 with highest loss:")
    for index, loss_0, _, _ in sorted_losses_0:
        print(f"Rollout Index: {index}, Loss: {loss_0:.2e}")

    print("\nTop 10 rollouts at step 50 with highest loss:")
    for index, _, loss_50, _ in sorted_losses_50:
        print(f"Rollout Index: {index}, Loss: {loss_50:.2e}")

    print("\nTop 10 rollouts at last step with highest loss:")
    for index, _, _, loss_last in sorted_losses_last:
        print(f"Rollout Index: {index}, Loss: {loss_last:.2e}")

    # 计算特定时间步的平均损失
    average_loss_0 = np.mean(losses_0)
    average_loss_50 = np.mean(losses_50)
    average_loss_last = np.mean(losses_last)

    print('///////////////////////////////////////////////////////////////')
    print(f"\nAverage RMSE at step 0: {average_loss_0:.2e}")
    print(f"Average RMSE at step 50: {average_loss_50:.2e}")
    print(f"Average RMSE at last step: {average_loss_last:.2e}")

    # 计算并保存每一步的平均误差
    average_step_losses = np.mean(np.array(all_step_losses), axis=0)
    print('///////////////////////////////////////////////////////////////')
    print(average_step_losses)
    np.savetxt(f'average_step_losses_step={step}_MP={message_passing_num}_{mode}.txt', average_step_losses, fmt='%.6e')