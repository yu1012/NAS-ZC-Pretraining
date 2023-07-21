import math
import numpy as np
import torch
from torch_geometric.data import Data
from nats_bench import create

#following BPR_NAS


op2onehot = {"skip_connect": [1, 0, 0, 0],
             "nor_conv_1x1": [0, 1, 0, 0],
             "nor_conv_3x3": [0, 0, 1, 0],
             "avg_pool_3x3": [0, 0, 0, 1]}
OP_NAMES = ["skip_connect", "none", "nor_conv_3x3", "nor_conv_1x1", "avg_pool_3x3"]
data_path = "../../data/dongyeong/NAS/"
matrix = np.array(
    [
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)
def parse_string(string):
    splited = string.split("+")
    edge_list = [[], []]
    edge_feat = []
    for i in range(3):
        node = splited[i].strip('|')
        ops = node.split('|')
        for opn in ops:
            op, node = opn.split('~')
            if op != 'none':
                edge_list[0].append(int(node))
                edge_list[1].append(i + 1)
                edge_feat.append(op2onehot[op])
    return torch.tensor(edge_list), torch.tensor(edge_feat)
def get_train_time_minmax(api):
    min_train_time = math.inf
    max_train_time = -math.inf
    min_train_time_h = math.inf
    max_train_time_h = -math.inf
    for i in range(len(api)):
        info = api.get_more_info(i, "cifar10-valid", hp="200")
        half_info = api.get_more_info(i, "cifar10-valid", iepoch=100, hp="200")
        train_time = info["train-all-time"]
        train_half_time = half_info["train-all-time"]
        min_train_time = min(min_train_time, train_time)
        max_train_time = max(max_train_time, train_time)
        min_train_time_h = min(min_train_time_h, train_half_time)
        max_train_time_h = max(max_train_time_h, train_half_time)
    return min_train_time_h, max_train_time_h, min_train_time, max_train_time
def get_dataset(api, min_train_time_h, max_train_time_h, min_train_time, max_train_time):
    dataset = []
    unique_strs = set()
    for i, arch_str in enumerate(api):
        unique_str = api.get_unique_str(i)
        if unique_str in unique_strs:
            continue
        else:
            unique_strs.add(unique_str)
        edge_index, edge_attr = parse_string(arch_str)
        info = api.get_more_info(i, "cifar10-valid", hp="200")
        half_info = api.get_more_info(i, "cifar10-valid", iepoch=100, hp="200")
        full_train_time = info["train-all-time"]
        half_train_time = half_info["train-all-time"]
        full_train_acc = info["train-accuracy"] / 100
        half_train_acc = half_info["train-accuracy"] / 100
        full_val_acc = info["valid-accuracy"] / 100
        half_val_acc = half_info["valid-accuracy"] / 100
        full_test_acc = info["test-accuracy"] / 100
        y = torch.tensor([half_train_time, half_train_acc, half_val_acc,
                          full_train_time, full_train_acc, full_val_acc, full_test_acc])
        dataset.append(Data(edge_index=edge_index, edge_attr=edge_attr, y=y))
    return dataset
def str_to_op(arch_str):
    splited = arch_str.split("+")
    ops_list = []
    for i in range(3):
        node = splited[i].strip('|')
        ops = node.split('|')
        for opn in ops:
            op = opn.split('~')[0]
            ops_list.append(op)
    ops = [OP_NAMES.index(name) for name in ops_list]
    ops = [op + 1 for op in ops]
    ops = [0, *ops, 6]
    return ops
def convert_arch_to_seq(matrix, ops, max_n=8):
    seq = []
    n = len(matrix)
    max_n = max_n
    assert n == len(ops)
    for col in range(1, max_n):
        if col >= n:
            seq += [0 for _ in range(col)]
            seq.append(0)
        else:
            for row in range(col):
                seq.append(matrix[row][col] + 1)
            seq.append(ops[col] + 2)
    assert len(seq) == (max_n + 2) * (max_n - 1) / 2
    return seq
def get_dataset_seminas(api):
    dataset = []
    unique_strs = set()
    labels = []
    for i, arch_str in enumerate(api):
        unique_str = api.get_unique_str(i)
        if unique_str in unique_strs:
            continue
        else:
            unique_strs.add(unique_str)
        seq = convert_arch_to_seq(matrix, str_to_op(arch_str))
        full_test_metric = api.get_more_info(i, "cifar10-valid", hp="200")["test-accuracy"]
        full_test_acc = full_test_metric / 100
        y = full_test_acc
        dataset.append((seq, y))
        labels.append(y)
    labels = torch.tensor(labels).float()
    return dataset, labels
if __name__ == "__main__":
    api = create(data_path + 'nats_raw/', 'tss', fast_mode=True, verbose=False)
    # graph representation
    min_time_h, max_time_h, min_time, max_time = get_train_time_minmax(api)
    dataset = get_dataset(api, min_time_h, max_time_h, min_time, max_time)
    torch.save(dataset, data_path + "processed/nasbench201_dataset.pt")
    # sequence for semiNAS
    dataset, labels = get_dataset_seminas(api)
    torch.save(dataset, data_path + "processed/nasbench201_seminas.pt")
    torch.save(labels, data_path + "processed/labels.pt")