import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def worker_init_fn(worked_id,seed=42):
    worker_seed = seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def loadData(batch_size):
    train_tensor = []
    dev_tensor = []
    test_tensor = []
    #For DEAP
    num_subjects = 32
    train_num = 20  # 0.5
    dev_num = 8  # 0.2
    test_num = 12  # 0.3
    # 读取数据并进行数据集划分
    for i in range(1,num_subjects+1):
        cur_data = torch.load('../AfterWST/DEAP/sub' + str(i).zfill(2) + 'WST.pt')
        indices = torch.randperm(cur_data.size(0))
        shuffled_tensor = cur_data[indices]
        train_tensor.append(shuffled_tensor[:train_num])
        dev_tensor.append(shuffled_tensor[train_num:train_num + dev_num])
        test_tensor.append(shuffled_tensor[train_num + dev_num:])
    train_tensor = torch.cat(train_tensor, dim=0)
    dev_tensor = torch.cat(dev_tensor, dim=0)
    test_tensor = torch.cat(test_tensor, dim=0)

    # 生成标签[28*123 ,1]
    train_labels = torch.arange(0, num_subjects).repeat_interleave(train_num, dim=0).unsqueeze(1)
    dev_labels = torch.arange(0, num_subjects).repeat_interleave(dev_num, dim=0).unsqueeze(1)
    test_labels = torch.arange(0, num_subjects).repeat_interleave(test_num, dim=0).unsqueeze(1)
    # 创建数据集
    train_dataset = TensorDataset(train_tensor, train_labels)
    dev_dataset = TensorDataset(dev_tensor, dev_labels)
    test_dataset = TensorDataset(test_tensor, test_labels)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,worker_init_fn=worker_init_fn,num_workers=0),
        'dev': DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,worker_init_fn=worker_init_fn,num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False,worker_init_fn=worker_init_fn,num_workers=0)
    }
    return dataloaders
