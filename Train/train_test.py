import torch
from tqdm import tqdm

def train_test(model_ft,loss_fn,optimizer,dataloaders,phase,device):
    # 记录损失值
    running_loss = 0.0
    # 记录正确个数
    running_corrects = 0
    for inputs, labels in tqdm(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        labels = labels.squeeze()
        # 只有训练的时候计算和更新梯度
        outputs = model_ft(inputs)
        loss = loss_fn(outputs, labels)

        # torch.max() 返回的是一个元组 第一个参数是返回的最大值的数值 第二个参数是最大值的序号
        _, preds = torch.max(outputs, 1)  # 前向传播 这里可以测试 在valid时梯度是否变化1
        # 训练阶段更新权重
        if phase == 'train':
            loss.backward()  # 反向传播
            optimizer.step()  # 优化权重
            optimizer.zero_grad()  # 梯度清零

        # 计算损失值
        running_loss += loss.item() * inputs.size(0)  # loss计算的是平均值，所以要乘上batch-size，计算损失的总和
        running_corrects += (preds == labels).sum()  # 计算预测正确总个数

    epoch_loss = running_loss / len(dataloaders[phase].sampler)  # 当前轮的总体平均损失值
    epoch_acc = float(running_corrects) / len(dataloaders[phase].sampler)  # 当前轮的总正确率
    return epoch_loss, epoch_acc