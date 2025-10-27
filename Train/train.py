
import logging
import random
from  datetime import datetime
import time
import copy
import argparse  # 新增参数解析模块


import numpy as np
import torch
from torch.optim import lr_scheduler

import Load_Data
from torch import nn, optim
#from torch.utils.tensorboard import SummaryWriter
import os

import Densenet
from train_test import train_test


seed = 345

dir_path = str('DEAPoutput')
os.mkdir(dir_path)  

os.environ['TORCH_USE_CUDA_DSA'] = '1'
# 创建main专用logger
main_logger = logging.getLogger("main")
main_logger.setLevel(logging.INFO)  # 设置日志级别

# 创建train专用logger
train_logger = logging.getLogger("train")
train_logger.setLevel(logging.INFO)


# 定义日志格式
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# 为main logger添加文件和控制台处理器
main_file_handler = logging.FileHandler(dir_path+'/main.log')
main_file_handler.setFormatter(formatter)
main_logger.addHandler(main_file_handler)
main_logger.addHandler(logging.StreamHandler())  # 可选：保持控制台输出

train_file_handler = logging.FileHandler(dir_path+'/train.log')
train_file_handler.setFormatter(formatter)
train_logger.addHandler(train_file_handler)
train_logger.addHandler(logging.StreamHandler())


def set_random_seed(seed):

    os.environ["CUDA_LAUNCH_BLOCKING"] = str(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def train(filename,isAttention,isCA,isEA,isTA,seed):
    set_random_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # 这里面的变量都相当于全局变量 ！！


    # GPU计算
    device = torch.device("cuda:3")

    #  训练总轮数
    total_epochs = 1000
    # 每次取出样本数
    batch_size = 64
    # 初始学习率
    Lr = 0.1
    #now = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    #filename = filename+now # 文件扩展名在保存时添加


    dataloaders = Load_Data.loadData(batch_size)



    # 模型

    model_ft = Densenet.densenet(isAttention,isCA,isEA,isTA)

    model_ft.to(device)

    # 创建损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # 训练模型
    # 显示要训练的模型
    train_logger.info("==============当前模型要训练的层==============")
    for name, params in model_ft.named_parameters():
        if params.requires_grad:
            train_logger.info(name)

    # 训练模型所需参数
    # 用于记录损失值未发生变化batch数
    counter = 0
    counter_stop = 0
    # 记录训练次数
    total_step = {
        'train': 0, 'dev': 0
    }
    # 记录开始时间
    since = time.time()
    # 记录当前最小损失值
    valid_loss_min = np.Inf
    # 保存最优正确率
    best_acc = 0
    # 保存的文件名
    save_name_t = ''
    optimizer = optim.SGD(model_ft.parameters(), lr=Lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=25)
    for epoch in range(total_epochs):
        train_logger.info('Epoch {}/{}'.format(epoch + 1, total_epochs))
        train_logger.info('-' * 10)
        train_logger.info('')
        if counter_stop == 100:
            break
        # 训练和验证 每一轮都是先训练train 再验证valid
        for phase in ['train', 'dev']:
            # 调整模型状态
            if phase == 'train':
                model_ft.train()  # 训练
            else:
                model_ft.eval()  # 验证


            epoch_loss,epoch_acc =train_test(model_ft,loss_fn,optimizer,dataloaders,phase,device)
            total_step[phase] += 1

            time_elapsed = time.time() - since
            train_logger.info('')
            train_logger.info('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            train_logger.info('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format(phase, epoch_loss, counter, epoch_acc))

            if phase == 'dev':
                # 得到最好那次的模型
                scheduler.step(epoch_loss)
                if epoch_loss < valid_loss_min:  # epoch_acc > best_acc:

                    best_acc = epoch_acc

                    # 保存当前模型
                    best_model_wts = copy.deepcopy(model_ft.state_dict())
                    state = {
                        'state_dict': model_ft.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                    }
                    # 只保存最近1次的训练结果

                    save_name_t = '{}/{}.pth'.format(dir_path,filename)
                    torch.save(state, save_name_t)  # \033[1;31m 字体颜色：红色\033[0m
                    train_logger.info("已保存最优模型，准确率:\033[1;31m {:.2f}%\033[0m，文件名：{}".format(best_acc * 100, save_name_t))

                    valid_loss_min = epoch_loss
                    counter = 0
                    counter_stop =0
                else:
                    counter += 1
                    counter_stop += 1

        train_logger.info('')
        train_logger.info('当前学习率 : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        train_logger.info('')

    model_ft.load_state_dict(torch.load('{}/{}.pth'.format(dir_path,filename))['state_dict'])
    epoch_loss, epoch_acc = train_test(model_ft, loss_fn, optimizer, dataloaders, 'test', device)

    # 训练结束
    time_elapsed = time.time() - since
    train_logger.info('')
    train_logger.info('任务完成！')
    train_logger.info('任务完成总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    train_logger.info('最高验证集准确率: {:4f}'.format(best_acc))
    train_logger.info('测试集loss:{:4f}'.format(epoch_loss))
    train_logger.info('测试集准确率: {:4f}'.format(epoch_acc))

    save_name_percentage = save_name_t
    os.rename(save_name_t, save_name_percentage)
    logging.info('最优模型保存在：{}'.format(save_name_t))

    return epoch_loss, epoch_acc

if __name__ == "__main__":


    main_logger.info("ablation result")


    CAADN_loss, CAADN_ACC = train('Noabl(CAADN)', True, True, True, True, seed)
    main_logger.info('Noabl(CAADN):' + str(CAADN_loss) + 'ACC:' + str(CAADN_ACC))
    

    CADN_loss, CADN_ACC = train('AA(CADN)', True,  True, False, False,seed)
    main_logger.info('AA(CADN):loss:'+str(CADN_loss)+'ACC:'+str(CADN_ACC))

    AADN_loss, AADN_ACC = train('CA(AADN)', True, False, True, True, seed)
    main_logger.info('CA(AADN):loss:' + str(AADN_loss) + 'ACC:' + str(AADN_ACC))

    DN_loss, DN_ACC = train('CAA(DN)', False,  False, False, False,seed)
    main_logger.info('CAA(DN):loss:'+str(DN_loss)+'ACC:'+str(DN_ACC))
