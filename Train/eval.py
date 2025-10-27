import logging
import random
import os
import numpy as np
import torch
from datetime import datetime
import Load_Data
import Densenet
import Load_DataFACED
import DensenetFACED
from thop import profile
from thop import clever_format
from train_test  import train_test
from torch import nn
import argparse
from decimal import Decimal, ROUND_HALF_UP

import csv


seedDEAP = 345
seedFACED = 8

def set_random_seed(seed):
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def save_ablation_results(output_path, deap_vals, faced_vals):
    """
    将 DEAP 和 FACED 的消融实验准确率结果保存到 CSV 文件中。
    参数：
    - output_path: 保存路径
    - deap_vals: list[float]，DEAP 的 6 个准确率
    - faced_vals: list[float]，FACED 的 6 个准确率
    """
    headers = [['Datasets', 'Performances with Ablation(%)'],['','Noabl', 'AA', 'CA', 'CAA']]
    deap_row = ['DEAP'] + deap_vals
    faced_row = ['FACED'] + faced_vals

    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])                   # 空行
        writer.writerow(['Ablation Result'])  # 表头
        writer.writerow(headers[0])
        writer.writerow(headers[1])
        writer.writerow(faced_row)
        writer.writerow(deap_row)

def save_full_model_results(output_path, model_name, results):
    """
    追加保存完整模型结果（无标题）
    参数：
    - output_path: CSV 文件路径
    - model_name: 模型名称（如 CAADN）
    - results: list[tuple]，如 [('FACED', 0.9685), ('DEAP', 0.9922)]
    """
    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        row = results
        writer.writerow(row)





def evaluate(model_path,model_type, isAttention, isCA, isEA, isTA, seed):
    set_random_seed(seed)

    # GPU device
    device = torch.device("cuda:1")

    # 加载数据
    batch_size = 64


    # 构建模型结构并加载权重
    if model_type == "Densenet":
        dataloaders = Load_Data.loadData(batch_size)
        model = Densenet.densenet(isAttention, isCA, isEA, isTA)
    elif model_type == "DensenetFACED":
        dataloaders = Load_DataFACED.loadData(batch_size)
        model = DensenetFACED.densenet(isAttention, isCA, isEA, isTA)
    model.to(device)
    model.load_state_dict(torch.load(model_path)['state_dict'])

    model.eval()

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss().to(device)

    # 测试模型
    test_loss, test_acc = train_test(model, loss_fn, None, dataloaders, 'test', device)

    print("Model:", model_path)
    print("Test Loss: {:.4f}".format(test_loss))
    print("Test Accuracy: {:.4f}".format(test_acc))

    return test_acc

def ablationStudy():
    # 示例：请替换为你保存的模型路径
    #DEAP

    DEAP_Noabl = evaluate('DEAPoutput/Noabl(CAADN).pth','Densenet', True, True, True, True, seedDEAP)
    DEAP_AA = evaluate('DEAPoutput/AA(CADN).pth','Densenet',True,  True, False, False,seedDEAP)
    DEAP_CA =evaluate('DEAPoutput/CA(AADN).pth','Densenet',True, False, True, True, seedDEAP)
    DEAP_CAA = evaluate('DEAPoutput/CAA(DN).pth','Densenet',False,  False, False, False,seedDEAP)


    #FACED
    FACED_Noabl = evaluate('FACEDOutput/Noabl(CAADN).pth', 'DensenetFACED',True, True, True, True, seedFACED)
    FACED_AA = evaluate('FACEDOutput/AA(CADN).pth','DensenetFACED',True,  True, False, False,seedFACED)
    FACED_CA =  evaluate('FACEDOutput/CA(AADN).pth','DensenetFACED',True, False, True, True, seedFACED)
    FACED_CAA = evaluate('FACEDOutput/CAA(DN).pth','DensenetFACED',False,  False, False, False,seedFACED)




    deap_vals = [f"{v * 100:.2f}" for v in [
        DEAP_Noabl,
        DEAP_AA,
        DEAP_CA,
        DEAP_CAA
    ]]

    faced_vals = [f"{v * 100:.2f}" for v in [
        FACED_Noabl,
        FACED_AA,
        FACED_CA,
        FACED_CAA
    ]]
    save_ablation_results('../Result.csv', deap_vals, faced_vals )

def fullModel():
    DEAP_acc = evaluate('DEAPoutput/Noabl(CAADN).pth', 'Densenet', True, True, True, True, 345)
    FACED_acc = evaluate('FACEDOutput/Noabl(CAADN).pth', 'DensenetFACED', True, True, True, True, 8)
    model_ft = Densenet.densenet(True,True,True,True)
    x = torch.randn(1,32, 175, 24)
    flops, params = profile(model_ft, inputs=(x, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
        # 将 params 统一转为 M 单位
    if params.endswith("K"):
        val = float(params[:-1]) / 1e3  # K → M
    elif params.endswith("G"):
        val = float(params[:-1]) * 1e3  # G → M
    elif params.endswith("M"):
        val = float(params[:-1])         # 已经是 M
    else:
        val = float(params) / 1e6        # 纯数字情况
    
    params = f"{val:.2f}M"

    results = ([
        "CAADN",
        params,
        round(DEAP_acc * 100, 2),
        round(FACED_acc * 100, 2)
    ])

    save_full_model_results('../Result.csv', 'CAADN', results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ablation study or full model evaluation")
    group = parser.add_mutually_exclusive_group(required=True)  # 互斥参数，必须传一个
    group.add_argument('--ablation', action='store_true', help='Run ablation study and export CSV.')
    group.add_argument('--full', action='store_true', help='Run full model evaluation and export CSV.')

    args = parser.parse_args()

    if args.ablation:
        ablationStudy()
    elif args.full:
        fullModel()