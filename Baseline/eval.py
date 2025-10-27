import logging
import random
import os
import csv
import numpy as np
import torch
from datetime import datetime
from torch import nn
from thop import profile, clever_format

# ==== Import your project modules ====
import Load_Data
import Load_DataFACED
import CNN_GRU
import CADCNN
from train_test import train_test

# ==========================================================
#               1. 固定随机种子
# ==========================================================
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


# ==========================================================
#               2. 计算模型复杂度（参数量、FLOPs）
# ==========================================================
def compute_model_complexity(model, input_shape):
    x = torch.randn(*input_shape)
    flops, params = profile(model, inputs=(x,), verbose=False)
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

    return params, flops



# ==========================================================
#               3. 模型测试函数
# ==========================================================
def evaluate(model_path, model_type, dataset, seed):
    set_random_seed(seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    # 数据加载与模型选择
    if model_type == "CADCNN":
        if dataset == 'DEAP':
            dataloaders = Load_Data.loadData(batch_size)
            model = CADCNN.CADCNN(32)
        elif dataset == 'FACED':
            dataloaders = Load_DataFACED.loadData(batch_size)
            model = CADCNN.CADCNN(30)

    elif model_type == "CNN_GRU":
        if dataset == 'DEAP':
            dataloaders = Load_Data.loadData(batch_size)
            model = CNN_GRU.CNN_GRU(32)
        elif dataset == 'FACED':
            dataloaders = Load_DataFACED.loadData(batch_size)
            model = CNN_GRU.CNN_GRU(30)

    model.to(device)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()

    loss_fn = nn.CrossEntropyLoss().to(device)
    test_loss, test_acc = train_test(model, loss_fn, None, dataloaders, 'test', device)

    print(f"[{model_type}-{dataset}] Test Accuracy: {test_acc:.4f}")
    return test_acc


# ==========================================================
#               4. 保存结果表格（包括复杂度+精度）
# ==========================================================
def save_baseline_results(output_path, results):
    """
    results = [
        {
            "Model": "CADCNN",
            "Params": "1.234M",
            "DEAP": 12.34,
            "FACED": 12.34
        },
        ...
    ]
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Params', 'Performances on Datasets (%)'])
        writer.writerow(['', '', 'DEAP', 'FACED'])

        for r in results:
            writer.writerow([
                r["Model"],
                r["Params"],
                r["DEAP"],
                r["FACED"]
            ])

# ==========================================================
#               5. 主流程：计算复杂度 + 测试 + 保存
# ==========================================================
def baselineStudy():
    results = []

    # ==== 1️⃣ CADCNN ====
    model_CADCNN = CADCNN.CADCNN(32)
    params, flops = compute_model_complexity(model_CADCNN, (1, 32, 175, 24))
    CADCNN_DEAP = evaluate('CADCNNOutput/CADCNNDEAP.pth', 'CADCNN', 'DEAP', 345)
    CADCNN_FACED = evaluate('CADCNNOutput/CADCNNFACED.pth', 'CADCNN', 'FACED', 8)

    results.append({
        "Model": "CADCNN",
        "Params": params,
        "DEAP": round(CADCNN_DEAP * 100, 2),
        "FACED": round(CADCNN_FACED * 100, 2)
    })

    # ==== 2️⃣ CNN_GRU ====
    model_CNNGRU = CNN_GRU.CNN_GRU(32)
    params, flops = compute_model_complexity(model_CNNGRU, (1, 32, 175, 24))
    CNN_GRU_DEAP = evaluate('CNN_GRUOutput/CNN_GRUDEAP.pth', 'CNN_GRU', 'DEAP', 345)
    CNN_GRU_FACED = evaluate('CNN_GRUOutput/CNN_GRUFACED.pth', 'CNN_GRU', 'FACED', 8)

    results.append({
        "Model": "CNN_GRU",
        "Params":params,
        "DEAP": round(CNN_GRU_DEAP * 100, 2),
        "FACED": round(CNN_GRU_FACED * 100, 2)
    })

    # ==== Save Results ====
    save_baseline_results('../Result.csv', results)
    print("\n✅ Baseline study complete. Results saved to '../Result.csv'.")


# ==========================================================
#                       运行入口
# ==========================================================
if __name__ == "__main__":
    baselineStudy()
