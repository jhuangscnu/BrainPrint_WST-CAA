import torch
from thop import profile
from thop import clever_format

# 假设你的模型类
import CADCNN # 修改为你实际的模型路径
import  CNN_GRU

# 创建模型
model_CADCNN = CADCNN.CADCNN(32)
model_CNNGRU = CNN_GRU.CNN_GRU(32)
# 创建一个假输入（根据你的数据形状）
# 例如输入是 32 通道、3840 采样点
x = torch.randn(1,32, 175, 24)

# 计算 FLOPs 和 参数量
flops, params = profile(model_CADCNN, inputs=(x, ), verbose=False)
flops, params = clever_format([flops, params], "%.3f")

print(f"CADCNN Model Parameters: {params}")
print(f"CADCNN Model FLOPs: {flops}")


flops, params = profile(model_CNNGRU, inputs=(x, ), verbose=False)
flops, params = clever_format([flops, params], "%.3f")

print(f"CNNGRU Model Parameters: {params}")
print(f"CNNGRU Model FLOPs: {flops}")

