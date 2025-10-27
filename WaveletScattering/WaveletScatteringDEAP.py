import os
import pickle
from  kymatio.torch import Scattering1D
import torch

#DEAP
save_dir = '../AfterWST/DEAP/sub'
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# DEAP
for i in range(1,33):
    with open('../ProcessedData/DEAP/s' + str(i).zfill(2) + '.pkl', 'rb') as f:
        print(str(i).zfill(3))
        sub = pickle.load(f)
        result = []
        for k in range(40):
            #提取所有视频
            fs = 128#采样率
            sub_tensor = torch.from_numpy(sub[k])
            sub_tensor = sub_tensor[:,3*128:27*128]
            sub_tensor = sub_tensor.type(torch.float32).to(device)
            J = 7
            N = 3072
            Q = 8
            S = Scattering1D( J, N, Q).to(device)
            Sx = S.scattering(sub_tensor).to(device)
            #排除第零阶散射系数（无作用）
            Sx = Sx[:,1:,:]
            #输出格式(通道*散射系数*时间)
            result.append(Sx.unsqueeze(0))
            #对最后一个维度进行平均得到时间平移不变性
            #torch.mean(result, dim=2)
        result = torch.cat(result,dim=0)
        save_path = save_dir+str(i).zfill(2)+'WST.pt'
        if not os.path.exists(save_dir[:-4]):
            os.makedirs(save_dir[:-4])
        torch.save(result, save_path)

