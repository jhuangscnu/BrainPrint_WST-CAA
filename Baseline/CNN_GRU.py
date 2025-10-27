import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_GRU(nn.Module):
    def __init__(self,in_channels):
        super(CNN_GRU, self).__init__()
        num_class = 0
        if in_channels==30:
            num_class = 123
        elif in_channels==32:
            num_class = 32
        #CNN Layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.ln1 = nn.LayerNorm([88,12])
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.ln2 = nn.LayerNorm([44,6])
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.ln3 = nn.LayerNorm([22,3])
        self.dropout3 = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        #FC1

        self.linear1 = nn.Linear(in_features=32*22, out_features=256)
        #GRU Layer
        self.gru1 = nn.GRU(input_size=256, hidden_size=128, num_layers=1,batch_first=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, num_layers=1,batch_first=True)
        self.fc = nn.Sequential(

            nn.Dropout(p=0.5),
            nn.Linear(64*3,128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.ln3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = x.permute(0,3,1,2)
        x = x.reshape(x.size(0),x.size(1),-1)
        x = self.linear1(x)
        x,_ = self.gru1(x)
        x,_ = self.gru2(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        return x



if __name__ == '__main__':
    test = torch.randn(64,30,175,24)
    model = CNN_GRU(in_channels=30)
    out = model(test)
    print(out)