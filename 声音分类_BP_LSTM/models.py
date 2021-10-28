from torch import nn
import torch

class Bp(nn.Module):
    
    def __init__(self,in_dim,out_dim):
        super(Bp,self).__init__()
        self.fc1=nn.Linear(in_dim,1024)
        self.fc2=nn.Linear(1024,2048)
        self.fc3=nn.Linear(2048,out_dim)
        self.relu=nn.ReLU()
        self.drop=nn.Dropout()

    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.drop(x)
        x=self.relu(self.fc2(x))
        x=self.drop(x)
        x=self.relu(self.fc3(x))

        return x

# 不太行
class BiLstm(nn.Module):
    def __init__(self,in_dim,hidden_dim,num_layers,out_dim):
        super(BiLstm,self).__init__()
        self.lstm=nn.LSTM(batch_first=True,input_size=in_dim,hidden_size=hidden_dim,num_layers=num_layers,bidirectional=True)
        self.fc=nn.Linear(hidden_dim,out_dim)
        self.ReLU=nn.ReLU()
        
    def forward(self,x):
        _,(h,_)=self.lstm(x)
        
        h=self.ReLU(h)
        h=self.ReLU(self.fc(h[3]))

        return h


