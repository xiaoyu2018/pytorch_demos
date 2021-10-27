from torch import nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.liner1=nn.Linear(331,128)
        self.liner2=nn.Linear(128,64)
        self.liner3=nn.Linear(64,1)
        self.ReLu=nn.ReLU()
    def forward(self,x):
        x=self.ReLu(self.liner1(x))
        x=self.ReLu(self.liner2(x))
        return self.liner3(x)
