from torch import nn
import torch

class SimpleCnn(nn.Module):
    def __init__(self):
        super(SimpleCnn,self).__init__()

        self.conv1=nn.Conv2d(3,16,kernel_size=5,padding=1)
        self.conv2=nn.Conv2d(16,32,kernel_size=5,padding=1)

        self.pool=nn.AvgPool2d(kernel_size=4)
        self.Relu=nn.ReLU()

        self.fc=nn.Linear(32,2)
    
    def forward(self,x:torch.Tensor):
        bs=x.size(0)

        x=self.Relu(self.pool(self.conv1(x)))
        x=self.Relu(self.pool(self.conv2(x)))

        x=x.view(bs,-1)
        
        x=self.fc(x)
        return x

class Inception(nn.Module):
    def __init__(self,input_num):
        super(Inception,self).__init__()
        # 各层卷积池化操作都要手动指定信息，保证不改变数据W和H
        self.pool_conv1=nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        # 1*1的卷积核是为了快速改变通道数
        self.pool_conv2=nn.Conv2d(input_num,24,kernel_size=1)

        self.conv=nn.Conv2d(input_num,16,kernel_size=1)

        self._2conv1=nn.Conv2d(input_num,16,kernel_size=1)
        self._2conv2=nn.Conv2d(16,24,kernel_size=5,padding=2)

        self._3conv_1=nn.Conv2d(input_num,16,kernel_size=1)
        self._3conv_2=nn.Conv2d(16,24,kernel_size=3,padding=1)
        self._3conv_3=nn.Conv2d(24,24,kernel_size=3,padding=1)

    def forward(self,x):
        res1=self.pool_conv1(x)
        res1=self.pool_conv2(res1)

        res2=self.conv(x)

        res3=self._2conv1(x)
        res3=self._2conv2(res3)

        res4=self._3conv_1(x)
        res4=self._3conv_2(res4)
        res4=self._3conv_3(res4)

        # 将四个输出以维度channel拼接
        # B C W H ，dim=1表明按C拼接
        res=torch.cat([res1,res2,res3,res4],dim=1)
        return res

# 简单地应用Inception
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv1=nn.Conv2d(3,10,kernel_size=5)
        self.conv2=nn.Conv2d(88,20,kernel_size=5)

        self.inception1=Inception(10)
        self.inception2=Inception(20)

        self.pool=nn.MaxPool2d(kernel_size=3)
        # 展平
        self.fc=nn.Linear(88,2,bias=False)

        self.ReLu=nn.ReLU()
    def forward(self,x):
        batch_size=x.size(0)
        
        # 池化层后接激活，对矩阵中每一个元素都激活，不改变B C W H
        x=self.ReLu(self.pool(self.conv1(x)))
        x=self.inception1(x)
        x=self.ReLu(self.pool(self.conv2(x)))
        x=self.inception2(x)
        
        # 展平
        x=x.view(batch_size,-1)
        # print(x.size())
        x=self.fc(x)

        return x
