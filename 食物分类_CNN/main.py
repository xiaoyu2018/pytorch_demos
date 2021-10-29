import torch
from torch import nn
from torch import optim
from torch.tensor import Tensor
from torch.types import Device
from torch.utils.data import dataloader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset,Subset
import numpy as np
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from PIL import Image
from tqdm.auto import tqdm
import os


BATCH_SIZE=256
EPOCH=20
BASE_DIR="D:/2021UCAS/机器学习/大作业/pytorch_learning/pytorch_demos/食物分类_CNN/dataset"
DEVICE="cuda:0"
DO_SEMI=False

device=torch.device(DEVICE)

train_trans=transforms.Compose([
    transforms.Resize((128,128)),

    transforms.ToTensor()
])
test_trans=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

train_set=DatasetFolder(os.path.join(BASE_DIR,"training","labeled"),extensions="jpg"
                        ,transform=train_trans,loader= lambda x: Image.open(x))
unlabeled_set=DatasetFolder(os.path.join(BASE_DIR,"training","unlabeled"),extensions="jpg"
                        ,transform=train_trans,loader= lambda x: Image.open(x))
val_set=DatasetFolder(os.path.join(BASE_DIR,"validation"),extensions="jpg"
                        ,transform=test_trans,loader= lambda x: Image.open(x))
test_set=DatasetFolder(os.path.join(BASE_DIR,"testing"),extensions="jpg"
                        ,transform=test_trans,loader= lambda x: Image.open(x))

train_iter=DataLoader(dataset=train_set,shuffle=True,batch_size=BATCH_SIZE)
val_iter=DataLoader(dataset=val_set,shuffle=False,batch_size=BATCH_SIZE)
test_iter=DataLoader(dataset=test_set,shuffle=False,batch_size=BATCH_SIZE)

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
        self.fc1=nn.Linear(12672,512,bias=False)
        self.fc2=nn.Linear(512,11,bias=False)

        self.ReLu=nn.ReLU()

        self.norm=nn.BatchNorm2d(88)
        self.drop=nn.Dropout()
        
    def forward(self,x):
        batch_size=x.size(0)
        
        # 池化层后接激活，对矩阵中每一个元素都激活，不改变B C W H
        x=self.ReLu(self.pool(self.conv1(x)))
        x=self.inception1(x)
        x=self.norm(x)
        x=self.ReLu(self.pool(self.conv2(x)))
        x=self.inception2(x)
        x=self.drop(x)
        
        # 展平
        x=x.view(batch_size,-1)
        # print(x.size())
        x=self.ReLu((self.fc1(x)))
        x=self.drop(x)
        x=self.fc2(x)
        return x

net=MyNet().to(device)
creterion=nn.CrossEntropyLoss()
opt=optim.Adam(net.parameters())


# 将超过0.65sofmax概率的的所有data整理成新数据集
def GetSemi(unlabeled_set:DatasetFolder,net:MyNet,threshold=0.65):
    net.load_state_dict(torch.load(os.path.join(os.getcwd(),"param_o.pkl")))
    net.eval()
    
    un_iter=DataLoader(unlabeled_set,batch_size=2048)
    softmax=nn.Softmax(dim=1)
    record=[]
    
    with torch.no_grad():        
        for X,y in un_iter:
            X=X.to(device)
            out=net(X)

            probs=softmax(out)

            for prob in probs:

                index=prob.argmax(dim=0)
                # print(prob[index].item())
                if(prob[index].item()>threshold):
                    record.append(index.item())
                else:
                    record.append(-1)
    
    
    l=len(record)
    for i in range(l):
        if(record[i]==-1):
            continue
        else:
            unlabeled_set.samples[i]=(unlabeled_set.samples[i][0],record[i])

    dels=[i for i in range(l) if record[i]==-1]
    dels.reverse()
    for i in dels:
        del unlabeled_set.samples[i]
                 
    
    # un_iter=DataLoader(unlabeled_set,batch_size=32)
    # for X,y in un_iter:
    #     print(X,y)
    #     break
    
    print("labeled unlabeled data...")
    return unlabeled_set

def Train():
    global train_iter
    global val_iter
    name="param_o.pkl"
    max_acc=0.0

    if(DO_SEMI):
        name="param_s.pkl"
        extra_set=GetSemi(unlabeled_set,net)
        concact_set=ConcatDataset([train_set,extra_set])
        train_iter=DataLoader(concact_set,shuffle=True,batch_size=BATCH_SIZE)
    
    
    for epoch in range(EPOCH):
        print("epoch %d:"%(epoch+1))

        net.train()
        train_loss=[]
        train_acc=[]
        for X,y in tqdm(train_iter):
            X=X.to(device)
            y=y.to(device)

            out=net(X)
            l=creterion(out,y)
            opt.zero_grad()
            l.backward()
            opt.step()

            train_loss.append(l.item())
            train_acc.append((out.argmax(dim=1)==y).float().mean())

        t_loss=sum(train_loss)/len(train_loss)
        t_acc=sum(train_acc)/len(train_acc)
        print("train_loss %f,train_acc %f"%(t_loss,t_acc))

        net.eval()
        val_loss=[]
        val_acc=[]
        for X,y in tqdm(val_iter):
            with torch.no_grad():
                X=X.to(device)
                y=y.to(device)

                out=net(X)
                l=creterion(out,y)

                val_loss.append(l.item())
                val_acc.append((out.argmax(dim=1)==y).float().mean())

        v_loss=sum(val_loss)/len(val_loss)
        v_acc=sum(val_acc)/len(val_acc)
        print("val_loss %f,val_acc %f"%(v_loss,v_acc))
        
        if(max_acc<v_acc):
            max_acc=v_acc
            torch.save(net.state_dict(),os.path.join(os.getcwd(),name))
            print("模型已保存，acc:%f"%max_acc)

def Pred():
    name="param_o.pkl"
    if(DO_SEMI):
        name="param_s.pkl"
    net.load_state_dict(torch.load(os.path.join(os.getcwd(),name)))
    
    net.eval()
    pred=[]

    for X,y in test_iter:
        with torch.no_grad():
            X=X.to(device)
            out=net(X)
        _,res=torch.max(out,dim=1)
        
        for p in res.cpu():
            pred.append(p.item())

    with open("./submission.csv","w") as f:
        f.write("Id,Category\n")

        for i,y in enumerate(pred):
            f.write("%d,%d\n"%(i,y))

    print("submission已生成!")

DO_SEMI=True    
Train()
Pred()