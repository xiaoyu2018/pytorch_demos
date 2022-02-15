<<<<<<< HEAD
from settings import *
import models
import data_proc
import torch
import os

device=torch.device(DEVICE)

net=models.MyNet().to(device)
loss_f=nn.CrossEntropyLoss()
opt=optim.Adam(net.parameters(),weight_decay=0.01)



def Train(data_iter,epoch):
    running_loss=0.0
    count=0
    total=0
    for X,y in data_iter:
        X=X.to(device)
        y=y.to(device)

        out=net(X)
        l=loss_f(out,y)

        opt.zero_grad()
        l.backward()
        opt.step()

        running_loss+=l.item()
        with torch.no_grad():
            total+=y.size(0)
            _,index=torch.max(out,dim=1)
            count+=(index==y).sum().item()

    print("epoch %d, loss %f train_accracy %f"%(epoch+1,running_loss,(count/total*1.0)))
def Test(data_iter):
    count=0
    total=0
    with torch.no_grad():
        for X,y in data_iter:
            X=X.to(device)
            y=y.to(device)

            out=net(X)
            total+=y.size(0)

            _,index=torch.max(out,dim=1)
            count+=(index==y).sum().item()
    print("test_accuracy: %f"%(count/total*1.0))


def main():
    data_iter_tr=data_proc.LoadData(train=True)
    data_iter_te=data_proc.LoadData(train=False)

    for epoch in range(EPOCH):
        Train(data_iter_tr,epoch)
        Test(data_iter_te)


if __name__=="__main__":
    net_param=str(net.__class__.__name__)+"_param.pkl"
    # print(net_param)
    if(os.path.exists(os.path.join(SAVE_DIR,net_param))):
        net.load_state_dict(torch.load(os.path.join(SAVE_DIR,net_param)))
        
        # print(net.parameters())
    else:
        main()
        torch.save(net.state_dict(),os.path.join(SAVE_DIR,net_param))
=======
from settings import *
import models
import data_proc
import torch
import os

device=torch.device(DEVICE)

net=models.MyNet().to(device)
loss_f=nn.CrossEntropyLoss()
opt=optim.Adam(net.parameters(),weight_decay=0.01)



def Train(data_iter,epoch):
    running_loss=0.0
    count=0
    total=0
    for X,y in data_iter:
        X=X.to(device)
        y=y.to(device)

        out=net(X)
        l=loss_f(out,y)

        opt.zero_grad()
        l.backward()
        opt.step()

        running_loss+=l.item()
        with torch.no_grad():
            total+=y.size(0)
            _,index=torch.max(out,dim=1)
            count+=(index==y).sum().item()

    print("epoch %d, loss %f train_accracy %f"%(epoch+1,running_loss,(count/total*1.0)))
def Test(data_iter):
    count=0
    total=0
    with torch.no_grad():
        for X,y in data_iter:
            X=X.to(device)
            y=y.to(device)

            out=net(X)
            total+=y.size(0)

            _,index=torch.max(out,dim=1)
            count+=(index==y).sum().item()
    print("test_accuracy: %f"%(count/total*1.0))


def main():
    data_iter_tr=data_proc.LoadData(train=True)
    data_iter_te=data_proc.LoadData(train=False)

    for epoch in range(EPOCH):
        Train(data_iter_tr,epoch)
        Test(data_iter_te)


if __name__=="__main__":
    net_param=str(net.__class__.__name__)+"_param.pkl"
    # print(net_param)
    if(os.path.exists(os.path.join(SAVE_DIR,net_param))):
        net.load_state_dict(torch.load(os.path.join(SAVE_DIR,net_param)))
        
        # print(net.parameters())
    else:
        main()
        torch.save(net.state_dict(),os.path.join(SAVE_DIR,net_param))
>>>>>>> 26700f85104969189a826e0cb75e794a9b87d2d2
