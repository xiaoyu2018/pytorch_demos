from pickle import TRUE
import torch
from torch import nn
from torch import optim
import settings,dataproc,models



device=torch.device(settings.DEVICE)

net1=models.Bp(settings.FEAUTRES,settings.NUM_CLASS).to(device)
# 重新分割了特征，将429个特征分为13个序列每个序列33个特征
net2=models.BiLstm(33,64,2,settings.NUM_CLASS).to(device)

criterion=nn.CrossEntropyLoss()
opt1=optim.Adam(net1.parameters(),lr=settings.LEARNING_RATE)
opt2=optim.Adam(net2.parameters(),lr=settings.LEARNING_RATE)

def Train(lstm=False):
    train,val=dataproc.GetTimit11()
    
    if(lstm):
        net=net2;opt=opt2

    else:
        net=net1;opt=opt1

    for epoch in range(settings.EPOCH):
        train_loss=0.0
        val_loss=0.0
        train_acc=0.0
        val_acc=0.0
        max_acc=0.0
        train_t=0.0
        val_t=0.0

        # 设置模型为训练模式
        net.train()
        for X,y in train:
            if(lstm):
                # 将原本429维特征分为13*33
                # (size,13,33)
                X=X.view(-1,13,33)

            X=X.to(device)
            y=y.to(device)
            
            out=net(X)
            l=criterion(out,y)
            opt.zero_grad()
            l.backward()
            opt.step()

            train_t+=y.size(0)
            _, train_pred = torch.max(out, 1)
            train_loss+=l.item()
            train_acc+=(train_pred.cpu() == y.cpu()).sum().item()
        # 设置模型为验证模式，一些层如DROPOUT不会执行
        net.eval()
        with torch.no_grad():
            for X,y in val:
                if(lstm):
                    X=X.view(-1,13,33)
                    
                X=X.to(device)
                y=y.to(device)
                
                out=net(X)
                l=criterion(out,y)
                
                val_loss+=l.item()

                val_t+=y.size(0)
                _, val_pred = torch.max(out, 1) 
                val_acc += (val_pred.cpu() == y.cpu()).sum().item()

                if(val_acc>max_acc):
                    max_acc=val_acc
                    torch.save(net.state_dict(),"./param.pkl")
                    
        print("epoch %d\ntrain_loss %f,train_acc %f\nval_loss %f,val_acc %f\n"%(epoch+1,train_loss/train_t,train_acc/train_t,val_loss/val_t,val_acc/val_t))

def Pred(lstm=False):
    test_iter=dataproc.GetTimit11(False)
    predict=[]
    if(lstm):
        net=net2
    else:
        net=net1
    net.load_state_dict(torch.load('./param.pkl'))
    net.eval()
    with torch.no_grad():
        for X in test_iter:
            if(lstm):
                X=X.view(-1,13,33)
            X=X.to(device)
            out=net(X)
            _,pred=torch.max(out,dim=1)

            for y in pred.cpu():
                
                predict.append(y.item())
        
    with open("submission.csv","w") as f:
        f.write("Id,Class\n")
        for i,y in enumerate(predict):
            f.write("%d,%d\n"%(i,y))

Train(False)
Pred(False)