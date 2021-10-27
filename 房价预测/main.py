import torch
from torch import nn
from torch import optim
import settings,models,dataproc
import pandas as pd


net=models.MyNet()
creterite=nn.MSELoss()
opt=optim.Adam(net.parameters(),lr=1.15)

# 提交时的评判函数
def LogRmse(pred,y):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定 
        clipped_preds = torch.max(pred, torch.tensor(1.0))
        rmse = torch.sqrt(creterite(clipped_preds.log(), y.log()))
    return rmse

def Train():
    train_iter=dataproc.GetProcessedData(1)
    val_iter=dataproc.GetProcessedData(2)
    min_loss=float("inf")
    for epoch in range(settings.EPOCH):
        train_loss=0
        val_loss=0
        vv_l=0
        for X,y in train_iter:
            out=net(X)
            l=creterite(out,y)

            opt.zero_grad()
            l.backward()
            opt.step()

            train_loss+=LogRmse(out,y).item()
       
        with torch.no_grad():
            for X,y in val_iter:
                out=net(X)
                l=LogRmse(out,y)
                v_l=creterite(out,y)
                vv_l+=v_l.item()
                val_loss+=l.item()
                
            if(vv_l<min_loss):
                min_loss=vv_l
                print("save...")
                torch.save(net.state_dict(),"./param.pkl")
        print("epoch %d,train_loss %f,val_loss %f"%((epoch+1),train_loss,val_loss))


def Pred():
    net.load_state_dict(torch.load('./param.pkl'))
    
    test_iter=dataproc.GetProcessedData(3)
    
    res=[i.item() for X in test_iter for i in net(X)]
    # print(enumerate(res))   
    submission = pd.DataFrame(enumerate(res,start=1461))
    submission.to_csv('./submission.csv', index=False)

Train()
# a=torch.tensor([-0.8734663783676423,-0.22727851670571111,-0.35021115192675084,0.6460727010653011,-0.5071973005920356,1.14511624057831,1.0402588194090898,-0.10149378518817222,0.532420985484946,-0.29302957385881423,-0.10413551236522488,0.3340147071163795,0.15398606853548685,-0.7848905642207595,-0.10117968082459,-0.5548053935898709,1.0864638056895892,-0.2497667209235715,0.7812319615337907,-0.7561914703898879,-1.0456221209685232,-0.2076628944867197,-0.2877089940430983,-0.9241528710642078,1.0904154115728633,0.3064227596279195,0.738761647444418,0.08923158307858196,0.09638389779396296,-0.3595391482902645,-0.10331282968963837,-0.28588647550360613,-0.06313935545242814,-0.08957661223756516,0.2898645965816903,-1.363335096808056,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0])
# print(net(a).item())

# net.load_state_dict(torch.load('./param.pkl'))
# print(net(a).item())

Pred()