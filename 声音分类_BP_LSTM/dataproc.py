from torch._C import dtype
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import settings
import torch 

class Timit11(Dataset):
    def __init__(self,X,y=None):
        self.X=torch.tensor(X,dtype=torch.float32)

        if(y is not None):
            self.y=torch.LongTensor(y)
        else:
            self.y=None

        self.len=self.X.shape[0]
    def __getitem__(self, index):
        if(self.y is not None):
            return self.X[index],self.y[index]
        return self.X[index]

    def __len__(self):
        return self.len


def GetTimit11(train=True):
    if(train==True):
        
        X=np.load(os.path.join(settings.DATA_DIR,"train_11.npy"))
        y=np.load(os.path.join(settings.DATA_DIR,"train_label_11.npy")).astype(np.int64)
        val_index=int(settings.VAL_RATIO*y.size)

        x_train,y_train,x_val,y_val=X[val_index:],y[val_index:],X[:val_index],y[:val_index]
        ret=(DataLoader(dataset=Timit11(x_train,y_train),shuffle=True,batch_size=settings.BATCH_SIZE),
            DataLoader(dataset=Timit11(x_val,y_val),batch_size=settings.BATCH_SIZE))
    else:
        X=np.load(os.path.join(settings.DATA_DIR,"test_11.npy"))
        
        ret=DataLoader(dataset=Timit11(X),batch_size=settings.BATCH_SIZE)
    
    return ret


if __name__=='__main__':

    train,val = GetTimit11()
    s1=0
    s2=0
    for X,y in train:
        s1+=y.size(0)
        
    for X,y in val:
        s2+=y.size(0)
    print(s1,s2)