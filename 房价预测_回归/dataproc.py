import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import settings
import os



class MyDataset(Dataset):
    def __init__(self,mode=1):
        self.mode=mode
        if (mode==1):
            path=os.path.join(settings.DATA_DIR,"processed_train.csv")
            self.X=np.genfromtxt(path,delimiter=",",dtype=np.float32)[1:,1:-1]
            self.y=np.genfromtxt(path,delimiter=",",dtype=np.float32)[1:,[-1]]
            
        elif(mode==2):
            path=os.path.join(settings.DATA_DIR,"processed_val.csv")
            self.X=np.genfromtxt(path,delimiter=",",dtype=np.float32)[1:,1:-1]
            self.y=np.genfromtxt(path,delimiter=",",dtype=np.float32)[1:,[-1]]
        
        else:
            path=os.path.join(settings.DATA_DIR,"processed_test.csv")
            self.X=np.genfromtxt(path,delimiter=",",dtype=np.float32)[1:,1:]
       
        self.len=self.X.shape[0]
    def __getitem__(self, index):
        if(self.mode==3):
            return self.X[index]
        
        return self.X[index],self.y[index]
        

    def __len__(self):
        return self.len


def GetProcessedData(mode=1):
    data_iter=DataLoader(dataset=MyDataset(mode),batch_size=settings.BATCH_SIZE,shuffle=True)
    
    return data_iter

if __name__=='__main__':
    for i in GetProcessedData(3):
        print(i)
        break