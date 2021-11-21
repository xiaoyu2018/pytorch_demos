from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from settings import *
import os

# 猫0狗1

class DCDataset(Dataset):
    def __init__(self,train:bool):
        imgs=[]
        path=DATA_DIR

        if(train==True):
            path=os.path.join(path,"training_set")
        else:
            path=os.path.join(path,"test_set")

        for file in os.listdir(os.path.join(path,"cats")):
                imgs.append((os.path.join(path,"cats",file),0))
        for file in os.listdir(os.path.join(path,"dogs")):
                imgs.append(((os.path.join(path,"dogs",file),1)))

        self.imgs=imgs
        self.len=len(imgs)
    def __getitem__(self, index):
        fp,label=self.imgs[index]
        img=Image.open(fp).convert("RGB")

        transform=self.GetTransform()
        img=transform(img)
     
        return img,label
    
    def __len__(self):
        return self.len

    def GetTransform(self):

        transform=transforms.Compose(
            [
                transforms.Resize((32,32)),
                # 自动归一化了，为每个像素除以了255
                transforms.ToTensor(),
            ]
        )
        return transform
    



def LoadData(train=True):
    dataset=DCDataset(train)
    data_iter=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
    return data_iter

