import toml
import pathlib

import torch
from torch import optim

from cmvc import *
from cmvc.utils.data.dataset import PairDataset


def collate_fn(batch):
    # batchはDatasetの返り値 (タプル) のリスト
    voices, images = [], []
    for voice, image, label in batch:
        voices.append(voice)
        images.append(image)
        #labels.append(label)
    
    
    # labelsはTensorリストのまま

    return voices, images

def train_net(net, train_loader,
             optimizer_cls=optim.Adam,
             n_iter=10, device="cpu"):
    train_losses = []
    
    optimizer = optimizer_cls(net.parameters(), lr=0.001)
    for epoch in range(n_iter):
        running_loss = 0.0
        
        net.train()
        
        for i, (xx, yy) in enumerate(train_loader):
            optimizer.zero_grad()
            losses = torch.zeros(1).to(device)
            
            for batch in range(len(xx)):
                voice = xx[batch].to(device)
                image = yy[batch].to(device)
                loss = net.loss(voice, image)
                losses += loss
            
            losses.backward()
            optimizer.step()
            print(losses.item(), flush=True)
            
            running_loss += losses.item()
        
        
        train_losses.append(running_loss / i)
        
        print(epoch, train_losses[-1], flush=True)


device = "cpu"

dict_toml = toml.load(open('/home/jun/Documents/CMVC/cmvc/config.toml'))

image_path = dict_toml["path"]["dataset"]["processing"]["image"]
voice_path = dict_toml["path"]["dataset"]["processing"]["voice"]


dataset = PairDataset(voice_path=voice_path , train=True, image_path=image_path)

batch_size = 32

trainloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 0, collate_fn = collate_fn)
net = Net()
        
        
net.to(device)
train_net(net, trainloader)