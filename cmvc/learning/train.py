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
             optimizer,
             n_iter=10, device="cpu",
             state=None, model_dir=None):
    train_losses = []
    
    d = list(model_dir.iterdir())
    if d:
        now_iter = max([int(i.stem) for i in d])
        
        checkpoint = torch.load(model_dir/(str(now_iter).zfill(4)+".cpt"))
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print(now_iter)
    else:
        now_iter = 1

    for epoch in range(n_iter):
        running_loss = 0.0
        
        net.train()
        
        for i, (xx, yy) in enumerate(train_loader):
            optimizer.zero_grad()
            losses = torch.zeros(1).to(device)
            
            for batch in range(len(xx)):
                voice = xx[batch].to(device)
                image = yy[batch].to(device)
                #print(voice.shape, image.shape)
                loss = net.loss(voice, image)
                losses += loss
            
            losses.backward()
            optimizer.step()
            print(losses.item(), flush=True)
            
            running_loss += losses.item()
        
        
        train_losses.append(running_loss / (i+1))
        if (model_dir and state):
            file_path = model_dir / (str(epoch + now_iter).zfill(4) + ".cpt")
            torch.save(state,  file_path)
        print(epoch+now_iter, train_losses[-1], flush=True)


device = "cpu"

dict_toml = toml.load(open('/home/jun/Documents/CMVC/cmvc/config.toml'))

image_path = dict_toml["path"]["dataset"]["processing"]["image"]
voice_path = dict_toml["path"]["dataset"]["processing"]["voice"]


dataset = PairDataset(voice_path=voice_path , train=True, image_path=image_path)

batch_size = 1

trainloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 0, collate_fn = collate_fn)

net = Net()
#net.float()
optimizer = optim.Adam(net.parameters(), lr=0.001)

print(device)

net.to(device)

state = {
    'net': net.state_dict(),
    'optimizer': optimizer.state_dict(),
}
model_dir = pathlib.Path('/home/jun/Documents/CMVC/checkpoint')

train_net(net, trainloader, optimizer, n_iter=1000, device=device, state=state, model_dir=model_dir)
