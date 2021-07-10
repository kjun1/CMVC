from random import sample
import pathlib
import itertools

import numpy as np
import pandas as pd
import cv2
import torch

from . import transform 


class PairDataset(torch.utils.data.Dataset):
    

    def __init__(self, voice_path, image_path, train=True):
        # path定義
        p = pathlib.Path(voice_path)
        if train:
            self.voice_path = p / "train"
        else:
            self.voice_path = p / "eval"
        
        self.image_path = pathlib.Path(image_path)
        
        
        voice_dir = [i for i in self.voice_path.iterdir() if i.is_dir()]       
        self.voice_file = list(itertools.chain.from_iterable([[j for j in i.iterdir()] for i in voice_dir]))
        
        self.voice_data = [pd.read_pickle(i) for i in self.voice_file]
        self.voice_label = [1 if "M" == str(i.parent)[-2] else -1 for i in self.voice_file]
        
        k = np.concatenate(self.voice_data,axis=1)
        self.voice_transform = transform.VoiceTrans(k.max(), k.min())
        
        
        self.voice_data = [torch.tensor([[self.voice_transform(i)]], dtype=torch.float32) for i in self.voice_data]
        
        
        
        
        
        image_dir = self.image_path
        image_file = [i for i in image_dir.iterdir() if i.suffix == ".jpg"]
        
        
        image_df = pd.read_table(self.image_path / "list_attr_celeba.txt", header=0,sep=" ", index_col=0)
        self.image_label = [image_df["Male"][i.name] for i  in image_file]
        self.image_label_male = [i for i, x in enumerate(self.image_label) if x == 1]
        self.image_label_female= [i for i, x in enumerate(self.image_label) if x == -1]
        
        self.image_transform = transform.ImageTrans()
        
        self.image_data = [torch.tensor(self.image_transform(cv2.imread(str(i))), dtype=torch.float32) for i in image_file]
 
        
    
    
        self.datanum = len(self.voice_data)
    
    
    
    def __len__(self):
        return self.datanum

    def __getitem__(self, idx, k=2):
        
        d = self.voice_file[idx]
        out_voice_data = self.voice_data[idx]
        out_label = self.voice_label[idx]
        
        if out_label == 1:
            c = sample(self.image_label_male, k)
        else:
            c = sample(self.image_label_female, k)
        
        out_image_data = torch.stack([self.image_data[i] for i in c])
        
        print(c)
        print(d)

        return out_voice_data, out_image_data, out_label