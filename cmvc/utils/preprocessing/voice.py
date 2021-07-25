import toml
import os
import librosa
import numpy as np
import pandas as pd

from pathlib import Path
from cmvc.utils.data.voice import Wave
from tqdm import tqdm

dict_toml = toml.load(open('/home/jun/Documents/CMVC/cmvc/config.toml'))


voice_path = dict_toml["path"]["dataset"]["voice"]

train_path = voice_path + "/vcc2018_database_training/vcc2018_training/"
eval_path = voice_path + "/vcc2018_database_evaluation/vcc2018_evaluation/"

output_path = os.getcwd()+"/voice/"

print(output_path)

fs = 22050

# trainファイル作成

# inputファイル認識

for i in tqdm(os.listdir(eval_path)):
    if i[:3] != "VCC":
        continue
    # ファイルのやつ　train_path+i
    os.makedirs(output_path+"eval/"+i, exist_ok=True)

    for j in os.listdir(eval_path+i):
        wave = Wave(wave_name=j, person_name=i, path=Path(i+"/"+j), data_path=Path(eval_path), ex_mc=True, ex_mfcc=True)
        pd.to_pickle(wave, output_path+"eval/"+i+"/"+j[:-4]+".pkl")
# 保存



