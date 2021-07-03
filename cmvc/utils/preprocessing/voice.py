import toml
import os
import librosa
import numpy as np
import pandas as pd



dict_toml = toml.load(open('/home/jun/Documents/CMVC/cmvc/config.toml'))


voice_path = dict_toml["path"]["dataset"]["voice"]

train_path = voice_path + "/vcc2018_database_training/vcc2018_training/"
eval_path = voice_path + "/vcc2018_database_evaluation/vcc2018_evaluation/"

output_path = os.getcwd()+"/voice/"

print(output_path)

fs = 22050

# trainファイル作成

# inputファイル認識

for i in os.listdir(train_path):
    if i[:3] != "VCC":
        continue
    # ファイルのやつ　train_path+i
    os.makedirs(output_path+"train/"+i, exist_ok=True)

    for j in os.listdir(train_path+i):
        x, fs = librosa.load(train_path+i+"/"+j, sr=fs, dtype=np.float64)# dtypeを忘れずに
        mccs = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=36)
        #print(output_path+"train/"+i+j[:-4])
        pd.to_pickle(mccs, output_path+"train/"+i+"/"+j[:-4]+".pkl")
# 保存



