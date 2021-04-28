from torch import nn
from cmvc.scripts.layer import *





from math import ceil

def p(kernel_size, stride):
    if kernel_size == stride:
        return 0
    
    else:
        return ceil((kernel_size-1)/stride)


    
class UttrEncoder(nn.Module):
  def __init__(self):

    super().__init__()
    self.uttr_enc_d1 = CBGLayer(in_channels=1,
                                out_channels=16,
                                kernel_size=(3, 9),
                                stride = (1,1),
                                padding=(1,4))
    
    self.uttr_enc_d2 = CBGLayer(in_channels=16,
                                out_channels=32,
                                kernel_size=(4, 8),
                                stride = (2,2),
                                padding=(1,3))    

    self.uttr_enc_d3 = CBGLayer(in_channels=32,
                                out_channels=32,
                                kernel_size=(4, 8),
                                stride = (2,2),
                                padding=(1,3))    
    
    self.uttr_enc_d4 = nn.Conv2d(in_channels=32,
                             out_channels=16,
                             kernel_size=(9, 5),
                             stride = (9,1),
                             padding=(0,2))

  def uttr_encoder(self, x):
    """
    音声のencoder
    """

    x = self.uttr_enc_d1(x)
    x = self.uttr_enc_d2(x)
    x = self.uttr_enc_d3(x)
    x = self.uttr_enc_d4(x)

    mean, log_var = torch.split(x, 8, dim=1) # 半分
     
    return mean, log_var

  def uttr_sample_z(self, mean, log_var):
    """
    潜在変数出すやつ
    """
    epsilon = torch.randn(mean.shape).to(device)
    return mean + torch.exp(log_var) * epsilon

  def forward(self, x):
    mean, log_var = self.uttr_encoder(x)
    z = self.uttr_sample_z(mean, log_var)
    return z



class FaceEncoder(nn.Module):
  def __init__(self):

    super().__init__()
    self.face_enc_d1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                               out_channels=32,
                                               kernel_size=(6, 6),
                                               stride = (2,2),
                                               padding=(2,2)),
                                     nn.LeakyReLU())

    self.face_enc_d2 = CBLLayer(in_channels=32,
                                out_channels=64,
                                kernel_size=(6, 6),
                                stride = (2,2),
                                padding=(2,2))
    
    self.face_enc_d3 = CBLLayer(in_channels=64,
                                out_channels=128,
                                kernel_size=(4, 4),
                                stride = (2,2),
                                padding=(1,1))
    
    self.face_enc_d4 = CBLLayer(in_channels=128,
                                out_channels=128,
                                kernel_size=(4, 4),
                                stride = (2,2),
                                padding=(1,1)
                                )
    
    self.face_enc_d5 = CBLLayer(in_channels=128,
                                out_channels=256,
                                kernel_size=(4, 4),
                                stride = (2,2),
                                padding=(1,1))
    
    self.face_enc_d6 = FlattenLayer() #flattenまでに(n, c, 1, 1)になってる前提

    self.face_enc_d7 = nn.Sequential(nn.Linear(256,256),
                                     nn.LeakyReLU())
    
    self.face_enc_d8 = nn.Sequential(nn.Linear(256,16),
                                     nn.LeakyReLU())

  def face_encoder(self, y):
    """
    顔面のencoder
    """

    y = self.face_enc_d1(y)
    y = self.face_enc_d2(y)
    y = self.face_enc_d3(y)
    y = self.face_enc_d4(y)
    y = self.face_enc_d5(y)
    y = self.face_enc_d6(y)
    y = self.face_enc_d7(y)
    y = self.face_enc_d8(y)

    
    
    mean, log_var = torch.split(y, 8, dim=1) # 半分
     
    return mean, log_var

  def face_sample_z(self, mean, log_var):
    """
    顔面の潜在変数出すやつ
    """
    epsilon = torch.randn(mean.shape).to(device)
    return mean + torch.exp(log_var) * epsilon

  def forward(self, y):
    mean, log_var = self.face_encoder(y)
    z = self.face_sample_z(mean, log_var)
    z = z.unsqueeze(-1).unsqueeze(-1)
    return z



class VoiceEncoder(nn.Module):
  def __init__(self):

    super().__init__()
    self.voice_enc_d1 = CBGLayer(in_channels=1,
                                 out_channels=32,
                                 kernel_size=(3,9),
                                 stride=(1,1),
                                 padding=(1,4))
    
    self.voice_enc_d2 = CBGLayer(in_channels=32,
                                 out_channels=64,
                                 kernel_size=(4,8),
                                 stride=(2,2),
                                 padding=(1,3))
    
    self.voice_enc_d3 = CBGLayer(in_channels=64,
                                 out_channels=128,
                                 kernel_size=(4,8),
                                 stride=(2,2),
                                 padding=(1,3))
    
    self.voice_enc_d4 = CBGLayer(in_channels=128,
                                 out_channels=128,
                                 kernel_size=(4,8),
                                 stride=(2,2),
                                 padding=(1,3))
    
    self.voice_enc_d5 = CBGLayer(in_channels=128,
                                 out_channels=128,
                                 kernel_size=(4,5),
                                 stride=(4,1),
                                 padding=(0,2))
    
    self.voice_enc_d6 = CBGLayer(in_channels=128,
                                 out_channels=64,
                                 kernel_size=(1,5),
                                 stride=(1,1),
                                 padding=(0,2))
    
    self.voice_enc_d7 = nn.Conv2d(in_channels=64,
                                  out_channels=16,
                                  kernel_size=(1,5),
                                  stride=(1,1),
                                 padding=(0,2))
                                 

  def voice_encoder(self, x):
    """
    顔面持ってくるencoder
    """
    x = self.voice_enc_d1(x)
    x = self.voice_enc_d2(x)
    x = self.voice_enc_d3(x)
    x = self.voice_enc_d4(x)
    x = self.voice_enc_d5(x)
    x = self.voice_enc_d6(x)
    x = self.voice_enc_d7(x)
        
    """
    第4層のConv2d出力のchannelの半分でmean半分でlog_varを予測している？
    """
    mean, log_var = torch.split(x, 8, dim=1) # 半分
     
    return mean, log_var

  def voice_sample_z(self, mean, log_var):
    """
    音声の潜在変数出すやつ
    """
    epsilon = torch.randn(mean.shape).to(device)
    return mean + torch.exp(log_var) * epsilon

  def forward(self, x):
    mean, log_var = self.voice_encoder(x)
    z = self.voice_sample_z(mean, log_var)

    z = z.squeeze(-1).squeeze(-1)
    return z
