from torch import nn
from cmvc.scripts.layer import *





class UttrDecoder(nn.Module):
  def __init__(self):

    super().__init__()

    self.uttr_dec_d1 = DBGLayer(in_channels=8,
                                out_channels=16,
                                kernel_size=(9,5),
                                stride=(9,1),
                                padding=(0,2))
  
    self.uttr_dec_d2 = DBGLayer(in_channels=16,
                                out_channels=16,
                                kernel_size=(4,8),
                                stride=(2,2),
                                padding=(1,3))
  
    self.uttr_dec_d3 = DBGLayer(in_channels=16,
                                out_channels=8,
                                kernel_size=(4,8),
                                stride=(2,2),
                                padding=(1,3))
  
    self.uttr_dec_d4 = nn.ConvTranspose2d(in_channels=8,
                                          out_channels=2,
                                          kernel_size=(3,9),
                                          stride=(1,1),
                                          padding=(1,4))
  
  def uttr_decoder(self, z, c):
    #print(z.size())
    
    x,_ = torch.broadcast_tensors(z, c)
    x = self.uttr_dec_d1(x)
    #print(x.size())
    
    x,_ = torch.broadcast_tensors(x, torch.cat((c, c),1))
    x = self.uttr_dec_d2(x)
    #print(x.size())

    x,_ = torch.broadcast_tensors(x, torch.cat((c, c),1))
    x = self.uttr_dec_d3(x)
    #print(x.size())

    x,_ = torch.broadcast_tensors(x, c)
    x = self.uttr_dec_d4(x)
    #print(x.size())

    mean, log_var = torch.split(x, 1, dim=1) # 半分
     
    return mean, log_var
  

  def uttr_sample_z(self, mean, log_var):

    epsilon = torch.randn(mean.shape).to(device)
    return mean + torch.exp(log_var) * epsilon
  

  def forward(self, z, c):
    mean, log_var = self.uttr_decoder(z, c)
    z = self.uttr_sample_z(mean, log_var)
    return z



class FaceDecoder(nn.Module):
  def __init__(self):

    super().__init__()
    self.face_dec_d1 = nn.Sequential(nn.Linear(8,128),
                                     nn.Softplus())
    
    self.face_dec_d2 = nn.Sequential(nn.Linear(128,2048),
                                     nn.Softplus())
    
    self.face_dec_d3 = ReshapeLayer()

    self.face_dec_d4 = DBSLayer(in_channels=128,
                                out_channels=128,
                                kernel_size=(3,3),
                                stride=(2,2),
                                padding=(2,2))

    self.face_dec_d5 = DBSLayer(in_channels=128,
                                out_channels=128,
                                kernel_size=(6,6),
                                stride=(2,2),
                                padding=(2,2))
    
    self.face_dec_d6 = DBSLayer(in_channels=128,
                                out_channels=64,
                                kernel_size=(6,6),
                                stride=(2,2),
                                padding=(2,2))
    
    self.face_dec_d7 = DBSLayer(in_channels=64,
                                out_channels=32,
                                kernel_size=(6,6),
                                stride=(2,2),
                                padding=(2,2))
    
    self.face_dec_d8 = nn.Conv2d(in_channels=32,
                                 out_channels=6,
                                 kernel_size=(5,5),
                                 stride=(1,1))
    
    

  def face_decoder(self, c):

    y = self.face_dec_d1(c)
    y = self.face_dec_d2(y)
    y = self.face_dec_d3(y, 128)
    y = self.face_dec_d4(y)
    y = self.face_dec_d5(y)
    y = self.face_dec_d6(y)
    y = self.face_dec_d7(y)
    y = self.face_dec_d8(y)

    mean, log_var = torch.split(y, 3, dim=1) # 半分
     
    return mean, log_var

  def face_sample_z(self, mean, log_var):
    """
    顔面の潜在変数出すやつ
    """
    epsilon = torch.randn(mean.shape).to(device)
    return mean + torch.exp(log_var) * epsilon

  def forward(self, y):
    mean, log_var = self.face_decoder(y)
    z = self.face_sample_z(mean, log_var)
    return z
