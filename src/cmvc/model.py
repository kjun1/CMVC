from torch import nn
from cmvc import *





class Net(nn.Module):
  def __init__(self):
    
    
    super().__init__()
    self.ue = UttrEncoder()
    self.ud = UttrDecoder()
    self.fe = FaceEncoder()
    self.fd = FaceDecoder()
    self.ve = VoiceEncoder()

  def forward(self, x, y):
    z = self.ue(x)
    c = self.fe(y)
    x_hat = self.ud(z, c)
    print(x_hat.size())
    c_hat = self.ve(x_hat)
    print(c_hat.size())
    y_hat = self.fd(c_hat)

    return y_hat
  
  def loss(self, x, y):
    """
    reconstruction + KL divergence
    """
    pass