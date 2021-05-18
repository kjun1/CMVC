import torch 
from torch import nn
from torch.nn import functional as F
from cmvc.models import (UttrEncoder, UttrDecoder, FaceEncoder, FaceDecoder,
                        VoiceEncoder)


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
        # print(x_hat.size())
        c_hat = self.ve(x_hat)
        print(c_hat.size())
        #print(c_hat[:,:,:,0].size())
        y_hat = self.fd(c_hat)

        return y_hat

    def loss(self, x, y):
        ue_mean, ue_log_var = self.ue.uttr_encoder(x)
        ue_KL = -0.5 * torch.mean(torch.sum(1 + ue_log_var - ue_mean**2 - torch.exp(ue_log_var)))
        
        z = self.ue.uttr_sample_z(ue_mean, ue_log_var)

        fe_mean, fe_log_var = self.fe.face_encoder(y)
        fe_KL = -0.5 * torch.mean(torch.sum(1 + fe_log_var - fe_mean**2 - torch.exp(fe_log_var)))
        
        c = self.fe.face_sample_z(fe_mean, fe_log_var)
  
        
        x_hat = self.ud(z, c)
        
        c_hat = self.ve(x_hat)

        y_hat = self.fd(c_hat)

        """
        reconstruction_MSE(uttr, face, voice) + KL divergence(uttr, face)
        """
        
        lower_bound = [-ue_KL, -fe_KL]
        
        return sum(lower_bound)
