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
        
        return x_hat

    def loss(self, x, y):
        ue_mean, ue_log_var = self.ue.uttr_encoder(x)
        ue_KL = -0.5 * torch.mean(torch.sum(1 + ue_log_var - ue_mean**2 - torch.exp(ue_log_var)))
        
        z = self.ue.uttr_sample_z(ue_mean, ue_log_var)

        fe_mean, fe_log_var = self.fe.face_encoder(y)
        fe_KL = 0.5 * torch.mean(torch.sum(1 + fe_log_var - fe_mean**2 - torch.exp(fe_log_var)))
        
        c = self.fe.face_sample_z(fe_mean, fe_log_var)
  
        x_hat = self.ud(z, c)
    
        uttr_MSE = F.mse_loss(x, x_hat)

        #print(c.size())
        c = c.squeeze(-1).squeeze(-1)
        y_hat_1 = self.fd(c)
        
        #print(y.size())
        #print(y_hat_1.size())
        face_MSE = F.mse_loss(y, y_hat_1)
        
        
        c_hat = self.ve(x_hat)

        l = []
        s = c_hat.size()[-1]
        for i in range(s):
            
            k = c_hat[:,:,:,i].squeeze(-1)
            #print(k.size())
            y_hat_2 = self.fd(k)
            voice_MSE = F.mse_loss(y, y_hat_2)/s
            l.append(voice_MSE)
            

        """
        reconstruction_MSE(uttr, face, voice) + KL divergence(uttr, face)
        """
        print(-ue_KL, -fe_KL, uttr_MSE, face_MSE, l)
        lower_bound = [-ue_KL, -fe_KL, uttr_MSE, face_MSE]
        lower_bound.extend(l)
        
        
        return sum(lower_bound)

    def test(self, x, y):
        ue_mean, ue_log_var = self.ue.uttr_encoder(x)
        ue_KL = -0.5 * torch.mean(torch.sum(1 + ue_log_var - ue_mean**2 - torch.exp(ue_log_var)))
        
        return ue_KL