from torch import nn 





class CBGLayer(nn.Module):
  """
  Conv+Bn+GLU
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride,padding=0):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride = stride,
                           padding=padding)
    self.conv2 = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride = stride,
                           padding=padding)

    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)


  def forward(self, x):
    x1 = self.bn1(self.conv1(x))
    x2 = self.bn2(self.conv2(x))

    x = torch.cat((x1,x2),1)
    x = nn.functional.glu(x,1)
    
    return x



class CBLLayer(nn.Module):
  """
  Conv+Bn+LReLU
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride,padding=0):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride = stride,
                          padding=padding)

    self.bn = nn.BatchNorm2d(out_channels)

    self.lrelu = nn.LeakyReLU()


  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.lrelu(x)

    return x



class DBGLayer(nn.Module):
  """
  Deconv + Bn + GLU
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride,padding=0):
    super().__init__()
    self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding)
    self.deconv2 = nn.ConvTranspose2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding)
    
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)



  def forward(self, x):
    x1 = self.bn1(self.deconv1(x))
    x2 = self.bn2(self.deconv2(x))

    x = torch.cat((x1,x2),1)
    x = nn.functional.glu(x,1)
    
    return x



class DBSLayer(nn.Module):
  """
  Deconv + Bn + SoftPlus
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
    super().__init__()
    self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)
    
    self.bn = nn.BatchNorm2d(out_channels)

    self.softplus = nn.Softplus()


  def forward(self, x):
    x = self.deconv(x)
    x = self.bn(x)
    x = self.softplus(x)

    return x



class FlattenLayer(nn.Module):
  """
  (N, C, H, W)を(N, C*H*W)にする
  """
  def forward(self, x):
    sizes = x.size()
    return x.view(sizes[0],  -1)

class ReshapeLayer(nn.Module):
  """
  (N, C*H*W)を(N, C, H, W)にする
  """
  def forward(self, x, out_channel):
    sizes = x.size()
    h = int((sizes[1]/out_channel)**0.5)
    return x.view(sizes[0],  out_channel, h, h)