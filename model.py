import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

class Flatten(torch.nn.Module):

    def __init__(self, conv_dim):
        self.conv_dim = conv_dim

    def forward(self, x):
        return x.view(-1, self.conv_dim*4*4*4)

# helper conv function
def conv(index, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, relu=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, 
                           padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    if relu:
        layers.append(nn.ReLU(inplace=True))
    else:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    return nn.Sequential(*layers)

# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transpose convolutional layer, with optional batch normalization.
    """
    layers = []
    # append transpose conv layer
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

# Networks

class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value
        self.conv1 = conv(1, 3, conv_dim, 4, batch_norm=False) # x, y = 64 depth = 3
        self.conv2 = conv(2, conv_dim, conv_dim * 2, 4) # x, y = 32 depth = 64
        self.conv3 = conv(3, conv_dim * 2, conv_dim * 4, 4) # x, y = 16 depth = 128
        
        self.fc = nn.Linear(conv_dim*4*4*4, 1)
        self.out = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # define feedforward behavior
        x = self.conv1(x)
        # x = self.dropout(x)
        x = self.conv2(x)
        # x = self.dropout(x)
        x = self.conv3(x)
        # x = self.dropout(x)
        x = x.view(-1, self.conv_dim*4*4*4)
        
        x = self.fc(x)
        x = self.dropout(x)
        
        # x = self.out(x)
        
        return x

class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        # complete init function
        # conv_dim = 128

        self.conv_dim = conv_dim        
        self.fc = nn.Linear(z_size, conv_dim*4*4*4)
        
        self.t_conv1 = deconv(conv_dim*4, conv_dim*2, 4 )
        self.t_conv2 = deconv(conv_dim*2, conv_dim, 4)
        self.t_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        
        x = self.fc(x)
        x = self.dropout(x)
        
        x = x.view(-1, self.conv_dim*4, 4, 4)
        
        x = F.relu(self.t_conv1(x))
        # x = self.dropout(x)
        x = F.relu(self.t_conv2(x))
        # x = self.dropout(x)
        x = F.tanh(self.t_conv3(x))
        
        return x

class Extractor(nn.Module):
  def __init__(self, conv_dim, input_nc, nf, nclasses):
    super(Extractor, self).__init__()

    self.conv_dim = conv_dim

    self.conv1 = conv(1, input_nc, conv_dim, 4, batch_norm=False)
    self.conv2 = conv(2, conv_dim, conv_dim * 2, 4, batch_norm=True)
    self.conv3 = conv(3, conv_dim * 2, conv_dim * 4, 4, batch_norm=True)
    self.conv4 = conv(4, conv_dim * 4, conv_dim * 2, 4, 1, 0, batch_norm=False)
    self.conv5 = nn.Sequential().add_module('conv5', nn.Conv2d(conv_dim * 2, nclasses, 1, 1, 0, bias=False))
    
    # input is 32
 
  def forward(self, x):
    out1 = self.conv1(x)
    out2 = self.conv2(out1)
    out3 = self.conv3(out2)
    out4 = self.conv4(out3)
    out5 = self.conv5(out4)
    return out5, out4