import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .convs import *
from .utils import *

class Encoder(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, activation=nn.LeakyReLU(inplace=True), stages=3
                 , skips=False):
        self.act = activation
        self.conv_in = Conv3D(in_channels,hidden_channels)
        self.conv_out = Conv3D(hidden_channels,out_channels)
        self.pool = nn.MaxPool3d(2)
        self.convs = nn.ModuleList([ConvBlock3D(3, hidden_channels, hidden_channels, activation=activation,
                                                batch_norm=True, residual=True)]*stages)
        self.skips=skips
        self.pre_forward = self.forward_vanilla
        self.pre_forward = self.forward_skip if skips else self.forward_vanilla
        
    def set_skips(self):
        self.skips=True
        self.pre_forward = self.forward_skip
        
    def unset_skips(self):
        self.skips=False
        self.pre_forward = self.forward_vanilla
        
    def forward_vanilla(self, x):
        x = self.act(self.conv_in(x))
        for conv in self.convs:
            x = self.pool(conv(x))
        x = self.act(self.conv_out(x))
        return x
    
    def forward_skip(self, x):
        x =self.convs[0](self.act(self.conv_in(x)))
        skips = []
        for conv in self.convs[1:]:
            skips = [x] + skips
            x = self.pool(x)
            x = conv(x)
        x = self.act(self.conv_out(x))
        return x,skips
    
    def forward(self, x):
        return self.pre_forward(x)
    
            
class Decoder(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 activation=nn.LeakyReLU(inplace=True), stages=3, skips=False):
        self.act = activation
        self.conv_in = Conv3D(in_channels,hidden_channels)
        self.conv_out = Conv3D(hidden_channels,out_channels)
        self.unpool = UpConv3D(up_scale=2)
        self.convs = nn.ModuleList([ConvBlock3D(3, hidden_channels, hidden_channels, activation=activation,
                                                batch_norm=True, residual=True)]*stages)
        self.skips = skips
        self.pre_forward = self.forward_vanilla
        self.pre_forward = self.forward_skip if skips else self.forward_vanilla
        
    def set_skips(self):
        self.skips=True
        self.pre_forward = self.forward_skip
        
    def unset_skips(self):
        self.skips=False
        self.pre_forward = self.forward_vanilla
    
    def forward_vanilla(self, x):
        x = self.act(self.conv_in(x))
        for conv in self.convs:
            x =  self.unpool(conv(x))
        x = self.act(self.conv_out(x))
        return x
        
    def forward_skip(self, x_skips):
        x, skips = x_skips
        x = self.convs[0](self.act(self.conv_in(x)))
        for conv,skip in zip(self.convs[1:],skips):
            x = self.act(self.unpool(x)) + skip
            x = conv(x)
        x = self.act(self.conv_out(x))
        return x
    
    def forward(self, x):
        return self.pre_forward(x)
