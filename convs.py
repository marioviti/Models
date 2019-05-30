import torch.nn as nn
from .utils import *
import numpy as np


class MeanGPool(nn.Module):
    def __init__(self):
        super(MeanGPool, self).__init__()
        
    def forward(self,x):
        x = x.view(*x.shape[0:2],-1).mean(-1)
        return x
    
    
class StdGPool(nn.Module):
    def __init__(self):
        super(StdGPool, self).__init__()
        
    def forward(self,x):
        x = x.view(*x.shape[0:2],-1).std(-1)
        return x
    
    
class ExpandGFunction(nn.Module):        
    def __init__(self, merge=True):
        super(ExpandGFunction, self).__init__()
        self.pre_forward = self.expand_forward
        self.pre_forward = self.merge_forward if merge else self.pre_forward
        
    def expand_forward(self,x_l,x_g):
        spatial_dims = x_l.shape[2:]
        bs,ch_g = x_g.shape
        x_g = x_g.unsqueeze(-1).unsqueeze(-1)
        x_g = x_g.expand(-1,-1,*spatial_dims)
        return x_g
        
    def merge_forward(self,x_l,x_g):
        x_g = self.expand_forward(x_l,x_g)
        x = torch.cat((x_l,x_g),1)
        return x
        
    def forward(self,x_l,x_g):
        return pre_forward(x_l,x_g)

    
class GFunction(nn.Module):
    """
    Implement spatial invaiant feature.
    This Gfunction are invariant to permutation of pixels therefore
    they are invariant to spatial transformation/deformation of the input.
    
    Mixing these features with transform variant features makes them variant aswell
    however it can be shown that they perform similar to histog
    
    Arguments:
    in_channels: int
        input_channels.
    global_channels: int
        output channels.
    stride: int
        stride of convolution.
    groups: int
        depthwise convolution factor.
    kernel_size: int
        size of convolution kernel.
    bias: bool
        if bias is added to convolution output.
    activatiton: nn.Module
        activation function.
    mode: str
        {avg, std, avg_std} define the type of global pooling function.
        in case avg given an input of bs,in_channels,h,w,d the output is
        bs,in_channels+global_channels,h,w,d in case of avg_std 
        bs,in_channels+global_channels*2,h,w,d.
    """
    def __init__(self, in_channels, global_channels, stride=1, groups=1, kernel_size=1, 
                 bias=True, activation=nn.LeakyReLU(inplace=True),
                 mode='avg', padding='same', merge=True, conv_mode='3d'):
        
        self.conv = Conv(in_channels, global_channels, 1, stride=stride,
                           padding=padding, groups=groups_in, bias=bias, mode=conv_mode)
        self.act = activation
        self.mean_pool = MeanGPool()
        self.std_pool = StdGPool()
        self.expander = ExpandGFunction(merge=merge)

        # instruction prefetch
        self.pre_forward = self.forward_avg_mode
        self.pre_forward = self.forward_avg_mode if mode == 'std' else self.pre_forward
        self.pre_forward = self.forward_avg_std_mode if mode == 'avg_std' else self.pre_forward
        
    def forward_avg_mode(self, x):
        return self.mean_pool(x)
    
    def forward_std_mode(self, x):
        return self.std_pool(x)
    
    def forward_avg_std_mode(self, x):
        return torch.cat((self.forward_std_mode(x),
                          self.forward_avg_mode(x)),1)
        
    def forward(self, x):
        return self.expander(self.pre_forward(x),x)
    
class UpSample(nn.Module):
    """
    Implementing generic upsampling strategy
    Parameters:
    up_scale: int
        as a result spatial dimensions of the output will be scaled by up_scale.
    """
    def __init__(self, up_scale=2, mode='3d'):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=up_scale, mode='trilinear' if mode =='3d' else 'bilinear')
        
    def forward(self, x):
        return self.up(x)
    
    
class UpConv(nn.Module):
    """
    Implementing generic upsampling strategy
    Parameters:
    up_scale: int
        as a result spatial dimensions of the output will be scaled by up_scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, groups=1, up_scale=2, bias=True, mode='3d'):
        super(UpConv, self).__init__()
        padding_in = compute_upconv_input_padding(2,4,up_scale,1,2*up_scale,output_padding=0)
        conv_type = nn.ConvTranspose3d if mode == '3d' else nn.ConvTranspose2d
        self.up = nn.Sequential(conv_type(in_channels, in_channels, kernel_size=2*up_scale, dilatation=dilatation,
                                          stride=up_scale, padding=padding_in), self.act)
            
    def forward(self, x):
        x = self.up(x)
        return x
    

class Conv(nn.Module):
    """
    Wrapping Conv3D with advanced padding options.
    args:
        in_channels, out_channels, kernel_size, dilatation, 
        groups, stride, bias: look at torch.nn.Conv3d
        padding: string {same, valid}
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, groups=1, bias=True,
                 stride=1, padding='same',mode='3d'):
        super(Conv, self).__init__()
        conv_type = nn.Conv3d if mode =='3d' else nn.Conv2d
        self.layer = conv_type(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                               dilation=dilation, groups=groups, bias=bias)
        self.stride, self.dilation, self.padding, self.kernel_size = stride, dilation, padding, kernel_size

    def forward(self, x):
        input_shape = list(x.shape)[:3]
        pad_func, output_padding = conv_padding_func(self.padding, input_shape, self.kernel_size, self.dilation, self.stride)
        return self.layer(pad_func(x))

            
class ConvBlock(nn.Module):
    """
    ConvBlock3D is the main building block of a convolutional neural network, it is parametrized to cover
    most use cases for conv nets like resblocks and batch norm.
    It mainly consists of a series of convolutions each followed by an activation function.
        
    the first convolution transforms:
        R^in_channels->R^out_channels
    the num_blocks - 1 following convolutions transforms: 
        R^out_channels->R^out_channels.
    
    Parametrizations:
    - residual and batch_norm are unset:
    act_in---|conv_0|act|---...---|conv_i|act|...---|conv_num_blocks|act|---act_out
                                
    - residual is unset and batch_norm is set:
    act_in---|conv_0|act|---...---|conv_i|act|---|bn_i|...---|conv_num_blocks|act|---|bn_num_blocks|---act_outù
    
    - residual is set and batch_norm is unset:
    act_in---|conv_0|act|---...---|conv_i|act|-+-...---|conv_num_blocks|act|-+-act_out
                                !______________!               ... __________! 
           
    - residual and batch_norm are set:
    act_in---|conv_0|act|---...---|conv_i|act|-+---|bn_i|...---|conv_num_blocks|act|-+-|bn_num_blocks|---act_out
                                !______________!                       ... __________! 
    Attributes:
    act: nn.Module 
        activation funtion.
    initial_conv: nn.Conv3D
        corresponds to conv_0.
    convs: nn.ModuleList
        list of convolutions layers.
    bns: nn.ModuleList
        list of batch norm layers.
        
    Parameters:
    num_convs: int
        number of convolution in this block
    in_channels: int
        number of input channels in this block
    out_channels: int
        number of output channels in this block
    kernel_size: {int, tuple of int}
        size of convolution kernel.
    stride:  {int, tuple of int}
        sampling step of convolution function, each convolution is computed each stride pixels.
    dilatation:  {int, tuple of int}
        implement atrous convolution, a trick to have bigger support and sparse kernels.
    groups_in: int
        depth wise convolution of conv 0
    groups_out: int
        depth wise convolution of conv 1,...,num_convs-1
    depth_initialization: int
        use fixup initialization starting at depth depth_initialization.
    bias: bool
        use bias in conv.
    residual: bool
        use residual scheme.
    batch_norm: bool
        use batch norm residual scheme.
    padding: string 
        {'same','valid'} padding schemes.
    activation: nn.Module
        activation to be used.
    """

    def __init__(self, num_convs, in_channels, out_channels, 
                 kernel_size=3, stride=1, dilation=1, groups_in=1, groups_out=1, depth_initialization=0, 
                 bias=True, residual=True, batch_norm=True, conv_mode='3d',
                 padding='same', activation=nn.LeakyReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        
        # activation   
        self.act = activation
        
        # convolutions
        # conv input_channels -> output_channels
        self.initial_conv = Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                                   dilation=dilation, groups=groups_in, bias=bias, mode=conv_mode)
        # instantiate module
        self.convs = nn.ModuleList([Conv(out_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                                           dilation=dilation, groups=groups_out, bias=bias, mode=conv_mode)] * (num_convs - 1))
        
        # depth fixup initialization arXiv:1901.09321v2
        if depth_initialization>0:
            self.initial_conv.apply(scale_weight_depth(depth_initialization))
            for i,conv in enumerate(self.convs): conv.apply(scale_weight_depth(depth_initialization + i + 1))
        
        # batch normalization
        # bn output_channels * (num_blocks-1)
        if batch_norm: 
            # instantiate module
            self.bns = nn.ModuleList([nn.BatchNorm3d(out_channels)] * (num_convs - 1))
        
        # instruction prefetch
        self.pre_forward = self.vanilla_forward
        self.pre_forward = self.residual_forward if residual else self.pre_forward
        self.pre_forward = self.batch_norm_forward if batch_norm else self.pre_forward
        self.pre_forward = self.residual_batch_norm_forward if residual and batch_norm else self.pre_forward
        
    def batch_norm_forward(self,x):
        for bn,conv in zip(self.bns, self.convs) : x = bn(self.act(conv(x)))
        return x
        
    def residual_batch_norm_forward(self,x):
        for bn,conv in zip(self.bns, self.convs) : x = bn(self.act(conv(x)) + x)
        return x

    def residual_forward(self, x):
        for conv in self.convs: x = self.act(conv(x)) + x
        return x

    def vanilla_forward(self, x):
        for conv in self.convs: x = self.act(conv(x))
        return x

    def forward(self, x):
        x = self.act(self.initial_conv(x))
        return self.pre_forward(x)
