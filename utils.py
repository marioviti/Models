import numpy as np
import torch.nn.functional as F

def scale_weights(scale_factor):
    def fun(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=scale_factor)
            if m.bias:
                torch.nn.init.zeros_(m.bias)
    return fun

def scale_weight_depth(depth):
    return scale_weights(1/np.sqrt(depth))

def numpy2tensor(nmpy, to_cuda=True, to_32float=True):
    tnsr = th.from_numpy(nmpy)
    if to_32float: tnsr = tnsr.float()
    if to_cuda: tnsr = tnsr.cuda()
    return tnsr

def tensor2numpy(tnsr, from_cuda=True, is_attached=True, squeeze=True):
    if from_cuda: tnsr = tnsr.cpu()
    if is_attached: tnsr = tnsr.detach()
    if squeeze: tnsr = tnsr.squeeze()
    return tnsr.numpy()

def compute_upconv_ouput_size(in_size,stride,padding,dilation,kernel_size,output_padding):
    out_size = (in_size-1) * stride - 2*padding + dilation*(kernel_size-1)+output_padding+1 
    return out_size

def compute_upconv_input_padding(in_size,out_size,stride,dilation,kernel_size,output_padding=0):
    padding = ((in_size-1) * stride +dilation*(kernel_size-1)+output_padding+1 - out_size)//2 
    return padding

def conv_padding_func(padding, input_shape, kernel_size, dilation, stride):
    if padding == 'valid':
        return lambda x: x, [0]*len(input_shape)*2
    if padding == 'same':
        output_padding = same_padding(input_shape, kernel_size, dilation, stride)
    if isinstance(padding, int):
        output_padding = [padding] * 3
    if all(isinstance(p, int) for p in padding):
        output_padding = padding
    output_padding = sum([[a] * 2 for a in output_padding], [])
    return lambda x: F.pad(x, list(reversed(output_padding)), 'constant', 0), output_padding

def coumpute_output_size(input_size, padding, kernel_size, dilatation, stride):
    i = input_size
    p = padding
    k = kernel_size
    d = dilatation
    s = stride
    return int(np.round((i + 2 * p - k - (k - 1) * (d - 1)) / s + 1))

def compute_output_shape(shape, padding, kernel_size, dilatation, stride):
    dimensions = len(shape)
    if isinstance(padding, int):
        padding = [padding] * dimensions
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * dimensions
    if isinstance(dilatation, int):
        dilatation = [dilatation] * dimensions
    if isinstance(stride, int):
        stride = [stride] * dimensions
    output_shape = [coumpute_output_size(i, p, k, d, s) for i, p, k, d, s in
                    zip(shape, padding, kernel_size, dilatation, stride)]
    return output_shape


def compute_padding(input_size, output_size, kernel_size, dilatation, stride):
    i = input_size
    o = output_size
    k = kernel_size
    d = dilatation
    s = stride
    return int(np.round(((o - 1) * s - i + k + (k - 1) * (d - 1)) / 2))


def compute_padding_shape(input_shape, output_shape, kernel_size, dilatation, stride):
    dimensions = len(input_shape)
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * dimensions
    if isinstance(dilatation, int):
        dilatation = [dilatation] * dimensions
    if isinstance(stride, int):
        stride = [stride] * dimensions
    output_padding = [compute_padding(i, o, k, d, s) for i, o, k, d, s in
                      zip(input_shape, output_shape, kernel_size, dilatation, stride)]
    return output_padding


def same_padding(input_shape, kernel_size, dilatation, stride):
    return compute_padding_shape(input_shape, input_shape, kernel_size, dilatation, stride)
