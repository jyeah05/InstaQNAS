import torch.nn as nn
import torch
import torch.nn.functional as F
from quantization.quant_functions import *
from utils import count_conv_flop
from collections import OrderedDict

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Qconv_Seperable(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, padding, wbit=8, abit=8, weight_only=False, same_padding=True, full_pretrain=False):
        super(Qconv_Seperable, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel = kernel
        self.pad = padding
        self.stride = stride
        self.wbit = wbit
        self.abit = abit
        self.conv_sep = nn.Sequential(OrderedDict([
            ('conv', QConv2d(self.in_planes, self.in_planes, kernel_size=self.kernel, stride=self.stride, padding=self.pad, groups=self.in_planes, bias=False,  bitw_min=8, bita_min=8, weight_only=weight_only, same_padding=same_padding, full_pretrain=full_pretrain)),
            ('bn', nn.BatchNorm2d(self.in_planes)),
            ('act', nn.ReLU(inplace=True)), ]))
        self.conv_point = nn.Sequential(OrderedDict([
            ('conv', QConv2d(self.in_planes, self.out_planes, kernel_size=1, stride=1, padding=0, bias=True,  bitw_min=self.wbit, bita_min=self.abit, weight_only=weight_only, same_padding=same_padding, full_pretrain=full_pretrain)),
            ]))
    def forward(self, x):
        x = self.conv_sep(x)
        out = self.conv_point(x)
        return out
    def get_flops(self, x):
        flops_0 = count_conv_flop(self.conv_sep.conv, x)
        x = self.conv_sep(x)
        flops_1 = count_conv_flop(self.conv_point.conv, x)
        out = self.conv_point(x)
        return flops_0 + flops_1 , out
    def get_bops(self, x):
        flops_0 = count_conv_flop(self.conv_sep.conv, x)
        flops_0 = flops_0 * 8 * 8
        x = self.conv_sep(x)
        flops_1 = count_conv_flop(self.conv_point.conv, x)
        flops_1 = flops_1 * self.wbit * self.abit
        out = self.conv_point(x)
        return flops_0 + flops_1, out


class Qconv_MERE(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, padding, wbit=8, abit=8, weight_only=False, same_padding=True, full_pretrain=False):
        super(Qconv_MERE, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel = kernel
        self.pad = padding
        self.stride = stride
        self.wbit = wbit
        self.abit = abit
        self.conv_q = nn.Sequential(OrderedDict([
            ('conv', QConv2d(self.in_planes, self.out_planes, kernel_size=self.kernel, stride=self.stride, padding=self.pad, bias=True,  bitw_min=self.wbit, bita_min=self.abit, weight_only=weight_only, same_padding=same_padding, full_pretrain=full_pretrain)),
            ]))
    def forward(self, x):
        out = self.conv_q(x)
        return out
    def get_flops(self, x):
        flops = count_conv_flop(self.conv_q.conv, x)
        out = self.conv_q(x)
        return flops, out
    def get_bops(self, x):
        flops = count_conv_flop(self.conv_q.conv, x)
        flops = flops * self.wbit * self.abit
        out = self.conv_q(x)
        return flops, out
class QconvBlock_BIAS(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, padding, wbit=8, abit=8, weight_only=False, same_padding=True, full_pretrain=False):
        super(QconvBlock_BIAS, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel = kernel
        self.pad = padding
        self.stride = stride
        self.wbit = wbit
        self.abit = abit
        self.conv_q = nn.Sequential(OrderedDict([
            ('conv', QConv2d(self.in_planes, out_channels=self.out_planes, kernel_size=self.kernel, stride=self.stride, padding=self.pad, bias=True,  bitw_min=self.wbit, bita_min=self.abit, weight_only=weight_only, same_padding=same_padding, full_pretrain=full_pretrain)),
            ('act', nn.ReLU(inplace=True)),]))
    def forward(self, x):
        out = self.conv_q(x)
        return out
    def get_flops(self, x):
        flops = count_conv_flop(self.conv_q.conv, x)
        out = self.conv_q(x)
        return flops, out
    def get_bops(self, x):
        flops = count_conv_flop(self.conv_q.conv, x)
        flops = flops * self.wbit * self.abit
        out = self.conv_q(x)
        return flops, out
class QLinearBlock(nn.Module):
    def __init__(self, in_planes, out_planes, wbit=8, abit=8, weight_only=False, full_pretrain=False):
        super(QLinearBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.wbit = wbit
        self.abit = abit
        self.linear = QLinear(self.in_planes, self.out_planes, bitw_min=self.wbit, bita_min=self.abit, weight_only=weight_only, full_pretrain=full_pretrain)
    def forward(self, x):
        out = self.linear(x)
        return out
    def get_flops(self, x):
        flops = self.linear.weight.numel()
        out = self.linear(x)
        return flops, out
    def get_bops(self, x):
        flops = self.linear.weight.numel()* self.wbit * self.abit
        out = self.linear(x)
        return flops, out 
class Qconv_Seperable_MBv1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, padding, wbit=8, abit=8, weight_only=False, same_padding=True, full_pretrain=False):
        super(Qconv_Seperable_MBv1, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel = kernel
        self.pad = padding
        self.stride = stride
        self.wbit = wbit
        self.abit = abit
        self.conv_sep = nn.Sequential(OrderedDict([
            ('conv', QConv2d(self.in_planes, self.in_planes, kernel_size=self.kernel, stride=self.stride, padding=self.pad, groups=self.in_planes, bias=True,  bitw_min=8, bita_min=8, weight_only=weight_only, same_padding=same_padding, full_pretrain=full_pretrain)),
            ('act', nn.ReLU(inplace=True)), ]))
        self.conv_point = nn.Sequential(OrderedDict([
            ('conv', QConv2d(self.in_planes, self.out_planes, kernel_size=1, stride=1, padding=0, bias=True,  bitw_min=self.wbit, bita_min=self.abit, weight_only=weight_only, same_padding=same_padding, full_pretrain=full_pretrain)),
            ]))
    def forward(self, x):
        x = self.conv_sep(x)
        out = self.conv_point(x)
        return out
    def get_flops(self, x):
        flops_0 = count_conv_flop(self.conv_sep.conv, x)
        x = self.conv_sep(x)
        flops_1 = count_conv_flop(self.conv_point.conv, x)
        out = self.conv_point(x)
        return flops_0 + flops_1 , out
    def get_bops(self, x):
        flops_0 = count_conv_flop(self.conv_sep.conv, x)
        flops_0 = flops_0 * 8 * 8
        x = self.conv_sep(x)
        flops_1 = count_conv_flop(self.conv_point.conv, x)
        flops_1 = flops_1 * self.wbit * self.abit
        out = self.conv_point(x)
        return flops_0 + flops_1, out

class QconvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, padding, wbit=8, abit=8, weight_only=False, same_padding=True, full_pretrain=False):
        super(QconvBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel = kernel
        self.pad = padding
        self.stride = stride
        self.wbit = wbit
        self.abit = abit
        self.conv_q = nn.Sequential(OrderedDict([
            ('conv', QConv2d(self.in_planes, self.out_planes, self.kernel, self.stride, padding=self.pad, bias=False,  bitw_min=self.wbit, bita_min=self.abit, weight_only=weight_only, same_padding=True, full_pretrain=full_pretrain)),
            ('bn', nn.BatchNorm2d(self.out_planes)),
            ('act', nn.ReLU(inplace=True)),]))
    def forward(self, x):
        out = self.conv_q(x)
        return out
    def get_flops(self, x):
        flops = count_conv_flop(self.conv_q.conv, x)
        out = self.conv_q(x)
        return flops, out
    def get_bops(self, x):
        flops = count_conv_flop(self.conv_q.conv, x)
        flops = flops * self.wbit * self.abit
        out = self.conv_q(x)
        return flops, out
class QconvBlock_BIAS(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, padding, wbit=8, abit=8, weight_only=False, same_padding=True, full_pretrain=False):
        super(QconvBlock_BIAS, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel = kernel
        self.pad = padding
        self.stride = stride
        self.wbit = wbit
        self.abit = abit
        self.conv_q = nn.Sequential(OrderedDict([
            ('conv', QConv2d(self.in_planes, self.out_planes, self.kernel, self.stride, padding=self.pad, bias=True,  bitw_min=self.wbit, bita_min=self.abit, weight_only=weight_only, same_padding=True, full_pretrain=full_pretrain)),
            ('act', nn.ReLU(inplace=True)),]))
    def forward(self, x):
        out = self.conv_q(x)
        return out
    def get_flops(self, x):
        flops = count_conv_flop(self.conv_q.conv, x)
        out = self.conv_q(x)
        return flops, out
    def get_bops(self, x):
        flops = count_conv_flop(self.conv_q.conv, x)
        flops = flops * self.wbit * self.abit
        out = self.conv_q(x)
        return flops, out

class QuantMBBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, kernel, expansion=1, wbit=8, abit=8, full_pretrain=False):
        super(QuantMBBlock, self).__init__()
        self.identity = stride == 1 and in_planes == out_planes
        self.stride = stride
        self.multiplier = 1.0
        self.lat = 0
        self.flops = 0
        self.params = 0
        self.wbit = wbit
        self.abit = abit

        planes = int(round(expansion * in_planes * self.multiplier))
        self.conv1 = QConv2d(planes, planes, kernel_size=kernel, stride=stride, padding=kernel//2, groups=planes, bias=False, bitw_min=8, bita_min=8, full_pretrain=full_pretrain)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = QConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False,  bitw_min=self.wbit, bita_min=self.abit, full_pretrain=full_pretrain)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)



    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out
    def get_flops(self, x ):
        flop0 = count_conv_flop(self.conv1, x)
        out = self.relu1(self.bn1(self.conv1(x)))
        flop1 = count_conv_flop(self.conv2, out)
        out =  self.relu2(self.bn2(self.conv2(out)))
        flops = flop0 + flop1
        return flops, out
    def get_bops(self, x ):
        flop0 = count_conv_flop(self.conv1, x) * 8 * 8
        out = self.relu1(self.bn1(self.conv1(x)))
        flop1 = count_conv_flop(self.conv2, out)* self.wbit* self.abit ## JYC: depthwise bit is fixed as 8bit.
        out = self.relu2(self.bn2(self.conv2(out)))
        
        flops = flop0 + flop1
        return flops, out

class QuantInvertedResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, kernel, expansion=1, wbit=8, abit=8, full_pretrain=False):
        super(QuantInvertedResBlock, self).__init__()
        self.identity = stride == 1 and in_planes == out_planes
        self.stride = stride
        self.multiplier = 1.0
        self.lat = 0
        self.flops = 0
        self.params = 0
        self.wbit = wbit
        self.abit = abit

        planes = int(round(expansion * in_planes * self.multiplier))
        self.conv1 = QConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False, bitw_min=self.wbit, bita_min=self.abit, full_pretrain=full_pretrain)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=kernel, stride=stride, padding=kernel//2, groups=planes, bias=False, bitw_min=self.wbit, bita_min=self.abit, full_pretrain=full_pretrain)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False,  bitw_min=self.wbit, bita_min=self.abit, full_pretrain=full_pretrain)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                QConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False, bitw_min=self.wbit, bita_min=self.abit, full_pretrain=full_pretrain),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.identity:
            out = out + x
        return out
    def get_flops(self, x ):
        flop0 = count_conv_flop(self.conv1, x)
        out = F.relu(self.bn1(self.conv1(x)))
        flop1 = count_conv_flop(self.conv2, out)
        out = F.relu(self.bn2(self.conv2(out)))
        flop2 = count_conv_flop(self.conv3, out)
        out = self.bn3(self.conv3(out))
        if self.identity:
            out = out + x
        flops = flop0 + flop1 + flop2
        return flops, out
    def get_bops(self, x ):
        flop0 = count_conv_flop(self.conv1, x) * self.wbit * self.abit
        out = F.relu(self.bn1(self.conv1(x)))
        flop1 = count_conv_flop(self.conv2, out)* 2* 2 ## JYC: depthwise bit is fixed as 8bit.
        out = F.relu(self.bn2(self.conv2(out)))
        flop2 = count_conv_flop(self.conv3, out)* self.wbit * self.abit
        out = self.bn3(self.conv3(out))
        if self.identity:
            out = out + x
        flops = flop0 + flop1 + flop2
        return flops, out
class InvertedResBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, stride, kernel, expansion):
        super(InvertedResBlock, self).__init__()
        self.stride = stride
        self.multiplier = 1.0

        planes = int(round(expansion * in_planes * self.multiplier))
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=stride, padding=kernel//2, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, oup, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inp, oup, stride)
        self.bn1 = nn.BatchNorm2d(oup)
        self.conv2 = conv3x3(oup, oup)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU6(inplace=True)
        self.lat = 0

    def forward(self, x):
       
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class DilatedBlock(nn.Module):
    def __init__(self, inp, oup, stride, kernel):
        super(DilatedBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, kernel_size=kernel, stride=stride, padding=(kernel-1), dilation=2, groups=inp, bias=False)
        self.conv2 = nn.Conv2d(inp, oup, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU6(inplace=True)
        self.lat = 0
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        return out

class DownsampleB(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)



