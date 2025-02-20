import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
#from utils.config import FLAGS
import math
from torch.nn.modules.utils import _pair

class ScaleSigner(Function):
    """take a real value x, output sign(x)*E(|x|)"""
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def scale_sign(input):
    return ScaleSigner.apply(input)

class PactQuantizeSTE(Function):
        
    @staticmethod
    def forward(ctx, input, eps, alpha):
        where_input_nonclipped = (input >= 0) * (input < alpha)
        where_input_gtalpha = (input >= alpha)
        ctx.save_for_backward(where_input_nonclipped, where_input_gtalpha)
        return ((input / (eps)).floor() * eps).clamp(0., alpha.data[0]-eps.data[0])

    @staticmethod
    def backward(ctx, grad_output):
        # see Hubara et al., Section 2.3
        where_input_nonclipped, where_input_gtalpha = ctx.saved_variables
        zero = torch.zeros(1).to(where_input_nonclipped.device)
        grad_input = torch.where(where_input_nonclipped, grad_output, zero)
        grad_alpha = torch.where(where_input_gtalpha, grad_output, zero).sum().expand(1)
        return grad_input, None, grad_alpha


class DoReFaQuantizeSTE(Function):
    @staticmethod
    def forward(ctx, input, nbit):
        scale = 2 ** nbit - 1
        return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def quantize(input, nbit):
    return DoReFaQuantizeSTE.apply(input, nbit)


def dorefa_w(w, nbit_w):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        w = 2 * quantize(w, nbit_w) - 1

    return w

def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
    scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)
    is_scalar = scalar_min and scalar_max

    if scalar_max and not scalar_min:
        sat_max = sat_max.to(sat_min.device)
    elif scalar_min and not scalar_max:
        sat_min = sat_min.to(sat_max.device)

    if any(sat_min > sat_max):
        raise ValueError('saturation_min must be smaller than saturation_max')

    n = 2 ** num_bits - 1

    # Make sure 0 is in the range
    sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
    sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

    diff = sat_max - sat_min
    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    diff[diff == 0] = n

    scale = n / diff
    zero_point = scale * sat_min
    if integral_zero_point:
        zero_point = zero_point.round()
    if signed:
        zero_point += 2 ** (num_bits - 1)
    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point
def dorefa_a(input, nbit_a):
    return quantize(torch.clamp(0.1 * input, 0, 1), nbit_a)
def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)

def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)
def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale
class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None

class LearnedClippedLinearQuantization(nn.Module):
    def __init__(self, a_bits, init_act_clip_val=8.0, dequantize=True, inplace=False, full_pretrain=False):
        super(LearnedClippedLinearQuantization, self).__init__()
        self.a_bits = a_bits
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]))
        self.dequantize = dequantize
        self.inplace = inplace
        self.full_pretrain = full_pretrain

    def forward(self, input):
        if self.full_pretrain:
            self.a_bits = 32
        # Clip between 0 to the learned clip_val
        input = F.relu(input, self.inplace)
        # Using the 'where' operation as follows gives us the correct gradient with respect to clip_val
        input = torch.where(input < self.clip_val, input, self.clip_val) #if condition, input else self.clip_val
        with torch.no_grad():
            scale, zero_point = asymmetric_linear_quantization_params(num_bits=self.a_bits, saturation_min=0, saturation_max=self.clip_val, signed=False)
        input = LinearQuantizeSTE.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return input
    def _get_new_optimizer_params_groups(self):
        clip_val_group = {'params': [param for name, param in self.model.named_parameters() if 'clip_val' in name]}
        return [clip_val_group]
    

class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit, full_pretrain=False):
    super(activation_quantize_fn, self).__init__()
    name_a_dict = {'dorefa': dorefa_a}
    self.quan_a = name_a_dict['dorefa']
    self.full_pretrain = full_pretrain
    self.a_bit = a_bit

  def forward(self, x):
    if self.full_pretrain is True:
        activation_q = x
    else:
        activation_q = self.quan_a(x, self.a_bit)
    return activation_q

class Conv2d_wQ(nn.Conv2d):
    """docstring for QuanConv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, quan_name_w='dorefa', nbit_w=32, 
                 padding=0, dilation=1, groups=1,
                 bias=False, full_pretrain=False):
        super(Conv2d_wQ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias=False)
        if full_pretrain is True:
            self.nbit_w = 32
        else:
            self.nbit_w = nbit_w
            
        name_w_dict = {'dorefa': dorefa_w}
        
        self.quan_w = name_w_dict[quan_name_w]
        

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            w = self.weight
        output = F.conv2d(input, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output


class Conv2d_Q(nn.Conv2d):
    """docstring for QuanConv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, quan_name_w='dorefa', nbit_w=32, nbit_a=32,
                 padding=0, dilation=1, groups=1,
                 bias=False, full_pretrain=False):
        super(Conv2d_Q, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias=False)
        if full_pretrain is True:
            self.nbit_w = 32
            self.nbit_a = 32
        else:
            self.nbit_w = nbit_w
            self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w}
        name_a_dict = {'dorefa': dorefa_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_w]

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            w = self.weight
        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a)
        else:
            x = input
        output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output
def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

class PactLayer(nn.Module):
    def __init__(self, a_bit=4, clip_val=8.0, full_pretrain=False):
        super(PactLayer, self).__init__()
        self.a_bit = a_bit
        self.p_alpha = clip_val
        self.full_pretrain = full_pretrain
        
    def forward(self, x):
        if self.full_pretrain is True:
            out = x
        else:
            eps = self.p_alpha / (2.0**(self.a_bit)-1)
            out = PactQuantizeSTE.apply(x, eps, self.p_alpha + eps)
        return out


class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quan_name_w='dorefa', nbit_w=8, nbit_a=8, clip_val=8.0, full_pretrain=False):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        if full_pretrain is True:
            self.nbit_w = 32
            self.nbit_a = 32
        name_w_dict = {'dorefa': dorefa_w }
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = nn.Sequential(PactLayer(self.nbit_a, clip_val, full_pretrain))


    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            w = self.weight    
        if self.nbit_a < 32:
            x = self.quan_a(input)
        else:
            x= input
        output = F.linear(x, w, self.bias)

        return output

class Linear_newPactQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quan_name_w='dorefa', quan_name_a='pact', nbit_w=8, nbit_a=8, full_pretrain=False):
        super(Linear_newPactQ, self).__init__(in_features, out_features, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        if full_pretrain is True:
            self.nbit_w = 32
            self.nbit_a = 32
        name_w_dict = {'dorefa': dorefa_w }
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = nn.Sequential(LearnedClippedLinearQuantization(self.nbit_a))

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            w = self.weight
        if self.nbit_a < 32:
            x = self.quan_a(input)
        else:
            x = input
        output = F.linear(x, w, self.bias)
        return output

class Linear_PactQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quan_name_w='dorefa', quan_name_a='pact', nbit_w=8, nbit_a=8, full_pretrain=False):
        super(Linear_PactQ, self).__init__(in_features, out_features, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        if full_pretrain is True:
            self.nbit_w = 32
            self.nbit_a = 32
        name_w_dict = {'dorefa': dorefa_w }
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = nn.Sequential(PactLayer(self.nbit_a))

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            w = self.weight
        if self.nbit_a < 32:
            x = self.quan_a(input)
        else:
            x = input
        output = F.linear(x, w, self.bias)
        return output


class aq_k(Function):
    """
        This is the quantization module.
        The input and output should be all on the interval [0, 1].
        bit is only defined on positive integer values.
    """
    @staticmethod
    def forward(ctx, input, bit, scheme='original'):
        assert bit > 0
        # assert torch.all(input >= 0) and torch.all(input <= 1)
        if scheme == 'original': # original dorefa
            a = (1 << bit) - 1
            res = torch.round(a * input)
            res.div_(a)
        elif scheme == 'modified':
            a = 1 << bit
            res = torch.floor(a * input)
            res.clamp_(max=a - 1).div_(a)
        else:
            raise NotImplementedError
        # assert torch.all(res >= 0) and torch.all(res <= 1)
        return res
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class q_k(Function):
    """
        This is the quantization module.
        The input and output should be all on the interval [0, 1].
        bit is only defined on positive integer values.
    """
    @staticmethod
    def forward(ctx, input, bit, scheme='original'):
        assert bit > 0
        assert torch.all(input >= 0) and torch.all(input <= 1)
        if scheme == 'original':
            a = (1 << bit) - 1
            res = torch.round(a * input)
            res.div_(a)
        elif scheme == 'modified':
            a = 1 << bit
            res = torch.floor(a * input)
            res.clamp_(max=a - 1).div_(a)
        else:
            raise NotImplementedError
        assert torch.all(res >= 0) and torch.all(res <= 1)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def round_width(width, wm=None,
                #width_divisor=getattr(FLAGS, 'width_divisor', 1),
                width_divisor=1,
                #min_width=getattr(FLAGS, 'min_width', 1)):
                min_width=1):
    if not wm:
        return width
    width *= wm
    if min_width is None:
        min_width = width_divisor
    new_width = max(min_width, int(width + width_divisor / 2) // width_divisor * width_divisor)
    if new_width < 0.9 * width:
        new_width += width_divisor
    return int(new_width)

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True,
                 same_padding=False,
                 bitw_min=None, bita_min=None,
                 pact_fp=False,
                 double_side=False,
                 weight_only=False, full_pretrain=False):
        super(QConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding if not same_padding else 0,
            dilation=dilation,
            groups=groups, bias=bias)
        self.same_padding = same_padding
        self.bitw_min = bitw_min
        self.bita_min = bita_min
        self.pact_fp = pact_fp
        self.double_side = double_side
        self.weight_only = weight_only
        self.quant = q_k.apply
        self.full_pretrain = full_pretrain
        self.alpha = nn.Parameter(torch.tensor(8.0))

    def forward(self, input):
        #print("full pretrain in Qconv2d: %d" %(self.full_pretrain))
        if self.same_padding:
            ih, iw = input.size()[-2:]
            kh, kw = self.weight.size()[-2:]
            sh, sw = self.stride
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
            pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                input = nn.functional.pad(input, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        bitw = self.bitw_min
        bita = self.bita_min
        if self.full_pretrain is True:
            bitw = 32
            bita = 32
        weight_quant_scheme = 'original'
        act_quant_scheme = 'original'

        if bitw == 0:
            return nn.Identity()(input)

        if bitw < 32:
            weight = torch.tanh(self.weight) / torch.max(torch.abs(torch.tanh(self.weight)))
            weight.add_(1.0)
            weight.div_(2.0)
            weight = self.quant(weight, bitw, weight_quant_scheme)
            weight.mul_(2.0)
            weight.sub_(1.0)
        else:
            weight = self.weight * 1.0

        if (bita < 32 and not self.weight_only) or self.pact_fp:
            alpha = torch.abs(self.alpha)
            if self.double_side:
                input_val = torch.where(input > -alpha, input, -alpha)
            else:
                input_val = torch.relu(input)
            input_val = torch.where(input_val < alpha, input_val, alpha)
            if bita < 32 and not self.weight_only:
                input_val.div_(alpha)
                if self.double_side:
                    input_val.add_(1.0)
                    input_val.div_(2.0)
                input_val = self.quant(input_val, bita, act_quant_scheme)
                if self.double_side:
                    input_val.mul_(2.0)
                    input_val.sub_(1.0)
                input_val.mul_(alpha)
        else:
            input_val = input
        #print(self.bias)
        #print(self.stride)
        y = nn.functional.conv2d(
            input_val, weight, bias=self.bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups)
        return y



class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bitw_min=None, bita_min=None, pact_fp=False, weight_only=False, full_pretrain=False):
        super(QLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.bitw_min = bitw_min
        self.bita_min = bita_min
        self.pact_fp = pact_fp
        self.full_pretrain = full_pretrain
        self.weight_only = weight_only
        self.quant = q_k.apply
        self.alpha = nn.Parameter(torch.tensor(10.0))

    def forward(self, input):
        bitw = self.bitw_min
        bita = self.bita_min
        if self.full_pretrain is True:
            bitw = 32
            bita = 32
        weight_quant_scheme = 'original'
        act_quant_scheme = 'original'

        if bitw == 0:
            return nn.Identity()(input)

        if bitw < 32:
            weight = torch.tanh(self.weight) / torch.max(torch.abs(torch.tanh(self.weight)))
            weight.add_(1.0)
            weight.div_(2.0)
            weight = self.quant(weight, bitw, weight_quant_scheme)
            weight.mul_(2.0)
            weight.sub_(1.0)
        else:    
            weight = self.weight * 1.0
        bias = self.bias

        if (bita < 32 and not self.weight_only) or self.pact_fp:
            alpha = torch.abs(self.alpha)
            input_val = torch.relu(input)
            input_val = torch.where(input_val < alpha, input_val, alpha)
            if bita < 32 and not self.weight_only:
                input_val.div_(alpha)
                input_val = self.quant(input_val, bita, act_quant_scheme)
                input_val.mul_(alpha)
        else:
            input_val = input
        return nn.functional.linear(input_val, weight, bias)

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

class _ActQ(nn.Module):
    def __init__(self, in_features, **kwargs_q):
        super(_ActQ, self).__init__()
        # self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        # if self.nbits < 0:
        #     self.register_parameter('alpha', None)
        #     self.register_parameter('zero_point', None)
        #     return
        # self.signed = kwargs_q['signed']
        # self.q_mode = kwargs_q['mode']
        self.scale = nn.Parameter(torch.Tensor(1))
        self.zero_point = nn.Parameter(torch.Tensor([0]))
        # if self.q_mode == Qmodes.kernel_wise:
        #     self.alpha = Parameter(torch.Tensor(in_features))
        #     self.zero_point = Parameter(torch.Tensor(in_features))
        #     torch.nn.init.zeros_(self.zero_point)
        # self.zero_point = Parameter(torch.Tensor([0]))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

class ActLSQ(_ActQ):
    def __init__(self, in_features, nbits_a=4, mode='layer_wise', **kwargs):
        super(ActLSQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        # print(self.alpha.shape, self.zero_point.shape)
        self.nbits = nbits_a
    def forward(self, x):
        if self.scale is None:
            return x

        if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            # print(self.signed)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            self.scale.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.zero_point.data.copy_(self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.scale.data * Qn))
            self.init_state.fill_(1)
        
        # print(self.signed)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        scale = grad_scale(self.scale, g)
        zero_point = grad_scale(zero_point, g)
        # x = round_pass((x / scale).clamp(Qn, Qp)) * scale
        if len(x.shape)==2:
            scale = scale.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape)==3:
            if x.shape[0] == scale.shape[0]:
                scale = scale.unsqueeze(1).unsqueeze(2)
                zero_point = zero_point.unsqueeze(1).unsqueeze(2)
            elif x.shape[1] == scale.shape[0]:
                scale = scale.unsqueeze(0).unsqueeze(2)
                zero_point = zero_point.unsqueeze(0).unsqueeze(2)
            elif x.shape[2] == scale.shape[0]:
                scale = scale.unsqueeze(0).unsqueeze(0)
                zero_point = zero_point.unsqueeze(0).unsqueeze(0)
        elif len(x.shape)==4:
            # A, B, C, D = x.shape
            if x.shape[0] == scale.shape[0]:
                scale = scale.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                zero_point = zero_point.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            elif x.shape[1] == scale.shape[0]:
                scale = scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            elif x.shape[2] == scale.shape[0]:
                scale = scale.unsqueeze(0).unsqueeze(0).unsqueeze(3)
                zero_point = zero_point.unsqueeze(0).unsqueeze(0).unsqueeze(3)
            elif x.shape[3] == scale.shape[0]:
                scale = scale.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                zero_point = zero_point.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # print(scale.shape, zero_point.shape)
        # print(x.shape)
        x = round_pass((x / scale + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * scale
        # x = x.clamp(Qn, Qp)
        # q_x = round_pass(x)
        # x_q = q_x * scale

        # Method2:
        # x_q = FunLSQ.apply(x, self.scale, g, Qn, Qp)
        return x

class DRFLSQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True,
                 same_padding=False,
                 bitw_min=None, bita_min=None,
                 pact_fp=False,
                 double_side=False,
                 weight_only=False, full_pretrain=False,
                 input_quant=True, layer_name = None):
        super(DRFLSQConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding if not same_padding else 0,
            dilation=dilation,
            groups=groups, bias=bias)
        self.same_padding = same_padding
        self.bitw_min = bitw_min
        self.bita_min = bita_min
        self.pact_fp = pact_fp
        self.double_side = double_side
        self.weight_only = weight_only
        self.input_quant = input_quant
        self.quant = q_k.apply
        self.full_pretrain = full_pretrain
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.layer_name = layer_name
        
        self.scale = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        
        self.act = ActLSQ(in_features=in_channels, nbits_a=bita_min)

    def forward(self, input):
        #print("full pretrain in Qconv2d: %d" %(self.full_pretrain))
        bitw = self.bitw_min
        bita = self.bita_min
        if self.full_pretrain is True:
            bitw = 32
            bita = 32
        weight_quant_scheme = 'original'
        act_quant_scheme = 'original'

        if bitw == 0:
            return nn.Identity()(input)

        # weight quant --> DoReFa
        if bitw < 32:
            #print("qted")
            weight = torch.tanh(self.weight) / torch.max(torch.abs(torch.tanh(self.weight)))
            weight.add_(1.0)
            weight.div_(2.0)
            weight = self.quant(weight, bitw, weight_quant_scheme)
            weight.mul_(2.0)
            weight.sub_(1.0)
            # breakpoint()
        else:
            weight = self.weight * 1.0
        # breakpoint()
        
        # activation quant --> LSQ+
        if (bita < 32 and self.input_quant):
            input_val = self.act(input)
        else:
            input_val = input
        #print(self.bias)
        #print(self.stride)
        y = nn.functional.conv2d(
            input_val, weight, bias=self.bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups)
        return y


class DRFQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True,
                 same_padding=False,
                 bitw_min=None, bita_min=None,
                 pact_fp=False,
                 double_side=False,
                 weight_only=False, full_pretrain=False,
                 input_quant=True, layer_name = None):
        super(DRFQConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding if not same_padding else 0,
            dilation=dilation,
            groups=groups, bias=bias)
        self.same_padding = same_padding
        self.bitw_min = bitw_min
        self.bita_min = bita_min
        self.pact_fp = pact_fp
        self.double_side = double_side
        self.weight_only = weight_only
        self.input_quant = input_quant
        self.wquant = q_k.apply
        self.aquant = aq_k.apply
        self.full_pretrain = full_pretrain
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.layer_name = layer_name

    def forward(self, input):
        #print("full pretrain in Qconv2d: %d" %(self.full_pretrain))
        # padding 300 --> 302됨
        if self.same_padding:
            ih, iw = input.size()[-2:]
            kh, kw = self.weight.size()[-2:]
            sh, sw = self.stride
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
            pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                input = nn.functional.pad(input, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        bitw = self.bitw_min
        bita = self.bita_min
        if self.full_pretrain is True:
            bitw = 32
            bita = 32
        weight_quant_scheme = 'original'
        act_quant_scheme = 'original'

        if bitw == 0:
            return nn.Identity()(input)

        if bitw < 32:
            #print("qted")
            weight = torch.tanh(self.weight) / torch.max(torch.abs(torch.tanh(self.weight)))
            weight.add_(1.0)
            weight.div_(2.0)
            weight = self.wquant(weight, bitw, weight_quant_scheme)
            weight.mul_(2.0)
            weight.sub_(1.0)
            # breakpoint()
        else:
            weight = self.weight * 1.0
        print(self.layer_name, weight.max().item())
        
        if (bita < 32 and not self.weight_only and self.input_quant):
            input_val = self.aquant(input, bita, act_quant_scheme)
        else:
            input_val = input
        print(self.layer_name, input_val.max().item())
        # breakpoint()
        y = nn.functional.conv2d(
            input_val, weight, bias=self.bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups)
        return y


class LSQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True,
                 same_padding=False,
                 bitw_min=None, bita_min=None,
                 double_side=False,
                 weight_only=False, full_pretrain=False,
                 input_quant=True, layer_name = None, unsigned=True, batch_init=20):
        super(LSQConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups, bias=bias)
        self.same_padding = same_padding
        self.bitw_min = bitw_min
        self.bita_min = bita_min
        self.double_side = double_side
        self.weight_only = weight_only
        self.input_quant = input_quant
        self.full_pretrain = full_pretrain
        self.layer_name = layer_name
        self.unsigned = unsigned
        self.batch_init = batch_init
        
        self.scale = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        
        self.act = ActLSQ(in_features=in_channels, nbits_a=bita_min)
        
        # self.activation_quantizer = LSQPlusActivationQuantizer(a_bits=bita_min, all_positive=unsigned,batch_init = batch_init)
        # self.weight_quantizer = LSQPlusWeightQuantizer(w_bits=bitw_min, all_positive=unsigned, per_channel=True,batch_init = batch_init)
        
    def forward(self, input):
        #print("full pretrain in Qconv2d: %d" %(self.full_pretrain))
        # padding 300 --> 302됨
        bitw = self.bitw_min
        bita = self.bita_min
        if self.full_pretrain:
            bitw = 32
            bita = 32
        # breakpoint()
        if bitw == 0:
            return nn.Identity()(input)
        weight_quant_scheme = 'original'
        act_quant_scheme = 'original'
        
        if bitw < 32:
            # weight = self.weight_quantizer(self.weight)
            Qn = -2 ** (bitw - 1)
            Qp = 2 ** (bitw - 1) - 1
            if self.training and self.init_state == 0:
                # self.scale.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
                if Qp >0:
                    self.scale.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
                else:
                    self.scale.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp+1))
                # self.scale.data.copy_(self.weight.abs().max() * 2)
                self.init_state.fill_(1)
            
            g = 1.0 / math.sqrt(self.weight.numel() * Qp) if Qp>0 else 1.0 / math.sqrt(self.weight.numel())
            
            scale = grad_scale(self.scale, g)
            scale = scale.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # breakpoint()
            weight = round_pass((self.weight / scale).clamp(Qn, Qp)) * scale
        else:
            weight = self.weight * 1.0
        # breakpoint()
        
        if (bita < 32 and self.input_quant):
            input_val = self.act(input)
        else:
            input_val = input
        #print(self.bias)
        #print(self.stride)
        y = nn.functional.conv2d(
            input_val, weight, bias=self.bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups)
        return y