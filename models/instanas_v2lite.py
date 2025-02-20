import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import torch.nn.init as torchinit
import math
from models import base
from utils import *


class InstaNas(nn.Module):
    def forward(self, x, policy, drop_path_prob=0, is_ssd=True):
        # breakpoint()
        flops = torch.zeros(x.size(0), requires_grad=False).cuda(non_blocking=True).float()
        delta_flop, x = self.conv1.get_bops(x)
        flops += delta_flop
        
        t = 0
        if is_ssd is True:
            outputs = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            for idx in range(num_blocks):
                action = policy[:, t, :].contiguous()
                action_mask     = [action[:, i].contiguous().float().view(-1, 1, 1, 1) for i in range(action.size(1))]
                delta_flop = []
                feature_map = []
                feature_map_ssd = []
                for i, j in zip(range(policy.shape[2]), range(action.size(1))):
                    # breakpoint()
                    _delta_flop, _x = self.layers[t][i].get_bops(x)
                    feature_map.append( _x * action_mask[j])
                    if is_ssd is True and self.conv2 is not None:
                        exp_out = self.layers[t][i].out
                        feature_map_ssd.append(exp_out * action_mask[j])
                    delta_flop.append(_delta_flop)
                delta_flop = [action[:,i] * delta_flop[i] for i in range(action.size(1))]
                flops += sum(delta_flop)
                x = sum(feature_map)
                exp_out = sum(feature_map_ssd)
                if is_ssd is True:
                    if t == 12 and self.conv2 is not None:
                        outputs.append(exp_out)
                    if t == 10 and self.conv2 is None:
                        outputs.append(x)
                # breakpoint()
                t += 1
        if is_ssd is True and self.conv2 is None:
            outputs.append(x)
            return outputs, flops
        if self.conv2 is not None:
            delta_flop, x = self.conv2.get_bops(x)
            flops += delta_flop
            if is_ssd is True:
                outputs.append(x)
                return outputs, flops
            x = F.avg_pool2d(x, 7)
        elif self.num_classes == 200:
            x = F.avg_pool2d(x, 4)
        else:
            x = F.avg_pool2d(x, 7)
            #x = F.avg_pool2d(x, 10) # 300x300
        x = x.view(x.size(0), -1)
        delta_flop, x = self.linear.get_bops(x)
        flops += delta_flop
        # breakpoint()
        return x, flops #, None #logits_aux #, flops
    def forward_test_ops(self, x, drop_path_prob=0, is_ssd=True):
       # print("forward test ops")
        flops = torch.zeros(x.size(0), requires_grad=False).cuda(non_blocking=True).float()
        delta_flop, x = self.conv1.get_bops(x)
        flops += delta_flop
        t = 0
        if is_ssd is True:
            outputs = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            for idx in range(num_blocks):
                x = self.layers[t][0](x)
                if is_ssd is True:
                    if t == 10:
                        outputs.append(x)
                t += 1
        if is_ssd is True:
            outputs.append(x)
            return outputs, flops
        if self.conv2 is not None:
            delta_flop, x = self.conv2.get_bops(x)
            flops += delta_flop
            x = F.avg_pool2d(x, 7)
        elif self.num_classes == 200:
            x = F.avg_pool2d(x, 4)
        else:
            x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        delta_flop, x = self.linear.get_bops(x)
        flops += delta_flop

    def forward_test(self, x, policy, drop_path_prob=0, is_ssd=True):
        flops = torch.zeros(x.size(0), requires_grad=False).cuda(non_blocking=True).float()
        delta_flop, x = self.conv1.get_bops(x)
        flops += delta_flop
        t = 0
        if is_ssd is True:
            outputs = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            for idx in range(num_blocks):
                action = policy[:, t, :].contiguous()
                action_mask = [action[:, i].contiguous().float().view(-1, 1, 1, 1) for i in range(action.size(1))]
                delta_flop = []
                feature_map = [self.layers[t][i](x) * action_mask[i] for i in range(action.size(1))]
                for i in range(action.size(1)):
                    _delta_flop, _x = self.layers[t][i].get_bops(x)
                    _delta_flop *= action[:,i].float()
                    delta_flop.append(_delta_flop)
                flops += sum(delta_flop)
                x = sum(feature_map)
                if is_ssd is True:
                    if t == 10 or t==12:
                        outputs.append(x)
                
                t += 1
        if is_ssd is True:
            outputs.append(x)
            return outputs, flops
        if self.conv2 is not None:
            delta_flop, x = self.conv2.get_bops(x)
            flops += delta_flop
            x = F.avg_pool2d(x, 7)
        elif self.num_classes == 200:
            x = F.avg_pool2d(x, 4)
        else:
            x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        delta_flop, x = self.linear.get_bops(x)
        flops += delta_flop
        return x , flops
    def forward_single(self, x, policy):
        # Stem
        x = F.relu(self.bn1(self.conv1(x)))
        t = 0
        for _, _, num_blocks, _ in self.cfg:
            for idx in range(num_blocks):
                feature = [] # torch.zeros_like()
                action_mask = [policy[t, i].data.numpy() for i in range(policy.size(1))]
                if sum(action_mask) == 0:
                    # Quick skip
                    if idx == 0:
                        x = self.layers[t][0](x)
                else:
                    for i, mask in enumerate(action_mask):
                        if mask == 1:
                            feature.append(self.layers[t][i](x))
                    x = sum(feature)
                t += 1
        if self.conv2 is not None:
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.avg_pool2d(x, 7)
        else:
            x = F.avg_pool2d(x,2)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                block = [self._make_action(in_planes, out_planes, expansion, stride, i) for i in range(self.num_of_actions)]
                block = nn.ModuleList(block)
                layers.append(block)
                in_planes = out_planes
        self.num_of_layers = len(layers)
        print(" [*] Total num of layers: ", self.num_of_layers)
        return nn.Sequential(*layers)
    def _make_layers_QT(self, in_planes, version, full_pretrain, ActQ='PACT', abit=2):
        print('activation quant scheme is ' + ActQ + '!')
        layers = []
        # breakpoint()
        idx = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            for n in range(num_blocks):
                if n == 0:
                    block = [self._make_action_QT(
                                in_planes, out_planes, expansion, stride, i, version, full_pretrain, 
                                layer_name='layers_'+str(idx)+'-'+str(i), ActQ=ActQ, 
                                wbit=self.action_list[i], abit=abit) 
                            for i in range(self.num_of_actions)]
                    block = nn.ModuleList(block)
                    layers.append(block)
                else:
                    block = [self._make_action_QT(
                                in_planes, out_planes, expansion, 1, i, version, full_pretrain, 
                                layer_name='layers_'+str(idx)+'-'+str(i), ActQ=ActQ, 
                                wbit=self.action_list[i], abit=abit) 
                            for i in range(self.num_of_actions)]
                    block = nn.ModuleList(block)
                    layers.append(block)
                    
                in_planes = out_planes
                idx += 1
        self.num_of_layers = len(layers)
        print(" [*] Total num of layers: ", self.num_of_layers)
        return nn.Sequential(*layers)
    
    def _make_action_QT(self, inp, oup, exp, stride, id, version, full_pretrain, 
                        layer_name=None, ActQ="PACT", wbit=8, abit=2):
        action_bit = abit
        print("ActQ is " + ActQ + " in make_action_QT!!!")
        if version == 'V2':
            action = base.QuantInvertedResBlock(inp, oup, stride, kernel=3, expansion=exp, 
                                                wbit=wbit, abit=action_bit, 
                                                full_pretrain=full_pretrain, 
                                                layer_name=layer_name, ActQ=ActQ)

        else:
            action = base.QuantMBBlock(inp, oup, stride, kernel=3, expansion=1, 
                                       wbit=wbit, abit=action_bit, 
                                       full_pretrain=full_pretrain, layer_name=layer_name, 
                                       ActQ=ActQ)
            
        return action
    
    def _make_action(self, inp, oup, _, stride, id):
        if id == 0:  # InvertedResBlock_3x3_6F
            action = base.InvertedResBlock(inp, oup, stride, kernel=3, expansion=6)
        elif id == 1:  # InvertedResBlock_3x3_3F
            action = base.InvertedResBlock(inp, oup, stride, kernel=3, expansion=3)
        elif id == 2:  # InvertedResBlock_5x5_6F
            action = base.InvertedResBlock(inp, oup, stride, kernel=5, expansion=6)
        elif id == 3:  # InvertedResBlock_5x5_3F
            action = base.InvertedResBlock(inp, oup, stride, kernel=5, expansion=3)
        elif id == 4:  # BasicBlock
            action = base.BasicBlock(inp, oup, stride)
        #elif id == 5:  # Turnover
        #    action = nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False)
        else:
            raise ValueError(" [*] No such action index")
        return action

    def freeze_PACT_param(self):
        for name, pg in self.named_parameters():
            if 'alpha' in name:
                pg.requires_grad = False
               # print(pg)
    def train_PACT_param(self):
        for name, pg in self.named_parameters():
            if 'alpha' in name:
                pg.requires_grad = True
               # print(pg)
    def _profile(self, input_size):
        # Synthetic Input
        x = torch.autograd.Variable(torch.ones(1, 3, input_size, input_size)).cpu()
        print("input x ")
        print(x.shape)
        t = 0
        self.baseline = Variable(torch.tensor(0.), requires_grad=False) # Count mobilenetv2 syn latency
        self.baseline_max = Variable(torch.tensor(0.), requires_grad=False)
        self.baseline_min = Variable(torch.tensor(0.), requires_grad=False)
        delta_flop = 0 
        delta_flop, x = self.conv1.get_bops(x)
        self.baseline += delta_flop
        self.baseline_max += delta_flop
        self.baseline_min += delta_flop
        for _, _, num_blocks, _ in self.cfg:
            for b_idx in range(num_blocks):
                feature_map_raw = [self.layers[t][k](x) for k in [0,1,2,3]]
                for a in [0,1,2,3]:
                    _delta_flop, _x = self.layers[t][a].get_bops(x)
                    if a == 0:
                        self.baseline_min += _delta_flop
                    elif a == 1:
                        self.baseline += _delta_flop
                    elif a ==3:
                        self.baseline_max += _delta_flop
                x = sum(feature_map_raw)              
                t += 1
        if self.conv2 is not None:
            delta_flop, x = self.conv2.get_bops(x)
            self.baseline += delta_flop
            self.baseline_min += delta_flop
            self.baseline_max += delta_flop
            x = F.avg_pool2d(x, 7)
        elif input_size == 224:
            x = F.avg_pool2d(x, 7)
        elif input_size == 300:
            x = F.avg_pool2d(x, 10)
        else:
            x = F.avg_pool2d(x, 2)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        delta_flop, x = self.linear.get_bops(x)
        self.baseline += delta_flop
        self.baseline_min += delta_flop
        self.baseline_max += delta_flop

    def _get_latency(self, op, x):
        total_iter = 5
        latency_list = []
        for _ in range(total_iter):
            start = time.time()
            _ = op(x)
            latency_list.append(time.time()-start)
        return np.mean(latency_list[1:])


class QT_MobileNet_V2(InstaNas):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, config=None, num_classes=1000, full_pretrain=True, ActQ='PACT', abit=2, action_list=None):
        super(QT_MobileNet_V2, self).__init__()
        self.action_list = action_list
        self.num_of_actions = len(action_list)
        self.num_of_blocks = sum([num_blocks for expansion, out_planes, num_blocks, stride in self.cfg])
        self.wbit = 8
        self.abit = 8
        self.conv1 = base.QconvBlock(in_planes=3, out_planes=32, kernel=3, stride=2, padding=1, 
                                     wbit=self.wbit, abit=self.abit, weight_only=True, 
                                     same_padding=False, full_pretrain=full_pretrain, ActQ=ActQ)
        self.layers = self._make_layers_QT(in_planes=32, version='V2', 
                                           full_pretrain=full_pretrain, ActQ=ActQ, abit=abit)
        self.conv2 = base.QconvBlock(320, 1280, kernel=1, stride=1, padding=0, 
                                     wbit=self.wbit, abit=self.abit, weight_only=False, 
                                     full_pretrain=full_pretrain, ActQ=ActQ)
        # self.linear = base.QLinearBlock(1280, num_classes, wbit=self.wbit, abit=self.abit, weight_only=True, full_pretrain=full_pretrain)
        # self._profile(input_size=32)
