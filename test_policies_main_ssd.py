#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

## prequisite
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import numpy as np
import torch.utils.data as torchdata
## models
from models.instanas import QT_MobileNet_V1_224, QT_MobileNet_V2_224, QT_MobileNet_CAPP_224
from models.mb1_ssd import create_mobilenetv1_ssd_aug, create_mobilenetv1_ssd_predictor
from models.mb1_ssd_lite import create_mobilenetv1_ssd_lite_aug
from models.mb1_capp_ssd import create_mobilenetv1_capp_ssd_aug
from models.mb2_ssd_lite import create_mobilenetv2_ssd_lite_aug, create_mobilenetv2_ssd_predictor
from models.controller import Policy224_HW
## utils
from colorama import Fore
from importlib import import_module
import config
from dataloader import getDataloaders
from utils import save_checkpoint, get_optimizer, create_save_folder, print_model_parm_nums
from args import arg_parser, arch_resume_names
from models.multibox_loss import MultiboxLoss
from models.mb1_ssd_config import generate_mb1_ssd_config
from data_provider.voc_dataset import VOCDataset
from data_provider.data_preprocessing import TestTransform
from utils import collate_voc_batch

## visualization tools
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas import Series
from models import instanas
from models import controller
import seaborn as sns

cudnn.benchmark = True
import warnings
warnings.filterwarnings("ignore")
global args
args = arg_parser.parse_args()
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
np.random.seed(args.manual_seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.makedirs(args.path, exist_ok=True)

args.distributed = False

if args.distributed:
    torch.cuda.set_device(args.local_rank)  # kihwan. 220220. for DDP
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
try:
    from tensorboard_logger import configure, log_value
except BaseException:
    configure = None

def get_bops_reduction_rate(bops):
    if args.arch_type == 'V1+SSD':
        ref_bops = 97.72715
    elif args.arch_type == 'V1+SSD_Lite':
        ref_bops = 69.71847
    elif args.arch_type == 'CAPP+SSD':
        ref_bops = 57.22179
    else:
        AssertionError("Do not support arch_type : {}".format(args.arch_type))
    return (1- bops/ref_bops) *100

def add_prefix(state_dict, prefix='module.'):
    new_dict = {}
    for k,v in state_dict.items():
        new_dict[prefix+k] = v
    return new_dict

def str_to_policy(policy_str, batch_size, num_blocks):
    policy_list = []
    policy_list.extend(policy_str)
    policy_result = []
    n = []
    for i, _n in enumerate(policy_list):
        n.append(_n)
        if len(n) == 4:
            t = [float(n[0]), float(n[1]), float(n[2]), float(n[3])]
            i_row = np.array(t, dtype=np.float64)
            n = []
            policy_result.append(i_row)
    policy_result = np.stack([p for p in policy_result],axis=0)
    policy_result = torch.tensor(policy_result)
    policy_result = torch.unsqueeze(policy_result, 0)
    policy_result = policy_result.repeat(batch_size,1,1)
    return policy_result   
def remove_prefix(state_dict, prefix):
    print('remove prefix %s', prefix )
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def add_prefix(state_dict, prefix='module.'):
    new_dict = {}
    for k,v in state_dict.items():
        new_dict[prefix+k] = v
    return new_dict

def getModel(arch, **kargs):
    if 'V1' in args.arch_type:
        base_net = QT_MobileNet_V1_224(num_classes=1001, full_pretrain=args.full_pretrain) #InstaNet
    elif 'V2' in args.arch_type:
        base_net = QT_MobileNet_V2_224(num_classes=1001, full_pretrain=args.full_pretrain) #InstaNet
    elif 'CAPP' in args.arch_type:
        base_net = QT_MobileNet_CAPP_224(num_classes=1001, full_pretrain=args.full_pretrain) #InstaNet
    else:
        raise NotImplementedError("Not existing arch_type")
    if 'V1+SSD' == args.arch_type:
        model = create_mobilenetv1_ssd_aug(base_net, num_classes=args.num_classes, wbit=args.extras_wbit, abit=args.extras_abit, head_wbit=args.head_wbit, head_abit=args.head_abit, config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain) # SSD
    elif 'V1+SSD_Lite' == args.arch_type:
        model = create_mobilenetv1_ssd_lite_aug(base_net, num_classes=args.num_classes, wbit=8, abit=8, head_wbit=8, head_abit=8, config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain) # SSD
    elif 'CAPP+SSD' == args.arch_type:
        model = create_mobilenetv1_capp_ssd_aug(base_net, num_classes=args.num_classes, wbit=8, abit=8, head_wbit=8, head_abit=8, config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain) # SSD
    elif 'V2+SSD' == args.arch_type:
        model = create_mobilenetv2_ssd_aug(base_net, num_classes=args.num_classes, wbit=8, abit=8, head_wbit=8, head_abit=8, config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain) # SSD
    elif 'V2+SSD_Lite' == args.arch_type:
        model = create_mobilenetv2_ssd_lite_aug(base_net, num_classes=args.num_classes, wbit=8, abit=8, head_wbit=8, head_abit=8, config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain) # SSD
    else:
        raise NotImplementedError("Not existing arch_type")
    if args.instassd_chkpt:
        instanet_checkpoint = torch.load(args.instassd_chkpt)
        new_state = remove_prefix(instanet_checkpoint, 'module.')
        new_state.update(new_state)
        try:
            model.load_state_dict(new_state)
            print("Loaded ssd from pretrained model!")
        except:
            print("could not load weight")
    args.start_epoch = 0
    model = torch.nn.DataParallel(model).cuda()
    print_model_parm_nums(model)
    return model
def get_agent( **kargs):
    agent = Policy224_HW([1,1,1,1], num_blocks=13, num_of_actions=4)
    if args.agent_chkpt:
        checkpoint= torch.load(args.agent_chkpt)
        agent.load_state_dict(checkpoint['agent'])
        print("Loaded agent!")
        agent = torch.nn.DataParallel(agent).cuda()
    return agent

def main():
    test_policies()

def test_policies():
    # optionally resume from a checkpoint
    dt = {'num_classes':args.num_classes, 'augmentation':args.augmentation}
    if 'V1' in args.arch_type or 'CAPP' in args.arch_type:
        ssd_config = generate_mb1_ssd_config(args.image_size, args.iou_threshold, args.center_variance, args.size_variance) #prior, specs, thresholds
    elif 'V2' in args.arch_type:
        ssd_config = generate_mb2_ssd_config(args.image_size, args.iou_threshold, args.center_variance, args.size_variance) #prior, specs, thresholds
    else:
        raise NotImplementedError("Not existing arch_type")

    dt['ssd_config']= ssd_config
    args.config_of_data = dt
    if args.resume is True:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            #old_args = checkpoint['args']
            #print('Old args:')
            #print(old_args)
            #for name in arch_resume_names:
            #    if name in vars(args) and name in vars(old_args):
            #        setattr(args, name, getattr(old_args, name))
            model = getModel(**vars(args))
            model.load_state_dict(add_prefix(checkpoint['instanet']))
            model.cuda()
            agent = get_agent(**vars(args))
            #agent.load_state_dict(add_prefix(checkpoint['agent']))
            agent.load_state_dict(checkpoint['agent'])
            agent.cuda()
            print("=> loaded checkpoint '{}' (epoch {})"
                 .format(args.resume, checkpoint['epoch']))
        else:
            print(
                "=> no checkpoint found at '{}'".format(
                    Fore.RED +
                    args.resume_path +
                    Fore.RESET),
                file=sys.stderr)
            return
    else:
        return

    if 'V1' in args.arch_type or 'CAPP' in args.arch_type:
        predictor = create_mobilenetv1_ssd_predictor(model, ssd_config.config)
    elif 'V2' in args.arch_type:
        predictor = create_mobilenetv2_ssd_predictor(model, ssd_config.config)
    else:
        raise NotImplementedError("Not existing arch_type")
    # set random seed
    optimizer = get_optimizer(model, args)
    criterion = MultiboxLoss(ssd_config.config['priors'], iou_threshold=args.iou_threshold, neg_pos_ratio=3, center_variance=args.center_variance, size_variance=args.size_variance, conf_threshold=0.1)

 
    Trainer = import_module(args.trainer).Trainer
    trainer = Trainer(model, criterion, optimizer, args, agent)
    
    # create dataloader
    test_transform = TestTransform(300, np.array([0,0,0]), 1.0)
    test_dataset = VOCDataset('/data/dataset/VOCdevkit/VOC2007TEST/', transform=test_transform, is_test=True)    
    test_loader =  torchdata.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, \
               collate_fn=collate_voc_batch, drop_last=False)
    trainer.set_true_labels(test_dataset)
    
    # evaluate
    _test_map, bops, policy_set = trainer.ssd_test(test_loader=test_loader, epoch=0, dataset=test_dataset, predictor=predictor, eval_path='./test', config=ssd_config.config, conf_threshold=args.conf_threshold, test_policy=False, return_policy=True)
    #_test_map, bops, policy_set = trainer.ssd_test(test_loader=test_loader, epoch=0, dataset=test_dataset, predictor=predictor, eval_path='./test', config=ssd_config.config, conf_threshold=args.conf_threshold, test_policy=True, return_policy=True)
    print("Test mAP : %.4f" %(_test_map))
    bops = bops.cpu().data.numpy()
    bops = np.mean(bops) 
    print("Test BOPs (G):")
    print(bops)
    if args.arch_type == 'MBv1+SSD':
        reduction_rate = get_bops_reduction_rate(bops)
        print("Reduction rate :")
        print(reduction_rate)
    bits = ['1', '2', '4', '8']
    policy_decimal = []
    for p in policy_set:
        tmp = []
        tmp.extend(p)
        count = 0
        p_decimal = ''
        for i in range(13):
            one_hot = np.array(tmp[4*i:4*i+4])
            decimal = bits[int(np.argmax(one_hot))]
            p_decimal += decimal
        policy_decimal.append(p_decimal)
    policy_decimal = set(policy_decimal)
    print(policy_decimal)
    print(len(policy_decimal))

    
if __name__ == '__main__':
    main()
