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
from models.instanas import QT_MobileNet_V1
from models.instanas_v2lite import QT_MobileNet_V2
from models.mb1_ssd import create_mobilenetv1_ssd_aug, create_mobilenetv1_ssd_predictor
from models.mb1_capp_ssd import create_mobilenetv1_capp_ssd_aug
from models.mb2_ssd_lite import create_mobilenetv2_ssd_lite_aug, create_mobilenetv2_ssd_predictor
from models.controller import Policy224
## utils
from colorama import Fore
from importlib import import_module
import config
from dataloader import getDataloaders
from utils import save_checkpoint, get_optimizer, create_save_folder, print_model_parm_nums
from args import arg_parser, arch_resume_names
from models.multibox_loss import MultiboxLoss
from models.mb1_ssd_config import generate_mb1_ssd_config
from models.mb2_ssd_config import generate_mb2_ssd_config
from data_provider.voc_dataset import VOCDataset
from data_provider.data_preprocessing import TestTransform
## DDP
from distributed import init_dist
from distributed import master_only_print as mprint # DDP. kihwan
from distributed import is_master
from utils import collate_voc_batch

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
torch.multiprocessing.set_sharing_strategy('file_system')
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)  # kihwan. 220220. for DDP
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

try:
    from tensorboard_logger import configure, log_value
except BaseException:
    configure = None

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
    '''
        fixed here
    '''
    # import pdb; pdb.set_trace()
    if 'V1' in args.arch_type: # fixed for implementing Linear QAT
        base_net = QT_MobileNet_V1(num_classes=1001, full_pretrain=args.full_pretrain, ActQ=args.ActQ, abit=args.abit, action_list=args.action_list) #InstaNet
        # base_net = LQT_MobileNet_V1_224(num_classes=1001, full_pretrain=args.full_pretrain) #InstaNet
    elif 'V2' in args.arch_type:
        base_net = QT_MobileNet_V2(num_classes=1001, full_pretrain=args.full_pretrain, ActQ=args.ActQ, abit=args.abit, action_list=args.action_list) #InstaNet
    else:
        raise NotImplementedError("Not existing arch_type")
    if 'V1+SSD' == args.arch_type: # fixed for implementing Linear QAT
        '''
            fixed here
        '''
        model = create_mobilenetv1_ssd_aug(base_net, num_classes=args.num_classes, wbit=args.extras_wbit, abit=args.extras_abit, head_wbit=args.head_wbit, head_abit=args.head_abit, 
                                           config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain, ActQ=args.ActQ) # SSD
        # model = create_mobilenetv1_ssd_aug_linear_quant(base_net, num_classes=args.num_classes, wbit=8, abit=8, head_wbit=8, head_abit=8, config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain) # SSD
    elif 'V2+SSD_Lite' == args.arch_type:
        model = create_mobilenetv2_ssd_lite_aug(base_net, num_classes=args.num_classes, wbit=args.extras_wbit, abit=args.extras_abit, head_wbit=args.head_wbit, head_abit=args.head_abit, 
                                                config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain, ActQ=args.ActQ) # SSD
    else:
        raise NotImplementedError("Not existing arch_type")
    if args.ssd_model:
        model.init_from_pretrained_ssd(args.ssd_model)
    if args.instanet_chkpt:
        instanet_checkpoint = torch.load(args.instanet_chkpt)
        #if args.test_ImgNet:
        new_state = remove_prefix(instanet_checkpoint['state_dict'], 'module.')
        new_state.update(new_state)
        model.base_net.load_state_dict(new_state)
        print("Loaded basenet from pretrained model!")
    if args.dist:
        gpu_id = init_dist()
        model = torch.nn.parallel.DistributedDataParallel(
                model.cuda(), [gpu_id], gpu_id, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()
    print_model_parm_nums(model)
    return model
def get_agent( **kargs):
    agent = Policy224([1,1,1,1], num_blocks=13, num_of_actions=4)
    if args.agent_chkpt:
        checkpoint= torch.load(args.agent_chkpt)
        agent.load_state_dict(checkpoint['agent'])
        mprint("Loaded agent!")
    if args.dist:
        gpu_id = init_dist()
        agent = torch.nn.parallel.DistributedDataParallel(
                agent.cuda(), [gpu_id], gpu_id, find_unused_parameters=True)
    else:
        agent = torch.nn.DataParallel(agent).cuda()
    return agent
def init_multiprocessing():
    try:
        torch.multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass

def main():
    init_multiprocessing()
    train_val_test()
def train_val_test():
    mprint("Is FP train? ")
    mprint(args.full_pretrain)
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
            # breakpoint()
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            
            if 'args' in checkpoint.keys():
                old_args = checkpoint['args']
                print('Old args:')
                print(old_args)
                for name in arch_resume_names:
                    if name in vars(args) and name in vars(old_args):
                        setattr(args, name, getattr(old_args, name))
            # set args based on checkpoint
            # breakpoint()
            if args.retraining is True:
                print('Training starts from epoch=0...')
                args.start_epoch = 0
            else:
                print('Training starts from epoch={}'.format(checkpoint['epoch'] + 1))
                args.start_epoch = checkpoint['epoch'] + 1
            
            model = getModel(**vars(args))
            # model(dummy_input, dummy_policy)
            if args.arch_type == 'V1+SSD':
                if args.from_fp_pretrain:
                    ckpt_to_load = checkpoint['state_dict']
                    fp = [0, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10, 11, 12]
                    qt = [0, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10, 11, 12]

                    # layer 1~13
                    for lidx in range(len(qt)):
                        for bidx in range(len(args.action_list)):
                            qt_name_w = 'module.base_net.layers.{}.{}.conv1.weight'.format(qt[lidx], bidx)
                            pt_name_w = 'module.base_net.layers.{}.0.conv1.weight'.format(fp[lidx])
                            ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                            pt_name_w = 'module.base_net.layers.{}.0.bn1.weight'.format(fp[lidx])
                            pt_name_b = 'module.base_net.layers.{}.0.bn1.bias'.format(fp[lidx])
                            pt_name_rm = 'module.base_net.layers.{}.0.bn1.running_mean'.format(fp[lidx])
                            pt_name_rv = 'module.base_net.layers.{}.0.bn1.running_var'.format(fp[lidx])
                            qt_name_w = 'module.base_net.layers.{}.{}.bn1.weight'.format(qt[lidx], bidx)
                            qt_name_b = 'module.base_net.layers.{}.{}.bn1.bias'.format(qt[lidx], bidx)
                            qt_name_rm = 'module.base_net.layers.{}.{}.bn1.running_mean'.format(qt[lidx], bidx)
                            qt_name_rv = 'module.base_net.layers.{}.{}.bn1.running_var'.format(qt[lidx], bidx)
                            ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                            ckpt_to_load[qt_name_b] = checkpoint['state_dict'][pt_name_b]
                            ckpt_to_load[qt_name_rm] = checkpoint['state_dict'][pt_name_rm]
                            ckpt_to_load[qt_name_rv] = checkpoint['state_dict'][pt_name_rv]
                            
                            qt_name_w = 'module.base_net.layers.{}.{}.conv2.weight'.format(qt[lidx], bidx)
                            pt_name_w = 'module.base_net.layers.{}.0.conv2.weight'.format(fp[lidx])
                            ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                            pt_name_w = 'module.base_net.layers.{}.0.bn2.weight'.format(fp[lidx])
                            pt_name_b = 'module.base_net.layers.{}.0.bn2.bias'.format(fp[lidx])
                            pt_name_rm = 'module.base_net.layers.{}.0.bn2.running_mean'.format(fp[lidx])
                            pt_name_rv = 'module.base_net.layers.{}.0.bn2.running_var'.format(fp[lidx])
                            qt_name_w = 'module.base_net.layers.{}.{}.bn2.weight'.format(qt[lidx], bidx)
                            qt_name_b = 'module.base_net.layers.{}.{}.bn2.bias'.format(qt[lidx], bidx)
                            qt_name_rm = 'module.base_net.layers.{}.{}.bn2.running_mean'.format(qt[lidx], bidx)
                            qt_name_rv = 'module.base_net.layers.{}.{}.bn2.running_var'.format(qt[lidx], bidx)
                            ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                            ckpt_to_load[qt_name_b] = checkpoint['state_dict'][pt_name_b]
                            ckpt_to_load[qt_name_rm] = checkpoint['state_dict'][pt_name_rm]
                            ckpt_to_load[qt_name_rv] = checkpoint['state_dict'][pt_name_rv]
       
                   
                else:
                    fp = [1, 2, 3, 4,  5,  6,  7,  8,  9,  10, 11, 12, 13]
                    qt = [0, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10, 11, 12]
                    ckpt_to_load = dict()
                    # conv1
                    qt_name_w = 'module.base_net.conv1.conv_q.conv.weight'
                    pt_name_w = 'base_net.0.0.weight'
                    ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                    pt_name_w = 'base_net.0.1.weight'
                    pt_name_b = 'base_net.0.1.bias'
                    pt_name_rm = 'base_net.0.1.running_mean'
                    pt_name_rv = 'base_net.0.1.running_var'
                    qt_name_w = 'module.base_net.conv1.conv_q.bn.weight'
                    qt_name_b = 'module.base_net.conv1.conv_q.bn.bias'
                    qt_name_rm = 'module.base_net.conv1.conv_q.bn.running_mean'
                    qt_name_rv = 'module.base_net.conv1.conv_q.bn.running_var'
                    ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                    ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                    ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                    ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                    
                
                    # layer 1~13
                    for lidx in range(len(qt)):
                        for bidx in range(len(args.action_list)):
                            qt_name_w = 'module.base_net.layers.{}.{}.conv1.weight'.format(qt[lidx], bidx)
                            pt_name_w = 'base_net.{}.0.weight'.format(fp[lidx])
                            ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                            pt_name_w = 'base_net.{}.1.weight'.format(fp[lidx])
                            pt_name_b = 'base_net.{}.1.bias'.format(fp[lidx])
                            pt_name_rm = 'base_net.{}.1.running_mean'.format(fp[lidx])
                            pt_name_rv = 'base_net.{}.1.running_var'.format(fp[lidx])
                            qt_name_w = 'module.base_net.layers.{}.{}.bn1.weight'.format(qt[lidx], bidx)
                            qt_name_b = 'module.base_net.layers.{}.{}.bn1.bias'.format(qt[lidx], bidx)
                            qt_name_rm = 'module.base_net.layers.{}.{}.bn1.running_mean'.format(qt[lidx], bidx)
                            qt_name_rv = 'module.base_net.layers.{}.{}.bn1.running_var'.format(qt[lidx], bidx)
                            ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                            ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                            ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                            ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                            
                            qt_name_w = 'module.base_net.layers.{}.{}.conv2.weight'.format(qt[lidx], bidx)
                            pt_name_w = 'base_net.{}.3.weight'.format(fp[lidx])
                            ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                            pt_name_w = 'base_net.{}.4.weight'.format(fp[lidx])
                            pt_name_b = 'base_net.{}.4.bias'.format(fp[lidx])
                            pt_name_rm = 'base_net.{}.4.running_mean'.format(fp[lidx])
                            pt_name_rv = 'base_net.{}.4.running_var'.format(fp[lidx])
                            qt_name_w = 'module.base_net.layers.{}.{}.bn2.weight'.format(qt[lidx], bidx)
                            qt_name_b = 'module.base_net.layers.{}.{}.bn2.bias'.format(qt[lidx], bidx)
                            qt_name_rm = 'module.base_net.layers.{}.{}.bn2.running_mean'.format(qt[lidx], bidx)
                            qt_name_rv = 'module.base_net.layers.{}.{}.bn2.running_var'.format(qt[lidx], bidx)
                            ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                            ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                            ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                            ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]

         
                    # extras
                    
                    for lidx in range(4):
                        qt_name_w = 'module.extras.{}.0.conv_q.conv.weight'.format(lidx)
                        pt_name_w = 'extras.{}.0.weight'.format(lidx)
                        qt_name_b = 'module.extras.{}.0.conv_q.conv.bias'.format(lidx)
                        pt_name_b = 'extras.{}.0.bias'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        
                        qt_name_w = 'module.extras.{}.1.conv_q.conv.weight'.format(lidx)
                        pt_name_w = 'extras.{}.2.weight'.format(lidx)
                        qt_name_b = 'module.extras.{}.1.conv_q.conv.bias'.format(lidx)
                        pt_name_b = 'extras.{}.2.bias'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                    
                    # headers
                    # breakpoint()
                    for lidx in range(5):
                        qt_name_w = 'module.classification_headers.{}.conv_q.conv.weight'.format(lidx)
                        pt_name_w = 'classification_headers.{}.weight'.format(lidx)
                        qt_name_b = 'module.classification_headers.{}.conv_q.conv.bias'.format(lidx)
                        pt_name_b = 'classification_headers.{}.bias'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        
                    for lidx in range(5):
                        qt_name_w = 'module.regression_headers.{}.conv_q.conv.weight'.format(lidx)
                        pt_name_w = 'regression_headers.{}.weight'.format(lidx)
                        qt_name_b = 'module.regression_headers.{}.conv_q.conv.bias'.format(lidx)
                        pt_name_b = 'regression_headers.{}.bias'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        
            
            elif args.arch_type == 'V2+SSD_Lite':
                if args.from_fp_pretrain:
                    
                    fp = [1, 2, 3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16]
                    qt = [1, 2, 3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16]
                    
                    ckpt_to_load = checkpoint['state_dict']
                    print("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
                    
                    # layer1
                    for bidx in range(len(args.action_list)):
                    # for bidx in range(4) :
                        qt_name_w = 'module.base_net.layers.0.{}.conv1.weight'.format(bidx)
                        pt_name_w = 'module.base_net.layers.0.0.conv1.weight'
                        ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                        pt_name_w = 'module.base_net.layers.0.0.bn1.weight'
                        pt_name_b = 'module.base_net.layers.0.0.bn1.bias'
                        pt_name_rm = 'module.base_net.layers.0.0.bn1.running_mean'
                        pt_name_rv = 'module.base_net.layers.0.0.bn1.running_var'
                        qt_name_w = 'module.base_net.layers.0.{}.bn1.weight'.format(bidx)
                        qt_name_b = 'module.base_net.layers.0.{}.bn1.bias'.format(bidx)
                        qt_name_rm = 'module.base_net.layers.0.{}.bn1.running_mean'.format(bidx)
                        qt_name_rv = 'module.base_net.layers.0.{}.bn1.running_var'.format(bidx)
                        ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint['state_dict'][pt_name_b]
                        ckpt_to_load[qt_name_rm] = checkpoint['state_dict'][pt_name_rm]
                        ckpt_to_load[qt_name_rv] = checkpoint['state_dict'][pt_name_rv]
                        
                        qt_name_w = 'module.base_net.layers.0.{}.conv2.weight'.format(bidx)
                        pt_name_w = 'module.base_net.layers.0.0.conv2.weight'
                        ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                        pt_name_w = 'module.base_net.layers.0.0.bn2.weight'
                        pt_name_b = 'module.base_net.layers.0.0.bn2.bias'
                        pt_name_rm = 'module.base_net.layers.0.0.bn2.running_mean'
                        pt_name_rv = 'module.base_net.layers.0.0.bn2.running_var'
                        qt_name_w = 'module.base_net.layers.0.{}.bn2.weight'.format(bidx)
                        qt_name_b = 'module.base_net.layers.0.{}.bn2.bias'.format(bidx)
                        qt_name_rm = 'module.base_net.layers.0.{}.bn2.running_mean'.format(bidx)
                        qt_name_rv = 'module.base_net.layers.0.{}.bn2.running_var'.format(bidx)
                        ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint['state_dict'][pt_name_b]
                        ckpt_to_load[qt_name_rm] = checkpoint['state_dict'][pt_name_rm]
                        ckpt_to_load[qt_name_rv] = checkpoint['state_dict'][pt_name_rv]
                        
                    # layer 2~17
                    for lidx in range(len(qt)):
                        for bidx in range(len(args.action_list)):
                            qt_name_w = 'module.base_net.layers.{}.{}.conv1.weight'.format(qt[lidx], bidx)
                            pt_name_w = 'module.base_net.layers.{}.0.conv1.weight'.format(fp[lidx])
                            ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                            pt_name_w = 'module.base_net.layers.{}.0.bn1.weight'.format(fp[lidx])
                            pt_name_b = 'module.base_net.layers.{}.0.bn1.bias'.format(fp[lidx])
                            pt_name_rm = 'module.base_net.layers.{}.0.bn1.running_mean'.format(fp[lidx])
                            pt_name_rv = 'module.base_net.layers.{}.0.bn1.running_var'.format(fp[lidx])
                            qt_name_w = 'module.base_net.layers.{}.{}.bn1.weight'.format(qt[lidx], bidx)
                            qt_name_b = 'module.base_net.layers.{}.{}.bn1.bias'.format(qt[lidx], bidx)
                            qt_name_rm = 'module.base_net.layers.{}.{}.bn1.running_mean'.format(qt[lidx], bidx)
                            qt_name_rv = 'module.base_net.layers.{}.{}.bn1.running_var'.format(qt[lidx], bidx)
                            ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                            ckpt_to_load[qt_name_b] = checkpoint['state_dict'][pt_name_b]
                            ckpt_to_load[qt_name_rm] = checkpoint['state_dict'][pt_name_rm]
                            ckpt_to_load[qt_name_rv] = checkpoint['state_dict'][pt_name_rv]
                            
                            qt_name_w = 'module.base_net.layers.{}.{}.conv2.weight'.format(qt[lidx], bidx)
                            pt_name_w = 'module.base_net.layers.{}.0.conv2.weight'.format(fp[lidx])
                            ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                            pt_name_w = 'module.base_net.layers.{}.0.bn2.weight'.format(fp[lidx])
                            pt_name_b = 'module.base_net.layers.{}.0.bn2.bias'.format(fp[lidx])
                            pt_name_rm = 'module.base_net.layers.{}.0.bn2.running_mean'.format(fp[lidx])
                            pt_name_rv = 'module.base_net.layers.{}.0.bn2.running_var'.format(fp[lidx])
                            qt_name_w = 'module.base_net.layers.{}.{}.bn2.weight'.format(qt[lidx], bidx)
                            qt_name_b = 'module.base_net.layers.{}.{}.bn2.bias'.format(qt[lidx], bidx)
                            qt_name_rm = 'module.base_net.layers.{}.{}.bn2.running_mean'.format(qt[lidx], bidx)
                            qt_name_rv = 'module.base_net.layers.{}.{}.bn2.running_var'.format(qt[lidx], bidx)
                            ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                            ckpt_to_load[qt_name_b] = checkpoint['state_dict'][pt_name_b]
                            ckpt_to_load[qt_name_rm] = checkpoint['state_dict'][pt_name_rm]
                            ckpt_to_load[qt_name_rv] = checkpoint['state_dict'][pt_name_rv]
                            
                            qt_name_w = 'module.base_net.layers.{}.{}.conv3.weight'.format(qt[lidx], bidx)
                            pt_name_w = 'module.base_net.layers.{}.0.conv3.weight'.format(fp[lidx])
                            ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                            pt_name_w = 'module.base_net.layers.{}.0.bn3.weight'.format(fp[lidx])
                            pt_name_b = 'module.base_net.layers.{}.0.bn3.bias'.format(fp[lidx])
                            pt_name_rm = 'module.base_net.layers.{}.0.bn3.running_mean'.format(fp[lidx])
                            pt_name_rv = 'module.base_net.layers.{}.0.bn3.running_var'.format(fp[lidx])
                            qt_name_w = 'module.base_net.layers.{}.{}.bn3.weight'.format(qt[lidx], bidx)
                            qt_name_b = 'module.base_net.layers.{}.{}.bn3.bias'.format(qt[lidx], bidx)
                            qt_name_rm = 'module.base_net.layers.{}.{}.bn3.running_mean'.format(qt[lidx], bidx)
                            qt_name_rv = 'module.base_net.layers.{}.{}.bn3.running_var'.format(qt[lidx], bidx)
                            ckpt_to_load[qt_name_w] = checkpoint['state_dict'][pt_name_w]
                            ckpt_to_load[qt_name_b] = checkpoint['state_dict'][pt_name_b]
                            ckpt_to_load[qt_name_rm] = checkpoint['state_dict'][pt_name_rm]
                            ckpt_to_load[qt_name_rv] = checkpoint['state_dict'][pt_name_rv]
                else:
                    # copy parameters to 1, 2, 4, 8 bit parameters of model
                    '''
                    <fp checkpoint>
                    ['base_net.0.0.weight', --> conv
                    'base_net.0.1.weight', 'base_net.0.1.bias', 'base_net.0.1.running_mean', 'base_net.0.1.running_var'--> bn
                    
                    'base_net.1~17.conv.0.weight', --> conv1
                    'base_net.1~17.conv.1.weight', 'base_net.1.conv.1.bias', 'base_net.1.conv.1.running_mean', 'base_net.1.conv.1.running_var'--> bn
                    'base_net.1~17.conv.3.weight' --> conv2
                    'base_net.1~17.conv.4.weight', 'base_net.4.conv.1.bias', 'base_net.1.conv.4.running_mean', 'base_net.1.conv.4.running_var'--> bn
                    'base_net.2~17.conv.6.weight' --> conv3
                    'base_net.2~17.conv.7.weight', 'base_net.7.conv.1.bias', 'base_net.1.conv.7.running_mean', 'base_net.1.conv.7.running_var'--> bn
                    
                    'base_net.18.0.weight', --> conv
                    'base_net.18.1.weight', 'base_net.0.1.bias', 'base_net.0.1.running_mean', 'base_net.0.1.running_var'--> bn
                    
                    'extras.0~3.conv.0.weight', 'extras.0.conv.1.weight', 'extras.0.conv.1.bias', 'extras.0.conv.1.running_mean', 'extras.0.conv.1.running_var',
                    'extras.0~3.conv.3.weight', 'extras.0.conv.4.weight', 'extras.0.conv.4.bias', 'extras.0.conv.4.running_mean', 'extras.0.conv.4.running_var',
                    'extras.0~3.conv.6.weight', 'extras.0.conv.7.weight', 'extras.0.conv.7.bias', 'extras.0.conv.7.running_mean', 'extras.0.conv.7.running_var',
                    
                    'classification_headers.0~5.0.weight', classification_headers.0.0.bias', 'classification_headers.0.1.weight', classification_headers.0.1.bias',
                    'classification_headers.0.3.weight', classification_headers.0.3.bias'
                    
                    
                    
                    <model state_dict>
                    'base_net.conv1.conv_q.conv.weight'
                    'base_net.conv1.conv_q.bn.weight'
                    'base_net.conv1.conv_q.bn.bias'
                    'base_net.conv1.conv_q.bn.running_mean'
                    'base_net.conv1.conv_q.bn.running_var'
                    
                    'base_net.layers.0.0~3.conv1.weight'
                    'base_net.layers.0.0~3.bn1.weight'
                    'base_net.layers.0.0~3.bn1.bias'
                    'base_net.layers.0.0~3.bn1.running_mean'
                    'base_net.layers.0.0~3.bn1.running_var'
                    'base_net.layers.0.0~3.conv2.weight'
                    'base_net.layers.0.0~3.bn2.weight'
                    'base_net.layers.0.0~3.bn2.bias'
                    'base_net.layers.0.0~3.bn2.running_mean'
                    'base_net.layers.0.0~3.bn2.running_var'
                    
                    'base_net.layers.1~16.0~3.conv1.weight'
                    'base_net.layers.1~16.0~3.bn1.weight'
                    'base_net.layers.1~16.0~3.bn1.bias'
                    'base_net.layers.1~16.0~3.bn1.running_mean'
                    'base_net.layers.1~16.0~3.bn1.running_var'
                    'base_net.layers.1~16.0~3.conv2.weight'
                    'base_net.layers.1~16.0~3.bn2.weight'
                    'base_net.layers.1~16.0~3.bn2.bias'
                    'base_net.layers.1~16.0~3.bn2.running_mean'
                    'base_net.layers.1~16.0~3.bn2.running_var'
                    'base_net.layers.1~16.0~3.conv3.weight'
                    'base_net.layers.1~16.0~3.bn3.weight'
                    'base_net.layers.1~16.0~3.bn3.bias'
                    'base_net.layers.1~16.0~3.bn3.running_mean'
                    'base_net.layers.1~16.0~3.bn3.running_var'

                    'base_net.conv2.conv_q.conv.weight'
                    'base_net.conv2.conv_q.bn.weight'
                    'base_net.conv2.conv_q.bn.bias'
                    'base_net.conv2.conv_q.bn.running_mean'
                    'base_net.conv2.conv_q.bn.running_var'
                    
                    'extras.0~3.0.conv1.weight'
                    'extras.0~3.0.bn1.weight'
                    'extras.0.0.bn1.bias'
                    'extras.0.0.bn1.running_mean'
                    'extras.0.0.bn1.running_var'
                    'extras.0.0.conv2.weight'
                    'extras.0.0.bn2.weight'
                    'extras.0.0.bn2.bias'
                    'extras.0.0.bn2.running_mean'
                    'extras.0.0.bn2.running_var'
                    
                    classification_headers.0.conv_sep.conv.weight
                    classification_headers.0.conv_sep.bn.weight
                    classification_headers.0.conv_sep.bn.bias
                    classification_headers.0.conv_sep.bn.running_mean
                    classification_headers.0.conv_sep.bn.running_var
                    classification_headers.0.conv_point.conv.weight
                    
                    '''
                    # breakpoint()
                    fp = [2, 3, 4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17]
                    # qt = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19]
                    qt = [1, 2, 3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16]
                    ckpt_to_load = dict()
                    # conv1
                    qt_name_w = 'module.base_net.conv1.conv_q.conv.weight'
                    pt_name_w = 'base_net.0.0.weight'
                    ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                    pt_name_w = 'base_net.0.1.weight'
                    pt_name_b = 'base_net.0.1.bias'
                    pt_name_rm = 'base_net.0.1.running_mean'
                    pt_name_rv = 'base_net.0.1.running_var'
                    qt_name_w = 'module.base_net.conv1.conv_q.bn.weight'
                    qt_name_b = 'module.base_net.conv1.conv_q.bn.bias'
                    qt_name_rm = 'module.base_net.conv1.conv_q.bn.running_mean'
                    qt_name_rv = 'module.base_net.conv1.conv_q.bn.running_var'
                    ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                    ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                    ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                    ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                    
                    # conv2
                    qt_name_w = 'module.base_net.conv2.conv_q.conv.weight'
                    pt_name_w = 'base_net.18.0.weight'
                    ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                    pt_name_w = 'base_net.18.1.weight'
                    pt_name_b = 'base_net.18.1.bias'
                    pt_name_rm = 'base_net.18.1.running_mean'
                    pt_name_rv = 'base_net.18.1.running_var'
                    qt_name_w = 'module.base_net.conv2.conv_q.bn.weight'
                    qt_name_b = 'module.base_net.conv2.conv_q.bn.bias'
                    qt_name_rm = 'module.base_net.conv2.conv_q.bn.running_mean'
                    qt_name_rv = 'module.base_net.conv2.conv_q.bn.running_var'
                    ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                    ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                    ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                    ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                    
                    # layer1
                    for bidx in range(len(args.action_list)):
                    # for bidx in range(4) :
                        qt_name_w = 'module.base_net.layers.0.{}.conv1.weight'.format(bidx)
                        pt_name_w = 'base_net.1.conv.0.weight'
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        pt_name_w = 'base_net.1.conv.1.weight'
                        pt_name_b = 'base_net.1.conv.1.bias'
                        pt_name_rm = 'base_net.1.conv.1.running_mean'
                        pt_name_rv = 'base_net.1.conv.1.running_var'
                        qt_name_w = 'module.base_net.layers.0.{}.bn1.weight'.format(bidx)
                        qt_name_b = 'module.base_net.layers.0.{}.bn1.bias'.format(bidx)
                        qt_name_rm = 'module.base_net.layers.0.{}.bn1.running_mean'.format(bidx)
                        qt_name_rv = 'module.base_net.layers.0.{}.bn1.running_var'.format(bidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                        ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                        
                        qt_name_w = 'module.base_net.layers.0.{}.conv2.weight'.format(bidx)
                        pt_name_w = 'base_net.1.conv.3.weight'
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        pt_name_w = 'base_net.1.conv.4.weight'
                        pt_name_b = 'base_net.1.conv.4.bias'
                        pt_name_rm = 'base_net.1.conv.4.running_mean'
                        pt_name_rv = 'base_net.1.conv.4.running_var'
                        qt_name_w = 'module.base_net.layers.0.{}.bn2.weight'.format(bidx)
                        qt_name_b = 'module.base_net.layers.0.{}.bn2.bias'.format(bidx)
                        qt_name_rm = 'module.base_net.layers.0.{}.bn2.running_mean'.format(bidx)
                        qt_name_rv = 'module.base_net.layers.0.{}.bn2.running_var'.format(bidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                        ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                        
                    # layer 2~17
                    for lidx in range(len(qt)):
                        for bidx in range(len(args.action_list)):
                            qt_name_w = 'module.base_net.layers.{}.{}.conv1.weight'.format(qt[lidx], bidx)
                            pt_name_w = 'base_net.{}.conv.0.weight'.format(fp[lidx])
                            ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                            pt_name_w = 'base_net.{}.conv.1.weight'.format(fp[lidx])
                            pt_name_b = 'base_net.{}.conv.1.bias'.format(fp[lidx])
                            pt_name_rm = 'base_net.{}.conv.1.running_mean'.format(fp[lidx])
                            pt_name_rv = 'base_net.{}.conv.1.running_var'.format(fp[lidx])
                            qt_name_w = 'module.base_net.layers.{}.{}.bn1.weight'.format(qt[lidx], bidx)
                            qt_name_b = 'module.base_net.layers.{}.{}.bn1.bias'.format(qt[lidx], bidx)
                            qt_name_rm = 'module.base_net.layers.{}.{}.bn1.running_mean'.format(qt[lidx], bidx)
                            qt_name_rv = 'module.base_net.layers.{}.{}.bn1.running_var'.format(qt[lidx], bidx)
                            ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                            ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                            ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                            ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                            
                            qt_name_w = 'module.base_net.layers.{}.{}.conv2.weight'.format(qt[lidx], bidx)
                            pt_name_w = 'base_net.{}.conv.3.weight'.format(fp[lidx])
                            ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                            pt_name_w = 'base_net.{}.conv.4.weight'.format(fp[lidx])
                            pt_name_b = 'base_net.{}.conv.4.bias'.format(fp[lidx])
                            pt_name_rm = 'base_net.{}.conv.4.running_mean'.format(fp[lidx])
                            pt_name_rv = 'base_net.{}.conv.4.running_var'.format(fp[lidx])
                            qt_name_w = 'module.base_net.layers.{}.{}.bn2.weight'.format(qt[lidx], bidx)
                            qt_name_b = 'module.base_net.layers.{}.{}.bn2.bias'.format(qt[lidx], bidx)
                            qt_name_rm = 'module.base_net.layers.{}.{}.bn2.running_mean'.format(qt[lidx], bidx)
                            qt_name_rv = 'module.base_net.layers.{}.{}.bn2.running_var'.format(qt[lidx], bidx)
                            ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                            ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                            ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                            ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                            
                            qt_name_w = 'module.base_net.layers.{}.{}.conv3.weight'.format(qt[lidx], bidx)
                            pt_name_w = 'base_net.{}.conv.6.weight'.format(fp[lidx])
                            ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                            pt_name_w = 'base_net.{}.conv.7.weight'.format(fp[lidx])
                            pt_name_b = 'base_net.{}.conv.7.bias'.format(fp[lidx])
                            pt_name_rm = 'base_net.{}.conv.7.running_mean'.format(fp[lidx])
                            pt_name_rv = 'base_net.{}.conv.7.running_var'.format(fp[lidx])
                            qt_name_w = 'module.base_net.layers.{}.{}.bn3.weight'.format(qt[lidx], bidx)
                            qt_name_b = 'module.base_net.layers.{}.{}.bn3.bias'.format(qt[lidx], bidx)
                            qt_name_rm = 'module.base_net.layers.{}.{}.bn3.running_mean'.format(qt[lidx], bidx)
                            qt_name_rv = 'module.base_net.layers.{}.{}.bn3.running_var'.format(qt[lidx], bidx)
                            ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                            ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                            ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                            ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                                
                                
                                
                    # extras
                    
                    for lidx in range(4):
                        qt_name_w = 'module.extras.{}.0.conv1.weight'.format(lidx)
                        pt_name_w = 'extras.{}.conv.0.weight'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        qt_name_w = 'module.extras.{}.0.bn1.weight'.format(lidx)
                        qt_name_b = 'module.extras.{}.0.bn1.bias'.format(lidx)
                        qt_name_rm = 'module.extras.{}.0.bn1.running_mean'.format(lidx)
                        qt_name_rv = 'module.extras.{}.0.bn1.running_var'.format(lidx)
                        pt_name_w = 'extras.{}.conv.1.weight'.format(lidx)
                        pt_name_b = 'extras.{}.conv.1.bias'.format(lidx)
                        pt_name_rm = 'extras.{}.conv.1.running_mean'.format(lidx)
                        pt_name_rv = 'extras.{}.conv.1.running_var'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                        ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                        
                        qt_name_w = 'module.extras.{}.0.conv2.weight'.format(lidx)
                        pt_name_w = 'extras.{}.conv.3.weight'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        qt_name_w = 'module.extras.{}.0.bn2.weight'.format(lidx)
                        qt_name_b = 'module.extras.{}.0.bn2.bias'.format(lidx)
                        qt_name_rm = 'module.extras.{}.0.bn2.running_mean'.format(lidx)
                        qt_name_rv = 'module.extras.{}.0.bn2.running_var'.format(lidx)
                        pt_name_w = 'extras.{}.conv.4.weight'.format(lidx)
                        pt_name_b = 'extras.{}.conv.4.bias'.format(lidx)
                        pt_name_rm = 'extras.{}.conv.4.running_mean'.format(lidx)
                        pt_name_rv = 'extras.{}.conv.4.running_var'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                        ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                        
                        qt_name_w = 'module.extras.{}.0.conv3.weight'.format(lidx)
                        pt_name_w = 'extras.{}.conv.6.weight'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        qt_name_w = 'module.extras.{}.0.bn3.weight'.format(lidx)
                        qt_name_b = 'module.extras.{}.0.bn3.bias'.format(lidx)
                        qt_name_rm = 'module.extras.{}.0.bn3.running_mean'.format(lidx)
                        qt_name_rv = 'module.extras.{}.0.bn3.running_var'.format(lidx)
                        pt_name_w = 'extras.{}.conv.7.weight'.format(lidx)
                        pt_name_b = 'extras.{}.conv.7.bias'.format(lidx)
                        pt_name_rm = 'extras.{}.conv.7.running_mean'.format(lidx)
                        pt_name_rv = 'extras.{}.conv.7.running_var'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                        ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                    
                    # headers
                    # there is bias in header
                    for lidx in range(5):
                        # breakpoint()
                        qt_name_w = 'module.classification_headers.{}.conv_sep.conv.weight'.format(lidx)
                        pt_name_w = 'classification_headers.{}.0.weight'.format(lidx)
                        qt_name_b = 'module.classification_headers.{}.conv_sep.conv.bias'.format(lidx)
                        pt_name_b = 'classification_headers.{}.0.bias'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        qt_name_w = 'module.classification_headers.{}.conv_sep.bn.weight'.format(lidx)
                        qt_name_b = 'module.classification_headers.{}.conv_sep.bn.bias'.format(lidx)
                        qt_name_rm = 'module.classification_headers.{}.conv_sep.bn.running_mean'.format(lidx)
                        qt_name_rv = 'module.classification_headers.{}.conv_sep.bn.running_var'.format(lidx)
                        pt_name_w = 'classification_headers.{}.1.weight'.format(lidx)
                        pt_name_b = 'classification_headers.{}.1.bias'.format(lidx)
                        pt_name_rm = 'classification_headers.{}.1.running_mean'.format(lidx)
                        pt_name_rv = 'classification_headers.{}.1.running_var'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                        ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                        qt_name_w = 'module.classification_headers.{}.conv_point.conv.weight'.format(lidx)
                        pt_name_w = 'classification_headers.{}.3.weight'.format(lidx)
                        qt_name_b = 'module.classification_headers.{}.conv_point.conv.bias'.format(lidx)
                        pt_name_b = 'classification_headers.{}.3.bias'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                    qt_name_w = 'module.classification_headers.5.conv_q.conv.weight'
                    pt_name_w = 'classification_headers.5.weight'
                    qt_name_b = 'module.classification_headers.5.conv_q.conv.bias'
                    pt_name_b = 'classification_headers.5.bias'
                    ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                    ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                    
                    for lidx in range(5):
                        qt_name_w = 'module.regression_headers.{}.conv_sep.conv.weight'.format(lidx)
                        pt_name_w = 'regression_headers.{}.0.weight'.format(lidx)
                        qt_name_b = 'module.regression_headers.{}.conv_sep.conv.bias'.format(lidx)
                        pt_name_b = 'regression_headers.{}.0.bias'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        qt_name_w = 'module.regression_headers.{}.conv_sep.bn.weight'.format(lidx)
                        qt_name_b = 'module.regression_headers.{}.conv_sep.bn.bias'.format(lidx)
                        qt_name_rm = 'module.regression_headers.{}.conv_sep.bn.running_mean'.format(lidx)
                        qt_name_rv = 'module.regression_headers.{}.conv_sep.bn.running_var'.format(lidx)
                        pt_name_w = 'regression_headers.{}.1.weight'.format(lidx)
                        pt_name_b = 'regression_headers.{}.1.bias'.format(lidx)
                        pt_name_rm = 'regression_headers.{}.1.running_mean'.format(lidx)
                        pt_name_rv = 'regression_headers.{}.1.running_var'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                        ckpt_to_load[qt_name_rm] = checkpoint[pt_name_rm]
                        ckpt_to_load[qt_name_rv] = checkpoint[pt_name_rv]
                        qt_name_w = 'module.regression_headers.{}.conv_point.conv.weight'.format(lidx)
                        pt_name_w = 'regression_headers.{}.3.weight'.format(lidx)
                        qt_name_b = 'module.regression_headers.{}.conv_point.conv.bias'.format(lidx)
                        pt_name_b = 'regression_headers.{}.3.bias'.format(lidx)
                        ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                        ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                    qt_name_w = 'module.regression_headers.5.conv_q.conv.weight'
                    pt_name_w = 'regression_headers.5.weight'
                    qt_name_b = 'module.regression_headers.5.conv_q.conv.bias'
                    pt_name_b = 'regression_headers.5.bias'
                    ckpt_to_load[qt_name_w] = checkpoint[pt_name_w]
                    ckpt_to_load[qt_name_b] = checkpoint[pt_name_b]
                    
            # breakpoint()
            model.load_state_dict(ckpt_to_load, strict=False)
            
            load_check = 0
            load_add_key_check = 0
            load_miss_key_check = 0
            for k in ckpt_to_load.keys():
                if k not in model.state_dict().keys():
                    load_miss_key_check += 1
                    print('there is missing key in model: [' +k +']')
            print('=====> there is {} missing keys in model'.format(load_miss_key_check))
            for k in model.state_dict().keys():
                if k not in ckpt_to_load.keys():
                    load_add_key_check += 1
                    if('weight' in k or 'bias' in k or 'running' in k):
                        print('there is additional key in model: [' +k +']')
            print('=====> there is {} additional keys in model'.format(load_add_key_check))
            
        else:
            print(
                "=> no checkpoint found at '{}'".format(
                    Fore.RED +
                    args.resume_path +
                    Fore.RESET),
                file=sys.stderr)
            return
    else:
        # create model
        print("=> creating model '{}'".format(args.arch))
        model = getModel(**vars(args))
    # breakpoint()
    if args.full_pretrain is True:
        model.module.freeze_PACT_param()
    else:
        model.module.train_PACT_param()
    if args.freeze_basenet is True:
        model.module.freeze_base_param()
    cudnn.benchmark = True

    if 'V1' in args.arch_type or 'CAPP' in args.arch_type:
        predictor = create_mobilenetv1_ssd_predictor(model, ssd_config.config)
    elif 'V2' in args.arch_type:
        predictor = create_mobilenetv2_ssd_predictor(model, ssd_config.config)
    else:
        raise NotImplementedError("Not existing arch_type")
    criterion = MultiboxLoss(ssd_config.config['priors'], iou_threshold=args.iou_threshold, neg_pos_ratio=3, center_variance=args.center_variance, size_variance=args.size_variance, conf_threshold=0.1).cuda()

    # define optimizer
    optimizer = get_optimizer(model, args)

    # set random seed
    torch.manual_seed(args.seed)

    Trainer = import_module(args.trainer).Trainer
    if args.agent_chkpt:
        agent = get_agent(**vars(args))
        checkpoint = torch.load(args.agent_chkpt)
        agent.load_state_dict(add_prefix(checkpoint['agent']), strict=False)
        print(' [*] Loaded agent from', args.agent_chkpt)
        trainer = Trainer(model, criterion, optimizer, args, agent)
    else:
        trainer = Trainer(model, criterion, optimizer, args)

    # create dataloader
    if args.evaluate == 'train':
        sampler_train, _, _, train_loader, _, _ = getDataloaders(splits=('train'), **vars(args))
        trainer.test(train_loader, best_epoch)
        return
    elif args.evaluate == 'val':
        _, sampler_val, _, _, val_loader, _ = getDataloaders(splits=('val'), **vars(args))
        trainer.test(val_loader, best_epoch)
        return
    elif args.evaluate == 'test':
        _, _, sampler_test, _, _, test_loader = getDataloaders(splits=('test'), **vars(args))
        trainer.test(test_loader, best_epoch)
        return
    else:
        sampler_train, _, _, train_loader, _, _ = getDataloaders(splits=('train'), **vars(args))
        _, sampler_val, _ , _, val_loader, _ = getDataloaders(splits=('val'), **vars(args))


    # set up logging
    global log_print, f_log
    os.makedirs(args.save, exist_ok=True)
    f_log = open(os.path.join(args.save, 'log.txt'), 'w')

    def log_print(*args):
        print(*args)
        print(*args, file=f_log)
    log_print('args:')
    log_print(args)
    print('model:', file=f_log)
    print(model, file=f_log)
    log_print('# of params:',str(sum([p.numel() for p in model.parameters()])))
    f_log.flush()
    
    torch.save(args, os.path.join(args.save, 'args.pth'))
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_err1'
                '\tval_err1\ttrain_err5\tval_err5']
    best_map = 0.
    best_epoch = 0
    if args.input_norm:
        test_transform = TestTransform(args.image_size, np.array([127,127,127]), 128.0)
    else:
        test_transform = TestTransform(args.image_size, np.array([0,0,0]), 1.0)
    test_dataset = VOCDataset('/SHARE_ST/capp_storage/dataset/VOCdevkit/VOC2007TEST/', transform=test_transform, is_test=True)
    if args.distributed:
        sampler_test = torch.utils.data.DistriubtedSampler(test_dataset) 
        test_loader = torchdata.DataLoader(test_dataset, sampler=sampler_test, batch_size=int(args.batch_size*1.5), shuffle=False, num_workers=4, collate_fn=collate_voc_batch)
    else:
        test_loader =  torchdata.DataLoader(test_dataset, batch_size=int(args.batch_size*1.5), shuffle=False, num_workers=4, collate_fn=collate_voc_batch)
    # breakpoint()
    if args.test_first is True:
        print('\n\nTest First')
        _test_map = trainer.ssd_test(test_loader=test_loader, epoch=0, 
                                     dataset=test_dataset, predictor=predictor, 
                                     eval_path=args.save, config=ssd_config.config, 
                                     conf_threshold=args.conf_threshold, 
                                     test_policy_all_bit=args.test_policy_all_bit,
                                     exclude_1bit=args.exclude_1bit)
    if args.test_ImgNet is True:
        _test_acc = trainer.test(val_loader=val_loader, epoch=0)
    # breakpoint()
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, __ , train_cls_loss, lr, __ = trainer.ssd_train(
            train_loader, epoch, args.test_policy, 
            args.test_policy_all_bit, args.exclude_1bit
            )
        val_loss, reg_loss, cls_loss  = trainer.ssd_valid(val_loader, epoch, 
                                                          args.test_policy, 
                                                          args.test_policy_all_bit,
                                                          args.exclude_1bit)
        if epoch % args.save_iter ==0:
            save_checkpoint({
                'args': args,
                'epoch': epoch,
                'best_epoch': best_epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_map': best_map,
                }, is_best=False, save_dir=args.save)
        if epoch % args.test_epoch == 0 and epoch >0 :
            cur_map = trainer.ssd_test(test_loader, epoch, 
                                       test_dataset, predictor, 
                                       args.save, ssd_config.config, 
                                       args.conf_threshold, test_policy=args.test_policy, 
                                       test_policy_all_bit=args.test_policy_all_bit,
                                       exclude_1bit=args.exclude_1bit)
            is_best = cur_map > best_map
            if is_best:
                best_map = cur_map
                best_epoch = epoch
                mprint(Fore.GREEN + 'Best mAP {}'.format(best_map) + Fore.RESET)
                save_checkpoint({
                'args': args,
                'epoch': epoch,
                'best_epoch': best_epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_map': best_map,
                }, is_best, args.save)
            if not is_best and epoch - best_epoch >= args.patience > 0:
                break
        # accidental deletion
        scores.append(('{}\t{}' + '\t{:.4f}' * 4)
                    .format(epoch, lr, train_loss, val_loss, train_cls_loss, best_map))
        with open(os.path.join(args.save, 'scores.tsv'), 'w') as f:
            print('\n'.join(scores), file=f)            
    print('Best mAP: {:.4f} at epoch {}'.format(best_map, best_epoch))

if __name__ == '__main__':
    main()
