#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import time
from datetime import datetime
import math
## prequistie
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys 
import torch.optim as optim
from torch.distributions import Bernoulli
## models
from models.controller import Policy224_HW, Policy224
from models.instanas import QT_MobileNet_V1_224, QT_MobileNet_V2_224, QT_MobileNet_CAPP_224
from models.mb1_ssd import create_mobilenetv1_ssd_aug, create_mobilenetv1_ssd_predictor
from models.mb1_ssd_lite import create_mobilenetv1_ssd_lite_aug
from models.mb1_capp_ssd import create_mobilenetv1_capp_ssd_aug
from models.mb2_ssd_lite import create_mobilenetv2_ssd_lite_aug, create_mobilenetv2_ssd_predictor

## utils
import utils
from colorama import Fore
from importlib import import_module
from dataloader import getDataloaders
from tqdm import tqdm
from utils import AverageMeter, adjust_learning_rate, save_checkpoint, get_optimizer, create_save_folder, print_model_parm_nums
from args import arg_parser, arch_resume_names
from models.multibox_loss import MultiboxLoss
from models.mb1_ssd_config import generate_mb1_ssd_config
from data_provider.voc_dataset import VOCDataset
from data_provider.data_preprocessing import TestTransform
from utils import collate_voc_batch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore")


global args
args = arg_parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)

torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
np.random.seed(args.manual_seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)  # kihwan. 220220. for DDP
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
def remove_prefix(state_dict, prefix):
    print('remove prefix %s', prefix )
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def add_prefix(state_dict, prefix='module.'):
    new_dict = {}
    for k,v in state_dict.items():
        new_dict[prefix+k] = v
    return new_dict
def get_reward_thre_ori(precs_real,  elasped, baseline, ub, lb, is_test=False):

    #print("bops")
    #print(elasped)
    highest_point = - (lb - ub)*(ub - lb)/4
    sparse_reward = -1 * (elasped - ub) * (elasped - lb) / highest_point
    sparse_reward = torch.clamp(sparse_reward, min=0.)

    #match = (pred_idx == targets).dataa
    if is_test:
        precs_real = precs_real / 100
    precs_real = torch.tensor(precs_real, requires_grad=False).cuda()
    match = (precs_real >= args.prec_thre).data
     
    cost_reward = sparse_reward ** 1.0
    if is_test:
        #print("highest_point")
        #print(highest_point)
        #print("BOPS")
        #print(elasped)
        #print("precs_real")
        #print(precs_real)
        #print("match")
        #print(match)
        cost_reward = cost_reward.cuda()
        #print("cost_reward")
        #print(cost_reward)
    #print("reward_cost")
    #print(cost_reward)
    
    cost_reward[match]   *= args.pos_w
    cost_reward[match==0] = args.neg_w
    #print("RTRA")
    #print(precs_real)
    #reward = reward * precs_real
    #print("reward after multiplied")
    #print(reward)
    if is_test:
        #print("precs_real")
        #print(precs_real)
        #input("enter")
        return cost_reward

    reward = cost_reward.unsqueeze(1)
    reward = reward.unsqueeze(2)

    #print(reward)
    #input("enter")
    return reward



def get_reward_thre(precs_real,  elasped, baseline, ub, lb, is_test=False):

    #print("bops")
    #print(elasped)
    highest_point = - (lb - ub)*(ub - lb)/4
    sparse_reward = -1 * (elasped - ub) * (elasped - lb) / highest_point
    sparse_reward = torch.clamp(sparse_reward, min=0.)

    #match = (pred_idx == targets).dataa
    if is_test:
        precs_real = precs_real / 100
    precs_real = torch.tensor(precs_real, requires_grad=False).cuda()
    #print("precs")
    #print(precs_real)
    match = (precs_real >= args.prec_thre).data
    #print(match)
     
    cost_reward = sparse_reward ** 1.0
    if is_test:
        #print("highest_point")
        #print(highest_point)
        #print("BOPS")
        #print(elasped)
        #print("precs_real")
        #print(precs_real)
        #print("match")
        #print(match)
        cost_reward = cost_reward.cuda()
        #print("cost_reward")
        #print(cost_reward)
    #print("reward_cost")
    #print(cost_reward)
    
    precs_real *= args.pos_w
    precs_real[match] *= cost_reward[match]
    #reward[match]   *= args.pos_w
    #reward[match==0] = args.neg_w
    #print("RTRA")
    #print(precs_real)
    #reward = reward * precs_real
    #print("reward after multiplied")
    #print(reward)
    if is_test:
        #print("precs_real")
        #print(precs_real)
        #input("enter")
        return precs_real

    reward = precs_real.unsqueeze(1)
    reward = reward.unsqueeze(2)

    #print(reward)
    #input("enter")
    return reward

def get_reward_REINFORCE(prec_real, _bops, is_test=False):
    prec_real = torch.from_numpy(prec_real).cuda()
    #prec_real = torch.from_numpy(prec_real)
    #_bops = torch.from_numpy(_bops)
    #print("prec_real")
    #print(prec_real)
    #print("_bops")
    #print(_bops)
    #print("prec_real.shape")
    #print(prec_real.shape)
    #print("_bops.shape")
    #print(_bops.shape)
    reward = prec_real * ((args.ref_value / _bops) ** args.tr)
    #print("reward.shape")
    #print(reward.shape)
    #print(reward.shape)
    #input("enter")
    if is_test:
        return reward/100
    reward = reward.unsqueeze(1)
    reward = reward.unsqueeze(2)
    return reward

def get_reward_prec(prec):
    prec = torch.from_numpy(prec)
    #print("prec")
    #print(prec)
    #print("prec.shape")
    #print(prec.shape)
    reward = prec.unsqueeze(1)
    reward = reward.unsqueeze(2)
    #print("reward")
    #print(reward)
    #print("reward.shape")
    #print(reward.shape)
    return reward

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
    if args.ssd_model:
        model.init_from_pretrained_ssd(args.ssd_model)
    if args.instanet_chkpt:
        instanet_checkpoint = torch.load(args.instanet_chkpt)
        new_state = remove_prefix(instanet_checkpoint['state_dict'], 'module.')
        new_state.update(new_state)
        model.base_net.load_state_dict(new_state)
        print("Loaded basenet from pretrained model!")

    if args.instassd_chkpt:
        instanet_checkpoint = torch.load(args.instassd_chkpt)
        #new_state = remove_prefix(instanet_checkpoint['state_dict'], 'module.')
        new_state = remove_prefix(instanet_checkpoint['instanet'], 'module.')
        new_state.update(new_state)
        try:
            model.load_state_dict(new_state)
            print("Loaded ssd from pretrained model!")
        except:
            print("could not load weight")
    #args.start_epoch = 0
    if args.dist:
        gpu_id = init_dist()
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), [gpu_id], gpu_id, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()
    print_model_parm_nums(model)
    return model
def get_agent(**kargs):
    agent = Policy224_HW([1,1,1,1], num_blocks=13, num_of_actions=4)
    if args.agent_chkpt:
        checkpoint = torch.load(args.agent_chkpt)
        agent.load_state_dict(checkpoint['agent'])
        print("Loaded agent!")
    #if args.agent_chkpt:
    #    gpu_id =  init_dist()
    #    agent = torch.nn.parallel.DistributedDataParallel(agent.cuda(), [gpu_id], gpu_id, find_unused_parameters=True)
    #else:
    agent = torch.nn.DataParallel(agent).cuda()
    return agent

def train_net(epoch, instanet, agent, trainer,  optimizer_net, trainloader_ft, scores):
    agent.eval()
    instanet.train() # Note: instanet now refers to SSD
    # set GT boxes
    #losses, regression_losses, classification_losses, lr = trainer.ssd_train(trainloader_ft, epoch, args.alpha, args.test_two_layers)
    losses, regression_losses, classification_losses, lr = trainer.ssd_finetune(trainloader_ft, epoch, args.alpha, args.test_two_layers)
    print('FT E: {:3d} | Train loss {loss:.4f} | '
              'Ref_loss@1 {reg_los:.4f} | '
              ' Class_loss@5{cls_los:.4f} | '
              ' LR {lr:.4f}  '
              .format(epoch, loss=losses, reg_los=regression_losses, cls_los=classification_losses, lr=lr))
    scores.append('FT E: {:3d} | Train loss {loss:.4f} | '
              'Ref_loss@1 {reg_los:.4f} | '
              ' Class_loss@5{cls_los:.4f} | '
              ' LR {lr:.4f}  '
              .format(epoch, loss=losses, reg_los=regression_losses, cls_los=classification_losses, lr=lr))
    with open(os.path.join(args.cv_dir, 'scores.tsv'), 'w') as f:
        print('\n'.join(scores), file=f)
    return lr


def test(epoch, instanet, agent, test_dataset, testloader, trainer, predictor, ssd_config, lb, ub, scores, best_map=0):
    agent.eval()
    instanet.eval() # SSD 
    map_real,  bops_real, policies_set =trainer.ssd_test(testloader, epoch, test_dataset, predictor, args.eval_path, config=ssd_config, conf_threshold=args.conf_threshold, test_policy=False, return_policy=True, test_two_layers=args.test_two_layers)
    if args.reward_type == 'prec+bops':
        reward_real = get_reward_REINFORCE(np.array(map_real), bops_real, is_test=True)
        #reward_real = np.mean(reward_real.squeeze(1).squeeze(1).cpu().numpy())
        reward_real = reward_real.cpu().numpy()
    elif args.reward_type == 'prec_only':
        reward_real = 0.01 * map_real
    elif args.reward_type == 'prec+bops_thre':
        reward_real = get_reward_thre(np.array(map_real), bops_real, args.baseline, ub, lb, is_test=True)
    elif args.reward_type == 'prec+bops_thre_ori':
        reward_real = get_reward_thre_ori(np.array(map_real), bops_real, args.baseline, ub, lb, is_test=True)
    else:
        reward_real = 0
    f1 = open(args.cv_dir+'/policies.log', 'w')
    f1.write(str(policies_set))
    f1.close()
    log_str = 'TS - mAP: %.3f | R: %.4f | #: %d | B (G): %.4f ' % (map_real, reward_real, len(policies_set), bops_real)
    #log_str = 'TS - mAP: %.3f | #: %d | B (G): %.4f ' % (map_real, len(policies_set), bops_real)
    print(log_str)
    #reward_real = np.mean(reward_real.squeeze(1).squeeze(1).cpu().numpy())
    scores.append(('{}\t{}\t{}\t{:.3f}\t{:.4f}\t{:d}\t{:.4f}')
                    .format(str(datetime.now()), epoch, 'TS', map_real, reward_real,
                            len(policies_set), bops_real))


    with open(os.path.join(args.cv_dir, 'scores.tsv'), 'w') as f:
        print('\n'.join(scores), file=f)

    agent_state_dict = agent.module.state_dict()
    net_state_dict = instanet.module.state_dict()

    state = {
        'instanet': net_state_dict,
        'agent': agent_state_dict,
        'epoch': epoch,
        'reward': reward_real,
        'maps': map_real,
        'bops': bops_real,
        }
    if True:
        is_best = False
        save_checkpoint(state, is_best, args.cv_dir, 'ckpt_E_%d_A_%.3f_R_%.2E_#_%d.pth.tar' %
               (epoch, map_real, reward_real, len(policies_set)))
    save_checkpoint(state, False, args.cv_dir, 'latest.pth.tar')

    return best_map

#--------------------------------------------------------------------------------------------------------#
def train_val_test():
    dt = {'num_classes':args.num_classes, 'augmentation':args.augmentation}
    if 'V1' in args.arch_type or 'CAPP' in args.arch_type:
        ssd_config = generate_mb1_ssd_config(args.image_size, args.iou_threshold, args.center_variance, args.size_variance) #prior, specs, thresholds
    elif 'V2' in args.arch_type:
        ssd_config = generate_mb2_ssd_config(args.image_size, args.iou_threshold, args.center_variance, args.size_variance) #prior, specs, thresholds
    else:
        raise NotImplementedError("Not existing arch_type")
    dt['ssd_config']= ssd_config
    args.config_of_data = dt
    os.makedirs(args.cv_dir, exist_ok=True)
    if args.resume is True:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            #old_args = checkpoint['args']
            #print('Old args:')
            #print(old_args)
            # set args based on checkpoint
            if args.retraining is True:
                args.start_epoch = 0
            else:
                args.start_epoch = checkpoint['epoch'] + 1
            #for name in arch_resume_names:
            #    if name in vars(args) and name in vars(old_args):
            #        setattr(args, name, getattr(old_args, name))
            instanet = getModel(**vars(args))
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
        print("=> creating model '{}'".format(args.arch))
        instanet = getModel(**vars(args))
    if args.full_pretrain is True:
        instanet.module.freeze_PACT_param()
    else:
        instanet.module.train_PACT_param()
    if args.freeze_basenet is True:
        instanet.module.freeze_base_param()
    agent = get_agent(**vars(args))
    if 'V1' in args.arch_type or 'CAPP' in args.arch_type:
        predictor = create_mobilenetv1_ssd_predictor(instanet, ssd_config.config)
    elif 'V2' in args.arch_type:
        predictor = create_mobilenetv2_ssd_predictor(instanet, ssd_config.config)
    else:
        raise NotImplementedError("Not existing arch_type")
    criterion = MultiboxLoss(ssd_config.config['priors'], iou_threshold=args.iou_threshold, neg_pos_ratio=3, center_variance=args.center_variance, size_variance=args.size_variance, conf_threshold=0.1)
    optimizer_net = get_optimizer(instanet, args)
    # set random seed
    torch.manual_seed(args.seed)
    Trainer = import_module(args.trainer).Trainer
    trainer = Trainer(instanet, criterion, optimizer_net, args, agent)
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
        train_dataset, __, _, train_loader, __ , _ = getDataloaders(splits=('train'), **vars(args))

    #if args.resume:
    #    checkpoint = torch.load(args.resume_path)
    #    agent.load_state_dict(checkpoint['agent'])
    #    print(' [*] Loaded agent from', args.resume)

    scores = ['timestamp\tepoch\tmode\t\mAP\treward\tnumOfPolicies\BOPs (G)'] #2
    best_map = 0.
    best_reward = 0.
    best_epoch = 0
    test_transform = TestTransform(300, np.array([0,0,0]), 1.0)
    test_dataset = VOCDataset('/data/dataset/VOCdevkit/VOC2007TEST/', transform=test_transform, is_test=True)
    test_loader =  torchdata.DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=8, collate_fn=collate_voc_batch)
    trainer.set_true_labels(test_dataset)
    baseline_max = args.baseline_max
    baseline = args.baseline
    baseline_min = args.baseline_min
    if args.test_first:
        #best_map = test(0, instanet, agent, test_dataset, test_loader, trainer, predictor, ssd_config.config, args.baseline_min, args.baseline_max, scores)
        best_map = trainer.ssd_test(test_loader, epoch=0, dataset=test_dataset, predictor=predictor, eval_path='./test', config=ssd_config.config, conf_threshold=args.conf_threshold, test_policy=False, return_policy=True)
        return
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.static:
            epoch_ratio = args.static_ep / args.epochs
        else:
            epoch_ratio = epoch / args.epochs
        
        ub_sp = baseline_max * 0.7
        ub_ep = baseline * 0.5
        #ub_ep = baseline * 0.7
        lb_sp = baseline * 0.5 
        lb_ep = 0
        cur_windows_size = ((ub_sp-lb_sp) - (ub_ep-lb_sp)) * epoch_ratio

        if args.static:
            ub = baseline_max 
            lb = baseline_min
        else:
            ub = (ub_sp) - (ub_sp-ub_ep) * epoch_ratio
            lb = baseline_min
        print(' [*] Current Baseline: {:4f}, MIN: {:4f}, UB: {:4f}, LB: {:4f}'.format(baseline, baseline_min, ub, lb))
        print(" [*] EXP: {}".format(args.cv_dir))

        for i in range(args.train_net_iter):
            print(" [*] Fine-tuning ... ")
            train_net(epoch, instanet, agent,trainer,  optimizer_net, train_loader, scores)
            best_map = test(epoch, instanet, agent, test_dataset, test_loader, trainer, predictor, ssd_config.config, lb, ub, scores)

def main():
    train_val_test()
    

if __name__ == "__main__":
    main()
