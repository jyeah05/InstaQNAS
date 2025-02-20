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
from models.instanas import QT_MobileNet_V1_224, QT_MobileNet_V1
from models.instanas_v2lite import QT_MobileNet_V2
from models.mb1_ssd import create_mobilenetv1_ssd_aug, create_mobilenetv1_ssd_predictor
from models.mb2_ssd_lite import create_mobilenetv2_ssd_lite_aug, create_mobilenetv2_ssd_predictor
from models.controller import Policy224_HW, Policy224, Policy300
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
from models.mb2_ssd_config import generate_mb2_ssd_config
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
def get_reward_thre_ori(precs_real,  elasped, baseline, ub, lb, is_batch=False, is_test=False):
    #print("bops")
    #print(elasped)
    if is_batch:
        elapsed = torch.mean(elapsed)
    highest_point = - (lb - ub)*(ub - lb)/4
    sparse_reward = -1 * (elasped - ub) * (elasped - lb) / highest_point
    sparse_reward = torch.clamp(sparse_reward, min=0.)

    if is_test:
        precs_real = precs_real / 100
    precs_real = torch.tensor(precs_real, requires_grad=False).cuda()
    match = (precs_real >= args.prec_thre).data
     
    cost_reward = sparse_reward ** 1.0
    if is_batch:
        cost_reward = cost_reward.cuda()
        if match > 0:
            cost_reward *= args.pos_w
        else:
            cost_reward *= args.neg_w
        cost_reward = cost_reward.unsqueeze(0)
    else:
         cost_reward[match]   *= args.pos_w
         cost_reward[match==0] = args.neg_w
    #print("cost_reward")
    #print(cost_reward)
    #print("match")
    #print(match)
    
    if is_test:
        return cost_reward

    reward = cost_reward.unsqueeze(1)
    reward = reward.unsqueeze(2)
    #print("reward")
    #print(reward)
    #input("enter")
    return reward

def get_reward_thre_2(precs_real,  elasped, baseline, ub, lb, is_batch=False, is_test=False):
    # breakpoint()
    #print("bops")
    #print(elasped)
    if is_batch:
        elasped = torch.mean(elasped)
    highest_point = - (lb - ub)*(ub - lb)/4
    sparse_reward = -1 * (elasped - ub) * (elasped - lb) / highest_point
    sparse_reward = torch.clamp(sparse_reward, min=0.)
    
    sparse_reward_neg = torch.pow(1.2, 
                                  elasped-(lb-math.log(4/((ub-lb)*math.log(1.2)), 1.2))) - 4/((ub-lb)*math.log(1.2))# torch.pow(1.2, torch.log(4/(6*torch.log(1.2)))/torch.log(1.2))
    sparse_reward_neg = torch.clamp(sparse_reward_neg, max=0., min=-1)
    
    sparse_reward_pos = torch.pow(1.2, 
                                  -elasped+(ub+math.log(4/((ub-lb)*math.log(1.2)), 1.2))) - 4/((ub-lb)*math.log(1.2))# torch.pow(1.2, torch.log(4/(6*torch.log(1.2)))/torch.log(1.2))
    sparse_reward_pos = torch.clamp(sparse_reward_pos, max=0., min=-1)

    #match = (pred_idx == targets).dataa
    if is_test:
        precs_real = precs_real / 100
    precs_real = torch.tensor(precs_real, requires_grad=False).cuda()
    if is_batch:
        precs_real = torch.mean(precs_real)
        precs_real = precs_real.unsqueeze(0)
        #print("precs")
    #print(precs_real)
    match = (precs_real >= args.prec_thre).data
    precs_cache = precs_real.clone().data
     
    cost_reward = (sparse_reward + sparse_reward_neg + sparse_reward_pos) ** 1.0
    precs_real *= args.pos_w
    if is_batch:
        cost_reward = cost_reward.cuda()
        if is_test:
            return precs_real
        if match >0:
            precs_real *= cost_reward
    else: 
        precs_real[match] *= cost_reward[match]
        precs_real[match==0] = args.neg_w * ((args.prec_thre+0.1) - precs_cache[match==0])
    #reward[match]   *= args.pos_w
    #reward[match==0] = args.neg_w
    #print("RTRA")
    #print(precs_real)
    #reward = reward * precs_real
    #print("reward after multiplied")
    #print(precs_real)

    reward = precs_real.unsqueeze(1)
    reward = reward.unsqueeze(2)

    #print(reward)
    #input("enter")
    return reward

def get_reward_thre(precs_real,  elasped, baseline, ub, lb, is_batch=False, is_test=False):

    #print("bops")
    #print(elasped)
    if is_batch:
        elasped = torch.mean(elasped)
    highest_point = - (lb - ub)*(ub - lb)/4
    sparse_reward = -1 * (elasped - ub) * (elasped - lb) / highest_point
    sparse_reward = torch.clamp(sparse_reward, min=0.)

    #match = (pred_idx == targets).dataa
    if is_test:
        precs_real = precs_real / 100
    precs_real = torch.tensor(precs_real, requires_grad=False).cuda()
    if is_batch:
        precs_real = torch.mean(precs_real)
        precs_real = precs_real.unsqueeze(0)
        #print("precs")
    #print(precs_real)
    match = (precs_real >= args.prec_thre).data
     
    cost_reward = sparse_reward ** 1.0
    precs_real *= args.pos_w
    if is_batch:
        cost_reward = cost_reward.cuda()
        if is_test:
            return precs_real
        if match >0:
            precs_real *= cost_reward
    else: 
        precs_real[match] *= cost_reward[match]
    #reward[match]   *= args.pos_w
    #reward[match==0] = args.neg_w
    #print("RTRA")
    #print(precs_real)
    #reward = reward * precs_real
    #print("reward after multiplied")
    #print(precs_real)

    reward = precs_real.unsqueeze(1)
    reward = reward.unsqueeze(2)

    #print(reward)
    #input("enter")
    return reward

def get_reward_REINFORCE(prec_real, _bops, is_batch=False, is_test=False):
    prec_real = torch.from_numpy(prec_real).cuda()
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
        base_net = QT_MobileNet_V1(num_classes=1000, full_pretrain=args.full_pretrain, ActQ=args.ActQ, abit=args.abit, action_list=args.action_list) #InstaNet
    elif 'V2' in args.arch_type:
        base_net = QT_MobileNet_V2(num_classes=1001, full_pretrain=args.full_pretrain, ActQ=args.ActQ, abit=args.abit, action_list=args.action_list)
    # base_net = QT_MobileNet_V1_224(num_classes=1001, full_pretrain=args.full_pretrain) #InstaNet
    model = create_mobilenetv1_ssd_aug(base_net, num_classes=args.num_classes, wbit=args.extras_wbit, abit=args.extras_abit, head_wbit=args.head_wbit, head_abit=args.head_abit, config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain) # SSD
    if 'V1+SSD' == args.arch_type:
        model = create_mobilenetv1_ssd_aug(base_net, num_classes=args.num_classes, wbit=args.extras_wbit, abit=args.extras_abit, head_wbit=args.head_wbit, head_abit=args.head_abit, 
                                           config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain, ActQ=args.ActQ) # SSD
    elif 'V2+SSD_Lite' == args.arch_type:
        model = create_mobilenetv2_ssd_lite_aug(base_net, num_classes=args.num_classes, wbit=args.extras_wbit, abit=args.extras_abit, head_wbit=args.head_wbit, head_abit=args.head_abit,
                                                config=args.config_of_data['ssd_config'].config, full_pretrain=args.full_pretrain, ActQ=args.ActQ) # SSD
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
        if args.finetune_only or not args.retraining:
            new_state = remove_prefix(instanet_checkpoint['instanet'], 'module.')
        else:
            new_state = remove_prefix(instanet_checkpoint['state_dict'], 'module.')
        # 
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
    if args.arch_type == "V1+SSD":
        agent = Policy300([1,1,1,1], num_blocks=13, num_of_actions=len(args.action_list))
    elif args.arch_type == "V2+SSD_Lite":
        agent = Policy300([1,1,1,1], num_blocks=17, num_of_actions=len(args.action_list))
    if args.agent_chkpt:
        breakpoint()
        checkpoint = torch.load(args.agent_chkpt)
        agent.load_state_dict(checkpoint['agent'])
        print("Loaded agent!")
    agent = torch.nn.DataParallel(agent).cuda()
    return agent

def train(epoch, instanet, agent, optimizer, searchloader, trainer, predictor, _config, lb, ub, scores):
    # train agent. fix instanet (SSD)
    agent.train()
    instanet.eval()
    prec, rewards, policies, bops, advantages  = [], [], [], [], []
    prec_, rewards_, policies_, bops_ = [], [], [], []
    # set GT boxes
    for idx, data in enumerate(tqdm(searchloader)):
        lr = adjust_learning_rate(optimizer, args.agent_lr, args.weight_decay, epoch, args.epochs, args.agent_lr_type, args.agent_milestones, args.agent_gamma, len(searchloader), idx)
        images, image_sizes, anno_boxes, anno_labels, anno_is_difficult = data
        images = images.cuda(non_blocking=True)
        image_sizes = image_sizes.cpu()
        probs, _ = agent(images)
        policy_real = probs.data.clone()
        max_ops = torch.argmax(policy_real, dim=2)
        policy_real = torch.zeros(policy_real.shape).cuda().scatter(2, max_ops.unsqueeze(2), 1.0)
        probs = probs*args.alpha + (1-probs)*(1-args.alpha) # state s bound
        distr = torch.distributions.Multinomial(1, probs)
        policy = distr.sample()
        # breakpoint()
        with torch.no_grad():
            v_inputs = torch.tensor(images.data.clone()).cuda().detach()
            #if args.batch_reward:
            #    prec_real, rec_real, bops_real = trainer.ssd_validate_batch_prec_bops(v_inputs, anno_boxes=anno_boxes, anno_labels=anno_labels, image_sizes=image_sizes, policy=policy_real, predictor=predictor, eval_path=args.search_eval_path, config=_config, conf_threshold=args.conf_threshold)

             #   prec_sample, rec_sample, bops_sample = trainer.ssd_validate_batch_prec_bops(v_inputs, anno_boxes=anno_boxes, anno_labels=anno_labels,  image_sizes=image_sizes, policy=policy, predictor=predictor, eval_path=args.sample_eval_path, config=_config, conf_threshold=args.conf_threshold)
            #else:
            prec_real, rec_real, bops_real, _ = trainer.ssd_validate_batch_single_prec_bops(v_inputs, anno_boxes=anno_boxes, anno_labels=anno_labels, image_sizes=image_sizes, policy=policy_real, predictor=predictor, eval_path=args.search_eval_path, config=_config, conf_threshold=args.conf_threshold)

            prec_sample, rec_sample, bops_sample, _ = trainer.ssd_validate_batch_single_prec_bops(v_inputs, anno_boxes=anno_boxes, anno_labels=anno_labels,  image_sizes=image_sizes, policy=policy, predictor=predictor, eval_path=args.sample_eval_path, config=_config, conf_threshold=args.conf_threshold)
        if args.reward_type == 'prec+bops':
            reward_real = get_reward_REINFORCE(prec_real, bops_real, is_batch=args.batch_reward)
            reward_sample  = get_reward_REINFORCE(prec_sample, bops_sample, is_batch=args.batch_reward)
        elif args.reward_type == 'prec_only':
            reward_real = get_reward_prec(prec_real)
            reward_sample = get_reward_prec(prec_sample)
        elif args.reward_type == 'prec+bops_thre':
            reward_real = get_reward_thre(prec_real,  bops_real, args.baseline, ub, lb, is_batch=args.batch_reward)
            reward_sample = get_reward_thre(prec_sample,  bops_sample, args.baseline, ub, lb, is_batch=args.batch_reward)
        elif args.reward_type == 'prec+bops_thre2':
            reward_real = get_reward_thre_2(prec_real,  bops_real, args.baseline, ub, lb, is_batch=args.batch_reward)
            reward_sample = get_reward_thre_2(prec_sample,  bops_sample, args.baseline, ub, lb, is_batch=args.batch_reward)
        elif args.reward_type == 'prec+bops_thre_ori':
            reward_real = get_reward_thre_ori(prec_real,  bops_real, args.baseline, ub, lb, is_batch=args.batch_reward)
            reward_sample = get_reward_thre_ori(prec_sample,  bops_sample, args.baseline, ub, lb, is_batch=args.batch_reward)
        else:
            reward_real  = get_reward_MAP(prec_real)
            reward_sample  = get_reward_MAP(prec_sample)
        
        advantage = reward_sample - reward_real
        advantage = advantage.cuda(non_blocking=True).expand_as(policy)
        loss = -distr.log_prob(policy)
        loss = loss.unsqueeze(2).repeat(1,1,len(args.action_list))
        #print(loss)
        loss = loss * advantage

        #print("loss")
        #print(loss)

        #print(loss)
        #input("enter")
        loss = loss.sum() 

        probs = probs.clamp(1e-15, 1-1e-15)
        entropy_loss = -probs*torch.log(probs)
        entropy_loss = args.beta*entropy_loss.sum()

        loss = (loss - entropy_loss)/v_inputs.size(0)
        loss = loss /v_inputs.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        advantages.append(np.mean(advantage.cpu().numpy()))
        
        ## actual policy
        prec_.append(np.mean(prec_real))
        rewards_.append(reward_real.cpu())
        policies_.append(policy_real.data.cpu())
        bops_.append(np.mean(bops_real.data.cpu().numpy()))
        ## sample policy
        prec.append(np.mean(prec_sample))
        rewards.append(reward_sample.cpu())
        policies.append(policy.data.cpu())
        bops.append(np.mean(bops_sample.data.cpu().numpy()))

    #print(prec_)
    #print(np.mean(prec_))
    rewards, policy_set_sample = utils.performance_stats(policies, rewards)
    log_str = 'Prob E: %d | P: %.3f | R: %.4f | #: %d | B (G): %.4f ' % (
        epoch, np.mean(prec), rewards, len(policy_set_sample), np.mean(bops))
    print(log_str)
    scores.append(('{}\t{}\t{}\t{:4f}\t{:3f}\t{:d}\t{:4f}')
                    .format(str(datetime.now()), epoch, 'Prob', np.mean(prec), rewards, len(policy_set_sample), np.mean(bops)))

    rewards_, policy_set = utils.performance_stats(policies_, rewards_)
    log_str = 'Real E: %d | P: %.3f | R: %.4f | #: %d | B (G): %.4f | ADV : %.4f' % (
        epoch, np.mean(prec_), rewards_, len(policy_set), np.mean(bops_), np.mean(advantages))
    print(log_str)
    scores.append(('{}\t{}\t{}\t{:4f}\t{:3f}\t{:d}\t{:4f}\t{:4f}')
                    .format(str(datetime.now()), epoch, 'Real', np.mean(prec_), rewards_, len(policy_set), np.mean(bops_), np.mean(advantages)))
    with open(os.path.join(args.cv_dir, 'scores.tsv'), 'w') as f:
        print('\n'.join(scores), file=f)


def train_net(epoch, instanet, agent, trainer,  optimizer_net, trainloader_ft, scores):
    agent.eval()
    instanet.train() # Note: instanet now refers to SSD
    # set GT boxes
    losses, regression_losses, classification_losses, lr = trainer.ssd_train(trainloader_ft, epoch, args.alpha)
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
    map_real,  bops_real, policies_set =trainer.ssd_test(testloader, epoch, test_dataset, predictor, args.cv_dir, config=ssd_config, conf_threshold=args.conf_threshold, test_policy=False, return_policy=True)
    if args.reward_type == 'prec+bops':
        reward_real = get_reward_REINFORCE(np.array(map_real), bops_real, is_batch=True,  is_test=True)
        reward_real = reward_real.cpu().numpy()
    elif args.reward_type == 'prec_only':
        reward_real = 0.01 * map_real
    elif args.reward_type == 'prec+bops_thre':
        reward_real = get_reward_thre(np.array(map_real), bops_real, args.baseline, ub, lb, is_batch=True, is_test=True)
    elif args.reward_type == 'prec+bops_thre2':
        reward_real = get_reward_thre_2(np.array(map_real), bops_real, args.baseline, ub, lb, is_batch=True, is_test=True)
    elif args.reward_type == 'prec+bops_thre_ori':
        reward_real = get_reward_thre_ori(np.array(map_real), bops_real, args.baseline, ub, lb, is_batch=True, is_test=True)
    else:
        reward_real = 0
    f1 = open(args.cv_dir+'/policies.log', 'w')
    f1.write(str(policies_set))
    f1.close()
    log_str = 'TS - mAP: %.3f | R: %.4f | #: %d | B (G): %.4f ' % (map_real, reward_real, len(policies_set), bops_real)
    print(log_str)
    #reward_real = np.mean(reward_real.squeeze(1).squeeze(1).cpu().numpy())
    bops_real = bops_real.cpu().numpy()
    scores.append(('{}\t{}\t{}\t{:.3f}\t{:.4f}\t{:d}\t{:.4f}')
                    .format(str(datetime.now()), epoch, 'TS', map_real, reward_real.cpu().numpy()[0],
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
    dt['ssd_config'] = ssd_config
    args.config_of_data = dt
    os.makedirs(args.cv_dir, exist_ok=True)
    if args.resume is True:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            # set args based on checkpoint
            if args.retraining is True:
                args.start_epoch = 0
            else:
                args.start_epoch = checkpoint['epoch'] + 1
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
    optimizer = optim.Adam(agent.parameters(), lr=args.agent_lr, weight_decay= args.weight_decay)
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
    search_transform = TestTransform(300, np.array([0,0,0]), 1.0)
    search_dataset = VOCDataset('/data/dataset/VOCdevkit/voc_07_12_train/', transform=search_transform)
    search_loader =  torchdata.DataLoader(search_dataset, batch_size=args.batch_size*4, shuffle=True, num_workers=4, collate_fn=collate_voc_batch)

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
    test_loader =  torchdata.DataLoader(test_dataset, batch_size=int(args.batch_size*1.5), shuffle=False, num_workers=8, collate_fn=collate_voc_batch)
    trainer.set_true_labels(test_dataset)
    baseline_max = args.baseline_max
    baseline = args.baseline
    baseline_min = args.baseline_min
    if args.test_first:
        best_map = test(0, instanet, agent, test_dataset, test_loader, trainer, predictor, ssd_config.config, args.baseline_min, args.baseline_max, scores)
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

        for i in range(args.train_agent_iter):
            if args.finetune_first and epoch==0:
                break
            print(" [*] Train agent ... ")
            train(epoch, instanet, agent, optimizer, search_loader, trainer, predictor, ssd_config.config, lb, ub, scores)
        for i in range(args.train_net_iter):
            print(" [*] Fine-tuning ... ")
            train_net(epoch, instanet, agent,trainer,  optimizer_net, train_loader, scores)
            best_map = test(epoch, instanet, agent, test_dataset, test_loader, trainer, predictor, ssd_config.config, lb, ub, scores)

#############################################################################################################################################

def fine_tune_net(epoch, instanet, agent, trainer,  optimizer_net, trainloader_ft, scores):
    agent.eval()
    instanet.train() # Note: instanet now refers to SSD
    # set GT boxes
    # breakpoint()
    losses, regression_losses, classification_losses, lr = trainer.ssd_fine_tune(trainloader_ft, epoch, args.alpha)
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

def fine_tune_val_test():
    dt = {'num_classes':args.num_classes, 'augmentation':args.augmentation}
    if 'V1' in args.arch_type or 'CAPP' in args.arch_type:
        ssd_config = generate_mb1_ssd_config(args.image_size, args.iou_threshold, args.center_variance, args.size_variance) #prior, specs, thresholds
    elif 'V2' in args.arch_type:
        ssd_config = generate_mb2_ssd_config(args.image_size, args.iou_threshold, args.center_variance, args.size_variance) #prior, specs, thresholds
    else:
        raise NotImplementedError("Not existing arch_type")
    dt['ssd_config'] = ssd_config
    args.config_of_data = dt
    os.makedirs(args.cv_dir, exist_ok=True)
    if args.resume is True:
        if os.path.isfile(args.instassd_chkpt):
            print("=> loading checkpoint '{}'".format(args.instassd_chkpt))
            # checkpoint = torch.load(args.instassd_chkpt)
            # set args based on checkpoint
            # breakpoint()
            if args.retraining is True:
                args.start_epoch = 0
            else:
                args.start_epoch = torch.load(args.instassd_chkpt)['epoch'] + 1
            instanet = getModel(**vars(args))
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, args.start_epoch - 1))
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


    scores = ['timestamp\tepoch\tmode\t\mAP\treward\tnumOfPolicies\tBOPs (G)'] #2
    best_map = 0.
    best_epoch = 0
    test_transform = TestTransform(args.image_size, np.array([127,127,127]), 128.0)
    # test_transform = TestTransform(224, np.array([127,127,127]), 128.0)
    test_dataset = VOCDataset('/data/dataset/VOCdevkit/VOC2007TEST/', transform=test_transform, is_test=True)
    test_loader =  torchdata.DataLoader(test_dataset, batch_size=int(args.batch_size*1.5), shuffle=False, num_workers=8, collate_fn=collate_voc_batch)
    
    trainer.set_true_labels(test_dataset)
    if args.test_first:
        map_real,  bops_real, policies_set =trainer.ssd_test(test_loader, 0, test_dataset, predictor, 
                                                             args.cv_dir, config=ssd_config.config, 
                                                             conf_threshold=args.conf_threshold, 
                                                             test_policy=args.test_policy, return_policy=True, atype=args.arch_type
                                                            )
        f1 = open(args.cv_dir+'/policies.log', 'w')
        f1.write(str(policies_set))
        f1.close()
        log_str = 'TS - mAP: %.3f | #: %d | B (G): %.4f ' % (map_real, len(policies_set), bops_real)
        print(log_str)
        #reward_real = np.mean(reward_real.squeeze(1).squeeze(1).cpu().numpy())
        bops_real = bops_real.cpu().numpy()
        scores.append(('{}\t{}\t{}\t{:.3f}\t{:d}\t{:.4f}')
                        .format(str(datetime.now()), 0, 'TS', map_real,
                                len(policies_set), bops_real))


        # with open(os.path.join(args.cv_dir, 'scores.tsv'), 'w') as f:
        #     print('\n'.join(scores), file=f)
        # best_map = test(0, instanet, agent, test_dataset, test_loader, trainer, predictor, ssd_config.config, args.baseline_min, args.baseline_max, scores)
    for epoch in range(args.start_epoch, args.epochs+1):
        print(" [*] EXP: {}".format(args.cv_dir))
        
        for i in range(args.train_net_iter):
            print(" [*] Fine-tuning ... ")
            fine_tune_net(epoch, instanet, agent, trainer, optimizer_net, train_loader, scores)
            map_real,  bops_real, policies_set =trainer.ssd_test(test_loader, epoch, test_dataset, predictor, 
                                                             args.cv_dir, config=ssd_config.config, 
                                                             conf_threshold=args.conf_threshold, 
                                                             test_policy=args.test_policy, return_policy=True,
                                                            )
            f1 = open(args.cv_dir+'/policies.log', 'w')
            f1.write(str(policies_set))
            f1.close()
            log_str = 'TS - mAP: %.3f | #: %d | B (G): %.4f ' % (map_real, len(policies_set), bops_real)
            print(log_str)
            bops_real = bops_real.cpu().numpy()
            # breakpoint()
            scores.append(('{}\t{}\t{}\t{:.3f}\t{:d}\t{:.4f}')
                            .format(str(datetime.now()), epoch, 'TS', map_real,
                                    len(policies_set), bops_real))


            with open(os.path.join(args.cv_dir, 'scores.tsv'), 'w') as f:
                print('\n'.join(scores), file=f)

            agent_state_dict = agent.module.state_dict()
            net_state_dict = instanet.module.state_dict()

            state = {
                'instanet': net_state_dict,
                'agent': agent_state_dict,
                'epoch': epoch,
                'maps': map_real,
                'bops': bops_real,
                }
            if map_real > best_map:
                best_map = map_real
                save_checkpoint(state, True, args.cv_dir, 'ckpt_E_%d_A_%.3f_#_%d.pth.tar' %
                    (epoch, map_real, len(policies_set)))
            save_checkpoint(state, False, args.cv_dir, 'latest.pth.tar')

def main():
    if args.finetune_only:
        fine_tune_val_test()
        exit()
        
    train_val_test()
    

if __name__ == "__main__":
    
    main()
