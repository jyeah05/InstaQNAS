import sys
import time
import os
import shutil
import torch
import numpy as np
import math
from colorama import Fore
import torch.nn.functional as F
def compute_average_precision(precision, recall):
    """
    It computes average precision based on the definition of Pascal Competition. It computes the under curve area
    of precision and recall. Recall follows the normal definition. Precision is a variant.
    pascal_precision[i] = typical_precision[i:].max()
    """
    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
    return areas.sum()

def performance_stats(policies, rewards):

    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)
    reward = rewards.mean()
    policy_set = [np.reshape(p.cpu().numpy().astype(
        np.int).astype(np.str), (-1)) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return reward, policy_set

def compute_voc2007_average_precision(precision, recall):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap

def count_conv_flop(layer, x):
    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])
    delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * \
                out_h * out_w / layer.groups
    return delta_ops

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.autograd.Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
    return x

def collate_voc_batch(batch):
    img, img_size, boxes, labels, is_difficult = zip(*batch)  # transposed
    return torch.stack(img,0), torch.stack(img_size,0), boxes, labels, is_difficult # img_size is tuple whose length is 1

def collate_voc_batch_with_id(batch):
    img_id, img, img_size, boxes, labels, is_difficult = zip(*batch)  # transposed
    return img_id, torch.stack(img,0), torch.stack(img_size,0), boxes, labels, is_difficult # img_size is tuple whose length is 1
    
def create_save_folder(save_path, force=False, ignore_patterns=[]):
    if os.path.exists(save_path):
        print(Fore.RED + save_path + Fore.RESET + ' already exists!')
    print('create folder: ' + Fore.GREEN + save_path + Fore.RESET)


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs, lr_type, milestones, gamma, nBatch, batch):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    milestone = [int(val) for val in milestones.split(',')]
    if lr_type == 'milestone':
       scale = 1
       for x in milestone:
           if epoch +1 > x:
               scale *= gamma
       lr = scale * lr_init
    elif lr_type == 'cosine':
       T_total = num_epochs * nBatch
       T_cur = epoch * nBatch  + batch
       lr = 0.5 * lr_init * (1+math.cos(math.pi * T_cur / T_total))
    else:
        lr = lr_init
        if epoch >= num_epochs * 0.75:
            lr *= decay_rate**2
        elif epoch >= num_epochs * 0.5:
            lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
