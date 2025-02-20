import os
import glob
import time
import argparse

import config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

model_names = list(map(lambda n: os.path.basename(n)[:-3],
                       glob.glob('models/[A-Za-z]*.py')))

arg_parser = argparse.ArgumentParser(
                description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')
exp_group.add_argument('--resume_path', default='', type=str, metavar='PATH',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--eval', '--evaluate', dest='evaluate', default='',
                       choices=['', 'train', 'val', 'test'],
                       help='eval mode: evaluate model on train/val/test set'
                       ' (default: \'\' i.e. training mode)')
exp_group.add_argument('-f', '--force', dest='force', action='store_true',
                       help='force to overwrite existing save path')
exp_group.add_argument('--print-freq', '-p', default=100, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--no_tensorboard', dest='tensorboard',
                       action='store_false',
                       help='do not use tensorboard_logger for logging')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--local_rank', type=int, default=0)    # kihwan. 220220. for DDP
exp_group.add_argument('--dist', action='store_true')

exp_group.add_argument('--ActQ', choices=['PACT', 'LSQ+', 'DoReFa'], default='PACT')
exp_group.add_argument('--finetune_only', action='store_true')
exp_group.add_argument('--test_policy_all_bit', default='8')
exp_group.add_argument('--exclude_1bit', action='store_true')
exp_group.add_argument('--test_policy', action='store_true')


# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar10',
                        choices=config.datasets.keys(),
                        help='datasets: ' +
                        ' | '.join(config.datasets.keys()) +
                        ' (default: cifar10)')
data_group.add_argument('--no_valid', action='store_false', dest='use_validset',
                        help='not hold out 10 percent of training data as validation')
data_group.add_argument('--data_root', metavar='DIR', default='../data',
                        help='path to dataset (default: data)')
data_group.add_argument('-j', '--workers', dest='num_workers', default=4,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
data_group.add_argument('--normalized', action='store_true',
                        help='normalize the data into zero mean and unit std')

# model arch related
arch_group = arg_parser.add_argument_group('arch',
                                           'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='meta-graph',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
arch_group.add_argument('-d', '--depth', default=56, type=int, metavar='D',
                        help='depth (default=56)')
arch_group.add_argument('--drop-rate', default=0.0, type=float,
                        metavar='DROPRATE', help='dropout rate (default: 0.2)')
arch_group.add_argument('--death-mode', default='none',
                        choices=['none', 'linear', 'uniform'],
                        help='death mode for stochastic depth (default: none)')
arch_group.add_argument('--death-rate', default=0.5, type=float,
                        help='death rate rate (default: 0.5)')
arch_group.add_argument('--growth-rate', default=12, type=int,
                        metavar='GR', help='Growth rate of DenseNet'
                        '(default: 12)')
arch_group.add_argument('--bn-size', default=4, type=int,
                        metavar='B', help='bottle neck ratio of DenseNet'
                        ' (0 means dot\'t use bottle necks) (default: 4)')
arch_group.add_argument('--compression', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')
# used to set the argument when to resume automatically
arch_resume_names = ['arch', 'depth', 'death_mode', 'death_rate', 'death_rate',
                     'growth_rate', 'bn_size', 'compression']
#SSD
arch_group.add_argument('--ssd_model', default=None, type=str, help='Load extra/classication headers from')
arch_group.add_argument('--instanet_chkpt', default=None, type=str, help='Load basenet INSTANAS from')
arch_group.add_argument('--agent_chkpt', default=None, type=str, help='Load basenet agent from')
arch_group.add_argument('--abit', type=int, default=2)
arch_group.add_argument('--action_list', type=int, nargs='*', default=[2,3,4,5,6])
arch_group.add_argument('--input_norm', type=str2bool, default=False)

# training related
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')
optim_group.add_argument('--trainer', default='train', type=str,
                         help='trainer file name without ".py"'
                         ' (default: train)')
optim_group.add_argument('--epochs', default=164, type=int, metavar='N',
                         help='number of total epochs to run (default: 164)')
optim_group.add_argument('--start-epoch', default=1, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('--patience', default=0, type=int, metavar='N',
                         help='patience for early stopping'
                         '(0 means no early stopping)')
optim_group.add_argument('-b', '--batch_size', default=64, type=int,
                         help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.1)')
optim_group.add_argument('--decay_rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--drop_path_prob', default=0.8, type=float, metavar='N',
                         help='drop_path_prob')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')
optim_group.add_argument('--alpha', default=0.8, type=float,
                         help='alpha for ')
optim_group.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                          help='weight decay (default: 1e-4)')
optim_group.add_argument('--path', type=str, default='./del_zero/learned_net')
optim_group.add_argument('--gpu', help='gpu available', default='0,1')
optim_group.add_argument('--retraining', type=str2bool, default=False) # do not use run config
optim_group.add_argument('--full_pretrain',type=str2bool,  default=False)  # (JYJ)
optim_group.add_argument('--resume_best', action='store_true')
optim_group.add_argument('--resume', action='store_true')
optim_group.add_argument('--manual_seed', default=0, type=int)
optim_group.add_argument('--lr_type', default='milestone', type=str, choices=['milestone', 'cosine'])
optim_group.add_argument('--milestones', type=str, default='60,120,150')
optim_group.add_argument('--gamma', type=float, default=0.1)
optim_group.add_argument('--num_classes', type=int, default=20)
optim_group.add_argument('--augmentation', type=str2bool, default=True)
optim_group.add_argument('--test_epoch', default=1, type=int, help='momentum (default=0.9)')
optim_group.add_argument('--image_size', default=300, type=int)
optim_group.add_argument('--save_iter', default=1, type=int)
optim_group.add_argument('--iou_threshold', default=0.45, type=float)
optim_group.add_argument('--center_variance', default=0.1, type=float)
optim_group.add_argument('--size_variance', default=0.2, type=float)
optim_group.add_argument('--conf_threshold', default=0.01, type=float)
optim_group.add_argument('--test_first', action='store_true')
optim_group.add_argument('--freeze_basenet', action='store_true')
optim_group.add_argument('--test_ImgNet', action='store_true')
optim_group.add_argument('--instassd_chkpt', type=str, default=None)
optim_group.add_argument('--from_fp_pretrain', action='store_true')

search_group = arg_parser.add_argument_group('search',
                                            'search setting')
search_group.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
search_group.add_argument('--search_eval_path', default='./search_eval')
search_group.add_argument('--sample_eval_path', default='./sample_eval')
search_group.add_argument('--eval_path', default='./eval')
search_group.add_argument('--agent_lr', type=float, default=1e-3, help='learning rate')
search_group.add_argument('--beta', type=float, default=0.8, help='entropy multiplier')
search_group.add_argument('--tr', type=float, default=0.1, help='trade-off ratio')
search_group.add_argument('--train_net_iter', type=int, default=1)
search_group.add_argument('--train_agent_iter', type=int, default=1)
search_group.add_argument('--finetune_first', action="store_true", default=False)
search_group.add_argument('--test_two_layers', action="store_true", default=False)
search_group.add_argument('--reward_type', type=str, default='prec_only', choices=['prec+bops', 'prec_only', 'f1+bops', 'prec+bops_thre', 'prec+bops_thre_ori', 'prec+bops_thre2'])
search_group.add_argument('--agent_lr_type', default='milestone', type=str, choices=['milestone', 'cosine'])
search_group.add_argument('--agent_milestones', type=str, default='60,120,150')
search_group.add_argument('--agent_gamma', type=float, default=0.1)
search_group.add_argument('--ref_value', type=float, default=24.4317) #8bit 97.72715,  all4bit 24.43179
search_group.add_argument('--extras_wbit', type=int, default=6)
search_group.add_argument('--extras_abit', type=int, default=6)
search_group.add_argument('--head_wbit', type=int, default=6)
search_group.add_argument('--head_abit', type=int, default=6)
search_group.add_argument('--baseline_min', type=float, default=24.2746)
search_group.add_argument('--baseline', type=float, default=79.4508)
search_group.add_argument('--baseline_max', type=float, default=84.5173)
search_group.add_argument('--prec_thre', type=float, default=0.6)
search_group.add_argument('--pos_w', type=int, default=30)
search_group.add_argument('--neg_w', type=int, default=0)
search_group.add_argument('--static', action="store_true", default=False)
search_group.add_argument('--batch_reward', action="store_true", default=False)
arch_group.add_argument('--arch_type', default='V1+SSD',
                        choices=['V1+SSD', 'V1+SSD_Lite','CAPP+SSD','V2+SSD', 'V2+SSD_Lite'],
                        help='arch_type (default: V1+SSD)')

