from random import shuffle
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as torchdata
from data_provider.transforms import *
from data_provider.data_preprocessing import TrainAugmentation_qat1, TestTransform
from data_provider.voc_dataset import VOCDataset
from models.ssd import MatchPrior
import numpy as np
from utils import collate_voc_batch

def cutout(mask_size, p, cutout_inside=False, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout

def getDataloaders(data, config_of_data, splits=['train', 'val', 'test'],
                   aug=True, use_validset=True, data_root='data', batch_size=64, normalized=True,
                   num_workers=3, **kwargs):
    train_loader, val_loader, test_loader = None, None, None
    sampler_train, sampler_test = None, None
    if data.find('cifar10') >= 0:
        print('loading ' + data)
        print(config_of_data)
        if data.find('cifar100') >= 0:
            d_func = dset.CIFAR100
        else:
            d_func = dset.CIFAR10
        normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        if config_of_data['augmentation']:
            print('with data augmentation')
            aug_trans = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                cutout(16, 1, False)
            ]
        else:
            aug_trans = []
        common_trans = [transforms.ToTensor()]
        common_trans.append(normalize)

        train_compose = transforms.Compose(aug_trans + common_trans)
        test_compose = transforms.Compose(common_trans)

        if 'train' in splits:
            train_set = d_func(data_root, train=True, transform=train_compose, download=True)
            if kwargs['dist'] is True:
                sampler_train = torchdata.DistributedSampler(train_set)
                train_loader = torchdata.DataLoader(train_set, sampler=sampler_train, batch_size=batch_size, shuffle=False, num_workers=16)
            else:
                train_loader = torchdata.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        if 'val' in splits or 'test' in splits:
            test_set = d_func(data_root, train=False, transform=test_compose)
            if kwargs['dist'] is True:
                sampler_test = torchdata.DistributedSampler(test_set)
                test_loader = torchdata.DataLoader(test_set, sampler=sampler_test, batch_size=batch_size, shuffle=False, num_workers=16)
            else:
                test_loader = torchdata.DataLoader(test_set, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True)
            val_loader = test_loader

    elif data.find('tiny') >= 0:
        print('loading ' + data)
        print(config_of_data)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if config_of_data['augmentation']:
            print('with data augmentation')
            aug_trans = [
                transforms.RandomHorizontalFlip(),
            ]
        else:
            aug_trans = []
        common_trans = [transforms.ToTensor()]
        common_trans.append(normalize)
        train_compose = transforms.Compose(aug_trans + common_trans)
        test_compose = transforms.Compose(common_trans)

        train_set = dset.ImageFolder('/home/vslab2018/data/tiny-imagenet-200/train', transform=train_compose)
        train_loader = torchdata.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        val_set = dset.ImageFolder('/home/vslab2018/data/tiny-imagenet-200/val', transform=test_compose)
        val_loader = torchdata.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        test_set = dset.ImageFolder('/home/vslab2018/data/tiny-imagenet-200/test', transform=test_compose)
        test_loader = torchdata.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    elif data.find('ImgNet') >= 0:
        print('loading ' + data)
        print(config_of_data)
        mean = [0, 0, 0]
        std = [1, 1, 1]
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        sampler_test , sampler_val = None, None
        if 'train' in splits:
            train_set = dset.ImageFolder('/data/ImageNet/train', transform=transform_train)
            if kwargs['dist'] is True:
                sampler_train = torch.utils.data.DistributedSampler(train_set)
                train_loader = torchdata.DataLoader(train_set, sampler=sampler_train, batch_size=batch_size, shuffle=False, num_workers=16)
            else:
                train_loader = torchdata.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        if 'val' in splits or 'test' in splits:
            test_set = dset.ImageFolder('/data/ImageNet/val', transform=transform_test)
            if kwargs['dist'] is True:
                sampler_test = torch.utils.data.DistributedSampler(test_set)
                test_loader = torchdata.DataLoader(test_set, sampler=sampler_test, batch_size=batch_size, shuffle=False, num_workers=16)
            else:
                test_loader = torchdata.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
            val_loader = test_loader
        val_loader = test_loader
        print("ImageNet test ")
    elif data.find('VOC') >= 0:
        print('loading ' + data)
        config = config_of_data['ssd_config'].config
        transform_train = TrainAugmentation_qat1(300, np.array([0,0,0]), 1.0)
        target_transform = MatchPrior(config['priors'], config['center_variance'], config['size_variance'], 0.5)
        test_transform = TestTransform(300, np.array([0,0,0]), 1.0)
        datasets = []
        dataset = VOCDataset('/data/dataset/VOCdevkit/voc_07_12_train/', transform=transform_train, target_transform=target_transform)
        datasets.append(dataset)
        valid_dataset = VOCDataset('/data/dataset/VOCdevkit/VOC2007TEST/', transform=test_transform, target_transform=target_transform, is_test=True)
        if 'train' in splits:
            train_dataset = dataset
            if kwargs['dist'] is True:
                sampler_train = torch.utils.data.DistributedSampler(train_dataset)
                train_loader = torchdata.DataLoader(train_dataset, sampler=sampler_train, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_voc_batch)
            else:
                sampler_train = None
                sampler_val = None
                sampler_test = None
                train_loader = torchdata.DataLoader(train_dataset, batch_size, num_workers=8, shuffle=True, collate_fn=collate_voc_batch)
        if 'val' in splits:
            if kwargs['dist'] is True:
                sampler_val = torch.utils.data.DistributedSampler(valid_dataset)
                val_loader = torchdata.DataLoader(valid_dataset, sampler=sampler_val, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=collate_voc_batch)
            else:
                val_loader = torchdata.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_voc_batch)
                sampler_train = None
                sampler_val = None
                sampler_test = None
        sampler_test, test_loader  = None, None
    else:
        raise NotImplemented

    return train_dataset, valid_dataset, sampler_test, train_loader, val_loader, test_loader
