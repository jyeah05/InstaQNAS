from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from utils import AverageMeter, adjust_learning_rate, error
import time
from distributed import allreduce_grads, get_rank
from distributed import master_only_print as mprint # DDP. kihwan
import pathlib
import measurements
import box_utils
from copy import deepcopy
from tqdm import tqdm
import cv2
class Trainer(object):
    def __init__(self, model, criterion=None, optimizer=None, args=None, agent=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer # optimizer net
        self.args = args
        self.agent = agent
        
    def set_true_labels(self,dataset):
        print("set true labels")
        true_case_stat, all_gb_boxes, all_difficult_cases = self.group_annotation_by_class(dataset)
        self.true_case_stat =true_case_stat
        self.all_gb_boxes = all_gb_boxes
        self.all_difficult_cases = all_difficult_cases
        
    def free_true_labels(self):
        self.true_case_stat = None
        self.all_gb_boxes = None
        self.all_difficult_cases = None
        
    def calculate_bops(self, basenet_bops, model_type='VGG+SSD'):
        # calcuate total bops by adding extra / headers bops with basenet_bops
        # import pdb; pdb.set_trace()
        # breakpoint()
        # breakpoint()
        if model_type == 'MBv1_SSD-Lite':
            headers = 5792256 * self.args.head_wbit * self.args.head_abit
            extras = 28352512 * self.args.extras_wbit * self.args.extras_abit
        elif model_type == 'V1+SSD': # MBv1_SSD
            if self.args.image_size==300:
                basenet_bops = basenet_bops / 1e9 # (GBOPs)
                headers_first = ((39.92371+199.61856) * 8 * 8) /1e3  # FLOPs * bit * bit
                # headers_rest = (160.3584 * self.args.head_wbit * self.args.head_abit) /1e3  # FLOPs * bit * bit
                # --> 0.411 G만큼 높게 BOPs 측정됨 
                headers_rest = (153.94406 * self.args.head_wbit * self.args.head_abit) /1e3  # FLOPs * bit * bit
                extras =  (61.898752 * self.args.extras_wbit * self.args.extras_abit) /1e3
            elif self.args.image_size==224:
                basenet_bops = basenet_bops / 1e9 # (GBOPs)
                headers_first = (130.06 * self.args.head_wbit * self.args.head_abit) /1e3  # FLOPs * bit * bit
                headers_rest = (77.64 * self.args.head_wbit * self.args.head_abit) /1e3  # FLOPs * bit * bit
                extras = 2.22088
        elif model_type=='VGG+SSD':
            if self.args.image_size==300:
                basenet_bops = basenet_bops / 1e9 # (GBOPs)
                headers_first = ((106+532) * 8 * 8) /1e3  # FLOPs * bit * bit
                headers_rest = (556 * self.args.head_wbit * self.args.head_abit) /1e3  # FLOPs * bit * bit
                extras =  (231 * self.args.extras_wbit * self.args.extras_abit) /1e3
        elif model_type=='V2+SSD_Lite':
            if self.args.image_size==300:
                # breakpoint()
                basenet_bops = basenet_bops / 1e9 # (GBOPs)
                headers_first = 2.15588  # FLOPs * bit * bit
                headers_rest = 1.49445
                extras =  0.61675
        return headers_first + headers_rest + extras + basenet_bops #GBOPs
    
    def str_to_policy(self, policy_str, batch_size, num_blocks):
        policy_list = []
        policy_list.extend(policy_str)
        #print(policy_list)
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

    def ssd_train(self, train_loader, epoch, alpha, test_policy=False, 
                  test_policy_all_bit=None, exclude_1bit=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        regression_losses = AverageMeter()
        classification_losses = AverageMeter()
        flops = AverageMeter()

        self.model.train()
        end = time.time()
        if self.args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        for i, data in enumerate(tqdm(train_loader)):
            # measure data loading time
            # with torch.autograd.detect_anomaly():
            images, _, anno_boxes, anno_labels, anno_is_difficult = data
            # breakpoint()
            boxes = torch.stack(anno_boxes,0).cuda(non_blocking=True)
            labels = torch.stack(anno_labels,0).cuda(non_blocking=True)
            lr = adjust_learning_rate(self.optimizer, self.args.lr,
                                self.args.decay_rate, epoch,
                                self.args.epochs, self.args.lr_type, self.args.milestones, self.args.gamma, len(train_loader), i)
            data_time.update(time.time() - end)
            input_var = images.cuda(non_blocking=True)
            # input_var = (input_var*255).round_()
            '''
            Uniform random sampling : JYJ
            '''
            if self.agent is not None:
                self.agent.eval()
                _probs, _ = self.agent(input_var)
                _probs = _probs.clone().detach().cuda(non_blocking=True)
                probs = _probs*alpha + (1-_probs)*(1-alpha)
                distr = torch.distributions.Multinomial(1, probs)
                policy = distr.sample()
            elif test_policy is True:
                if test_policy_all_bit == '8':
                    _policy = '000100010001000100010001000100010001000100010001000100010001' # All 8bit
                elif test_policy_all_bit == '4':
                    _policy = '00100010001000100010001000100010001000100010001000100010001000100010' # All 4bit # VGG
                elif test_policy_all_bit == '2':
                    _policy = '010001000100010001000100010001000100010001000100010001000100' # All 2bit
                elif test_policy_all_bit == '1':
                    _policy = '100010001000100010001000100010001000100010001000100010001000' # All 1bit
                else:
                    raise NotImplementedError("Not existing arch_type")
                # breakpoint()
                policy = self.str_to_policy(_policy, len(images), self.model.module.base_net.num_of_blocks).cuda(non_blocking=True)
            else:
                if exclude_1bit:
                    # breakpoint()
                    policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
                    get_random_cand = lambda:tuple(np.random.randint(1, self.model.module.base_net.num_of_actions) for i in range(self.model.module.base_net.num_of_blocks))
                    for indx in range(images.shape[0]):
                        sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                        tmp = np.zeros((self.model.module.base_net.num_of_blocks, self.model.module.base_net.num_of_actions))
                        for b, a in enumerate(sampled_config):
                            tmp[b,a] = 1
                        policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
                    policy = torch.stack([p for p in policy], dim=0)
                else:
                    policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
                    get_random_cand = lambda:tuple(np.random.randint(self.model.module.base_net.num_of_actions) for i in range(self.model.module.base_net.num_of_blocks))
                    for indx in range(images.shape[0]):
                        sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                        tmp = np.zeros((self.model.module.base_net.num_of_blocks, self.model.module.base_net.num_of_actions))
                        for b, a in enumerate(sampled_config):
                            tmp[b,a] = 1
                        policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
                    policy = torch.stack([p for p in policy], dim=0)
            # import pdb; pdb.set_trace()
            confidences, locations, _ = self.model(input_var, policy)
            #print("confidence : ", confidences.shape)
            #print("locations :", locations.shape)
            # breakpoint()
            regression_loss, classification_loss  = self.criterion(confidences, locations, labels, boxes)
            regression_losses.update(regression_loss.data.item(), input_var.size(0))
            classification_losses.update(classification_loss.data.item(), input_var.size(0))
            loss = regression_loss + classification_loss
            # measure error and record loss
            losses.update(loss.data.item(), input_var.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.distributed:
                allreduce_grads(self.model)
            for p in self.model.module.parameters():
                if p.grad is not None and p.grad.sum() ==0:
                    p.grad = None
            self.optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        print('Epoch: {:3d} Train loss {loss.avg:.4f} '
              'Ref_loss@1 {reg_los.avg:.4f}'
              ' Class_loss@5 {cls_los.avg:.4f},'
              .format(epoch, loss=losses, reg_los=regression_losses, cls_los=classification_losses))
                
        return losses.avg, regression_losses.avg, classification_losses.avg, lr
    
    def ssd_fine_tune(self, train_loader, epoch, alpha, test_policy=False, 
                  test_policy_all_bit=None, exclude_1bit=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        regression_losses = AverageMeter()
        classification_losses = AverageMeter()
        flops = AverageMeter()

        self.model.train()
        end = time.time()
        if self.args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        for i, data in enumerate(tqdm(train_loader)):
            # measure data loading time
            # with torch.autograd.detect_anomaly():
            images, _, anno_boxes, anno_labels, anno_is_difficult = data
            # breakpoint()
            boxes = torch.stack(anno_boxes,0).cuda(non_blocking=True)
            labels = torch.stack(anno_labels,0).cuda(non_blocking=True)
            lr = adjust_learning_rate(self.optimizer, self.args.lr,
                                self.args.decay_rate, epoch,
                                self.args.epochs, self.args.lr_type, self.args.milestones, self.args.gamma, len(train_loader), i)
            data_time.update(time.time() - end)
            input_var = images.cuda(non_blocking=True)
            # input_var = (input_var*255).round_()
            '''
            Uniform random sampling : JYJ
            '''
            if self.agent is not None:
                self.agent.eval()
                with torch.no_grad():
                    _probs, _ = self.agent(input_var)
                    max_ops = torch.argmax(_probs, dim=2)
                    policy = torch.zeros(_probs.shape).cuda().scatter(2, max_ops.unsqueeze(2), 1.0)
            elif test_policy is True:
                if test_policy_all_bit == '8':
                    _policy = '000100010001000100010001000100010001000100010001000100010001' # All 8bit
                elif test_policy_all_bit == '4':
                    _policy = '001000100010001000100010001000100010001000100010001000100010' # All 4bit # VGG
                elif test_policy_all_bit == '2':
                    _policy = '010001000100010001000100010001000100010001000100010001000100' # All 2bit
                elif test_policy_all_bit == '1':
                    _policy = '100010001000100010001000100010001000100010001000100010001000' # All 1bit
                else:
                    raise NotImplementedError("Not existing arch_type")
                # breakpoint()
                policy = self.str_to_policy(_policy, len(images), self.model.module.base_net.num_of_blocks).cuda(non_blocking=True)
            else:
                if exclude_1bit:
                    # breakpoint()
                    policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
                    get_random_cand = lambda:tuple(np.random.randint(1, self.model.module.base_net.num_of_actions) for i in range(self.model.module.base_net.num_of_blocks))
                    for indx in range(images.shape[0]):
                        sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                        tmp = np.zeros((self.model.module.base_net.num_of_blocks, self.model.module.base_net.num_of_actions))
                        for b, a in enumerate(sampled_config):
                            tmp[b,a] = 1
                        policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
                    policy = torch.stack([p for p in policy], dim=0)
                else:
                    policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
                    get_random_cand = lambda:tuple(np.random.randint(self.model.module.base_net.num_of_actions) for i in range(self.model.module.base_net.num_of_blocks))
                    for indx in range(images.shape[0]):
                        sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                        tmp = np.zeros((self.model.module.base_net.num_of_blocks, self.model.module.base_net.num_of_actions))
                        for b, a in enumerate(sampled_config):
                            tmp[b,a] = 1
                        policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
                    policy = torch.stack([p for p in policy], dim=0)
            # import pdb; pdb.set_trace()
            confidences, locations, _ = self.model(input_var, policy)
            #print("confidence : ", confidences.shape)
            #print("locations :", locations.shape)
            # breakpoint()
            regression_loss, classification_loss  = self.criterion(confidences, locations, labels, boxes)
            regression_losses.update(regression_loss.data.item(), input_var.size(0))
            classification_losses.update(classification_loss.data.item(), input_var.size(0))
            loss = regression_loss + classification_loss
            # measure error and record loss
            losses.update(loss.data.item(), input_var.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.distributed:
                allreduce_grads(self.model)
            for p in self.model.module.parameters():
                if p.grad is not None and p.grad.sum() ==0:
                    p.grad = None
            self.optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        print('Epoch: {:3d} Train loss {loss.avg:.4f} '
              'Ref_loss@1 {reg_los.avg:.4f}'
              ' Class_loss@5 {cls_los.avg:.4f},'
              .format(epoch, loss=losses, reg_los=regression_losses, cls_los=classification_losses))
                
        return losses.avg, regression_losses.avg, classification_losses.avg, lr
    
    def save_sample_image(self, fname, image, labels, preds, gts, height, width, IMG_SIZE, class_names):
        device = "cuda:{}".format(image.get_device())
        preds = preds.to(device)
        original_size = torch.tensor([[width, height, width, height]]).to(device)
        preds = IMG_SIZE * (preds/original_size)
        preds = preds.clone().cpu().numpy()

        #gts *= IMG_SIZE # percent to network input image size
        gts = gts*IMG_SIZE if gts is not None else None # percent to network input image size
        #image = image*128+127
        image = image/255
        temp = image[2,:,:].clone()
        image[2,:,:] = image[0,:,:].clone()
        image[0,:,:] = temp.clone()
        image = image.permute(1,2,0).cpu().numpy()
        image = image.copy()

        #print("image shape", image.shape)
        #print("image \n", image[0,0,:10])
        # draw prediction boxes
        boxes = preds
        for i in range(len(boxes)):
            x1, y1, x2, y2= int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3])
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
            name = class_names[labels[i]]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, name , (x1, y2), font , 0.8, (255,0,0), 2)
            cv2.imwrite(fname, image)

        # draw gt boxes
        if gts is not None:
            boxes = gts
            for i in range(len(boxes)):
                x1, y1, x2, y2= int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3])
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(0,255,0), thickness = 2)
            cv2.imwrite(fname, image)

    def ssd_valid(self, valid_loader, epoch, test_policy=False, 
                  test_policy_all_bit=None, exclude_1bit=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        regression_losses = AverageMeter()
        classification_losses = AverageMeter()
        flops = AverageMeter()

        self.model.eval()
        end = time.time()
        targets_list = []
        if self.args.distributed:
            valid_loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                # measure data loading time 
                images, _, anno_boxes, anno_labels, anno_is_difficult = data
                boxes = torch.stack(anno_boxes,0).cuda(non_blocking=True)
                labels = torch.stack(anno_labels,0).cuda(non_blocking=True)
                data_time.update(time.time() - end)
                input_var = images.cuda(non_blocking=True)
                # input_var = (input_var*255).round_()
                '''
                Uniform random sampling : JYJ
                '''
                if self.agent is not None:
                    self.agent.eval()
                    _probs, _ = self.agent(input_var)
                    #print(probs)
                    _probs = _probs.clone().detach().cuda(non_blocking=True)
                    max_ops = torch.argmax(_probs, dim=2)
                    policy = torch.zeros(_probs.shape).cuda(non_blocking=True).scatter(2, max_ops.unsqueeze(2), 1.0)
                    #print("policy")
                elif test_policy is True:
                    if test_policy_all_bit == '8':
                        _policy = '000100010001000100010001000100010001000100010001000100010001' # All 8bit
                    elif test_policy_all_bit == '4':
                        _policy = '001000100010001000100010001000100010001000100010001000100010' # All 4bit # VGG
                    elif test_policy_all_bit == '2':
                        _policy = '010001000100010001000100010001000100010001000100010001000100' # All 2bit
                    elif test_policy_all_bit == '1':
                        _policy = '100010001000100010001000100010001000100010001000100010001000' # All 1bit
                    else:
                        raise NotImplementedError("Not existing arch_type")
                    # breakpoint()
                    policy = self.str_to_policy(_policy, len(images), 13).cuda(non_blocking=True)
                else:
                    if exclude_1bit:
                        policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
                        get_random_cand = lambda:tuple(np.random.randint(1, self.model.module.base_net.num_of_actions) for i in range(self.model.module.base_net.num_of_blocks))
                        for indx in range(images.shape[0]):
                            sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                            tmp = np.zeros((self.model.module.base_net.num_of_blocks, self.model.module.base_net.num_of_actions))
                            for b, a in enumerate(sampled_config):
                                tmp[b,a] = 1
                            policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
                        policy = torch.stack([p for p in policy], dim=0)
                    else:
                        policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
                        get_random_cand = lambda:tuple(np.random.randint(self.model.module.base_net.num_of_actions) for i in range(self.model.module.base_net.num_of_blocks))
                        for indx in range(input_var.shape[0]):
                            sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                            tmp = np.zeros((self.model.module.base_net.num_of_blocks, self.model.module.base_net.num_of_actions))
                            for b, a in enumerate(sampled_config):
                                tmp[b,a] = 1
                            policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
                        policy = torch.stack([p for p in policy], dim=0)
                
                confidences, locations, _ = self.model(input_var, policy)
                regression_loss, classification_loss  = self.criterion(confidences, locations, labels, boxes)
                regression_losses.update(regression_loss.data.item(), input_var.size(0))
                classification_losses.update(classification_loss.data.item(), input_var.size(0))
                loss = regression_loss + classification_loss
                # measure error and record loss
                losses.update(loss.data.item(), input_var.size(0))
                #flops.update(flop.data.item(), input_var.size(0))
                del data
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if self.args.print_freq > 0 and \
                    (i + 1) % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.6f}\t'
                      'Reg_loss {reg_los.val:.6f}\t'
                      'Class_loss {cls_los.val:.6f}\t'.format(
                          epoch, i + 1, len(valid_loader),
                          batch_time=batch_time, data_time=data_time, 
                          loss=losses, reg_los=regression_losses, cls_los=classification_losses))
        print('Epoch: {:3d} Val loss {loss.avg:.4f}\t'
              'Ref_loss@1 {reg_los.avg:.4f}\t'
              ' Class_loss@5 {cls_los.avg:.4f},'
              .format(epoch, loss=losses, reg_los=regression_losses, cls_los=classification_losses))
                
        return losses.avg, regression_losses.avg, classification_losses.avg
    def train(self, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()
        
        end = time.time()
        if self.args.distributed:
            train_loader.sampler.set_epoch(epoch)

        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            lr = adjust_learning_rate(self.optimizer, self.args.lr,
                                  self.args.decay_rate, epoch,
                                  self.args.epochs, self.args.lr_type, self.args.milestones, self.args.gamma, len(train_loader), i)
            data_time.update(time.time() - end)
            input_var = input.cuda(non_blocking=True)
            # input_var = (input_var*255).round_()
            target_var = target.cuda(non_blocking=True)
            '''
            Uniform random sampling : JYJ
            '''
            policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
            get_random_cand = lambda:tuple(np.random.randint(self.model.module.num_of_actions) for i in range(self.model.module.num_of_blocks))

            for indx in range(input.shape[0]):
                sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                    # make one-hot
                tmp = np.zeros((self.model.module.num_of_blocks, self.model.module.num_of_actions))
                 #print(sampled_config)
                for b, a in enumerate(sampled_config):
                    tmp[b,a] = 1
                    #print(tmp)
                policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
            policy = torch.stack([p for p in policy], dim=0)
            output, _ = self.model(input_var, policy, is_ssd=False)
            
            loss = self.criterion(output, target_var)

            # measure error and record loss
            err1, err5 = error(output.data, target_var, topk=(1, 5))
            losses.update(loss.data.item(), input_var.size(0))
            top1.update(err1.item(), input_var.size(0))
            top5.update(err5.item(), input_var.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.distributed:
                allreduce_grads(self.model)
            for p in self.model.module.parameters():
                if p.grad is not None and p.grad.sum() ==0:
                    p.grad = None
            self.optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.args.print_freq > 0 and \
                    (i + 1) % self.args.print_freq == 0:
                mprint('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Err@1 {top1.val:.4f}\t'
                      'Err@5 {top5.val:.4f}\t''LR: {lr}'.format(
                          epoch, i + 1, len(train_loader),
                          batch_time=batch_time, data_time=data_time,
                          loss=losses, top1=top1, top5=top5, lr=lr))
        mprint('Epoch: {:3d} Train loss {loss.avg:.4f} '
              'Err@1 {top1.avg:.4f}'
              ' Err@5 {top5.avg:.4f},'
              .format(epoch, loss=losses, top1=top1, top5=top5))
                
        return losses.avg, top1.avg, top5.avg, lr, 0

    def test(self, val_loader, epoch, silence=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        end = time.time()
        total_start = time.time()
        criterion =  torch.nn.CrossEntropyLoss()
        if self.args.distributed:
            val_loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input_var = input.cuda(non_blocking=True)
                # input_var = (input_var*255).round_()
                target_var = target.cuda(non_blocking=True)
                policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
                get_random_cand = lambda:tuple(np.random.randint(self.model.module.num_of_actions) for i in range(self.model.module.num_of_blocks))
                for indx in range(input.shape[0]):
                    sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                    # make one-hot
                    tmp = np.zeros((self.model.module.num_of_blocks, self.model.module.num_of_actions))
                    for b, a in enumerate(sampled_config):
                        tmp[b,a] = 1
                    policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
                policy = torch.stack([p for p in policy], dim=0)
                output, _ = self.model(input_var, policy, is_ssd=False)
                loss = criterion(output, target_var)

            # measure error and record loss
            err1, err5 = error(output.data, target_var, topk=(1, 5))
            losses.update(loss.data.item(), input_var.size(0))
            top1.update(err1.item(), input_var.size(0))
            top5.update(err5.item(), input_var.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if not silence:
            mprint('Epoch: {:3d} val   loss {loss.avg:.4f} Err@1 {top1.avg:.4f}'
                  ' Err@5 {top5.avg:.4f} Dur {t:4f}'.format(epoch, loss=losses,
                                                 top1=top1, top5=top5, t=time.time()-total_start))

        return losses.avg, top1.avg, top5.avg
    def ssd_test(self, test_loader, epoch, dataset, predictor=None, 
                 eval_path='./test', config=None, conf_threshold=0.01, 
                 test_policy=False, silence=False, test_policy_all_bit=None,
                 exclude_1bit=False, return_policy=False, atype='VGG+SSD'):
        batch_time = AverageMeter()
        avg_bops = []
        avg_aps = AverageMeter()
        eval_path = pathlib.Path(eval_path)
        eval_path.mkdir(exist_ok=True)
        class_names = [name.strip() for name in open('./voc-model-labels.txt').readlines()]
        # switch to evaluate mode
        self.model.eval()
        IMG_SIZE = config['image_size']
        true_case_stat, all_gb_boxes, all_difficult_cases = self.group_annotation_by_class(dataset)
        end = time.time()
        total_start = time.time()
        if self.args.distributed:
            print("distributed")
            test_loader.sampler.set_epoch(epoch)
        results = []
        target_list = []
        policies = []
        for k, data in enumerate(tqdm(test_loader)):
            # breakpoint()
            images, image_sizes, anno_boxes, anno_labels, anno_is_difficult = data# read batch iamges
            images = images.cuda(non_blocking=True)
            # images = (images*255).round_()
            batch_size = len(images)
            image_sizes = image_sizes.cpu()
            if self.agent is not None and not test_policy:
                # breakpoint()
                self.agent.eval()
                _probs, _ = self.agent(images)
                _probs = _probs.clone().detach().cuda(non_blocking=True)
                max_ops = torch.argmax(_probs, dim=2)
                policy = torch.zeros(_probs.shape).cuda(non_blocking=True).scatter(2, max_ops.unsqueeze(2), 1.0)
                if return_policy:
                    policies.append(policy.data.cpu())
            elif test_policy is True:
                # breakpoint()
                if atype == 'VGG+SSD':
                    if test_policy_all_bit == '8':
                        _policy = '000100010001000100010001000100010001000100010001000100010001' # All 8bit
                    elif test_policy_all_bit == '4':
                        _policy = '001000100010001000100010001000100010001000100010001000100010' # All 4bit # VGG
                    elif test_policy_all_bit == '2':
                        _policy = '010001000100010001000100010001000100010001000100010001000100' # All 2bit
                    elif test_policy_all_bit == '1':
                        _policy = '100010001000100010001000100010001000100010001000100010001000' # All 1bit
                    else:
                        raise NotImplementedError("Not existing arch_type")
                    # breakpoint()
                    policy = self.str_to_policy(_policy, len(images), 15).cuda(non_blocking=True)
                elif atype == 'V1+SSD':
                    if test_policy_all_bit == '8':
                        _policy = '0001000100010001000100010001000100010001000100010001' # All 8bit
                    elif test_policy_all_bit == '4':
                        _policy = '0010001000100010001000100010001000100010001000100010' # All 4bit # VGG
                    elif test_policy_all_bit == '2':
                        _policy = '0100010001000100010001000100010001000100010001000100' # All 2bit
                    elif test_policy_all_bit == '1':
                        _policy = '1000100010001000100010001000100010001000100010001000' # All 1bit
                    else:
                        raise NotImplementedError("Not existing arch_type")
                    # breakpoint()
                    policy = self.str_to_policy(_policy, len(images), 13).cuda(non_blocking=True)
                elif atype == 'V2+SSD_Lite':
                    if test_policy_all_bit == '8':
                        _policy = '0001000100010001000100010001000100010001000100010001' # All 8bit
                    elif test_policy_all_bit == '4':
                        _policy = '00100010001000100010001000100010001000100010001000100010001000100010' # All 4bit # VGG
                    elif test_policy_all_bit == '2':
                        _policy = '01000100010001000100010001000100010001000100010001000100010001000100' # All 2bit
                    elif test_policy_all_bit == '1':
                        _policy = '1000100010001000100010001000100010001000100010001000' # All 1bit
                    else:
                        raise NotImplementedError("Not existing arch_type")
                    # breakpoint()
                    policy = self.str_to_policy(_policy, len(images), 17).cuda(non_blocking=True)
            else:
                if exclude_1bit:
                    policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
                    get_random_cand = lambda:tuple(np.random.randint(1, self.model.module.base_net.num_of_actions) for i in range(self.model.module.base_net.num_of_blocks))
                    for indx in range(images.shape[0]):
                        sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                        tmp = np.zeros((self.model.module.base_net.num_of_blocks, self.model.module.base_net.num_of_actions))
                        for b, a in enumerate(sampled_config):
                            tmp[b,a] = 1
                        policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
                    policy = torch.stack([p for p in policy], dim=0)
                else:
                    policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
                    get_random_cand = lambda:tuple(np.random.randint(self.model.module.base_net.num_of_actions) for i in range(self.model.module.base_net.num_of_blocks))
                    for indx in range(images.shape[0]):
                        sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                        tmp = np.zeros((self.model.module.base_net.num_of_blocks, self.model.module.base_net.num_of_actions))
                        for b, a in enumerate(sampled_config):
                            tmp[b,a] = 1
                        policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
                    policy = torch.stack([p for p in policy], dim=0)
            # breakpoint()
            predictions, flops = predictor.predict_batch(images, policy=policy, top_k=-1, prob_threshold=conf_threshold, return_flops=True)
            bops = self.calculate_bops(flops, self.args.arch_type)
            avg_bops.append(bops.data.cpu())
            
            for i, (boxes, labels, probs) in enumerate(predictions): # return batch size
                #print("boxes device : " , boxes.shape, boxes.get_device())
                if len(boxes) == 0:
                    continue
                else:
                    with torch.no_grad():
                        height, width = image_sizes[i][1], image_sizes[i][2]
                        boxes[:, 0] *= (width / IMG_SIZE) # return to original size
                        boxes[:, 1] *= (height / IMG_SIZE)
                        boxes[:, 2] *= (width / IMG_SIZE)
                        boxes[:, 3] *= (height / IMG_SIZE)
                        #print("boxes")
                        #print(boxes)
                        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * (k*int(1.5*self.args.batch_size) + i)
                        results.append(torch.cat([indexes.reshape(-1, 1),labels.reshape(-1, 1).float(),probs.reshape(-1, 1),boxes + 1.0], dim=1))
        # breakpoint()
        try:
            results = torch.cat(results)
        except:
            return 0
        for class_index, class_name in enumerate(class_names):
            if class_index == 0: continue  # ignore background
            prediction_path = eval_path / f"det_test_{class_name}.txt"
            with open(prediction_path, "w") as f:
                sub = results[results[:, 1] == class_index, :]
                for i in range(sub.size(0)):
                    prob_box = sub[i, 2:].numpy()
                    ids = int(sub[i,0])
                    try:
                        image_id = dataset.ids[int(sub[i, 0])]
                        print(image_id + " " + " ".join([str(v) for v in prob_box]), file=f)
                    except:
                        a = int(sub[i,0])
        aps = []
        print("\n\nAverage Precision Per-class:")
        for class_index, class_name in enumerate(class_names):
            if class_index == 0:
                continue
            prediction_path = eval_path / f"det_test_{class_name}.txt"
            ap = self.compute_average_precision_per_class(
                true_case_stat[class_index],
                all_gb_boxes[class_index],
                all_difficult_cases[class_index],
                prediction_path,
                config['iou_threshold'],
                True
            )
            aps.append(ap)
            print(f"{class_name}: {ap}")
        avg_ap = 100*sum(aps)/len(aps)            
        # measure error and record loss
        avg_aps.update(avg_ap, images.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not silence:
            mprint('Epoch: {:3d} test   mAP {avg_ap.avg:.4f} Dur {t:4f}'.format(epoch, avg_ap=avg_aps,
                                                 t=time.time()-total_start))
        if return_policy:
            policies = torch.cat(policies, 0)
            policy_set = [np.reshape(p.cpu().numpy().astype(np.int).astype(np.str), (-1)) for p in policies]
            policy_set = set([''.join(p) for p in policy_set])
            avg_bops = torch.cat(avg_bops, 0)
            avg_bops  = avg_bops.mean()
            return avg_ap, avg_bops, policy_set
        avg_bops = torch.cat(avg_bops, 0)
        avg_bops  = avg_bops.mean()
        return avg_ap, avg_bops
    def ssd_validate_batch_prec_bops(self, images, anno_boxes, anno_labels, image_sizes, policy, predictor=None, eval_path='./test', config=None, conf_threshold=0.01):
        eval_path = pathlib.Path(eval_path)
        eval_path.mkdir(exist_ok=True) # Note this dataset is ""traindataset""
        # switch to evaluate mode
        self.model.eval()
        IMG_SIZE = config['image_size']
        batch_size = len(images)
        predictions, flops = predictor.predict_batch(images, policy=policy, prob_threshold=conf_threshold, return_flops=True)
        bops = self.calculate_bops(flops, self.args.arch_type)
        #print('policy in train')
        #print(policy)
        #print('flops')
        #print(flops)
        #print('bops')
        #print(bops)
        #input("enter")
        iou_threshold = 0.45
        batch_cls_set = set()
        # difficult 가 1인 애들은 애초에 vocdataset에서 짤려서 들어옴.
        for idx, anno_label in enumerate(anno_labels):
            anno_label_ = torch.from_numpy(anno_label).clone().detach()
            anno_label_list = anno_label_.tolist()
            cur_cls_set = list(set(anno_label_list))
            batch_cls_set.update(cur_cls_set)
        cls_tp_num, cls_gt_num, cls_fp_num , cls_fn_num= {}, {}, {}, {}
        cls_num = len(batch_cls_set)
        #print(batch_cls_set)
        for _cls in batch_cls_set:
            cls_tp_num[_cls] = 0
            cls_gt_num[_cls] = 0
            cls_fp_num[_cls] = 0
            cls_fn_num[_cls] = 0
        for img_id, (anno_box, anno_label) in enumerate(zip(anno_boxes, anno_labels)):
            height, width = image_sizes[img_id][1], image_sizes[img_id][2]
            anno_box = torch.from_numpy(anno_box).clone().detach()
            anno_label = torch.from_numpy(anno_label).clone().detach()
            anno_box[:,0] *= width # return to original size
            anno_box[:,1] *= height
            anno_box[:,2] *= width
            anno_box[:,3] *= height
            # anno_box 에서 class만 뽑아내서 set으로 만듬
            # set을 iteration시켜서 class 별 tp / fp 구함
            # 이때 주의할 것, 하나의 GT에 여러 개의 detection이 있는 경우 가장 큰 iou를 가진 box 외에 나머지는 다 fp가 됨
            # labels는 0, 0 , 0, c, c, c, 3, 3, 3, 4, 6, 6, 6, .. 이런 식으로 되어 있고
            # box는 (x1,y1,x2,y2), (x1,y1,x2,y2), (x1,y1,x2,y2), ..., .., .., (x1,y1,x2,y2), (x1,y1,x2,y2),  .. 이런 식
            # prob는 0.56, 0.3, 0.6, , .., .., .., 0.4, 0.6, 0.2, ... 이런식
            cur_anno_label_list = anno_label.tolist()
            pred_boxes, pred_labels, pred_probs = predictions[img_id]
            for gid, g_label in enumerate(cur_anno_label_list):
                gt_box = anno_box[gid]
                #print("gt_box")
                #print(gt_box)
                max_iou = 0
                max_box_id = -1
                with torch.no_grad():
                    for pred_box_id, pbox in enumerate(pred_boxes):
                        if pred_labels[pred_box_id] == g_label:
                            pbox[0] *= (width / IMG_SIZE) # return to original size
                            pbox[1] *= (height / IMG_SIZE)
                            pbox[2] *= (width / IMG_SIZE)
                            pbox[3] *= (height / IMG_SIZE)
                            this_iou = box_utils.iou_of(pbox, gt_box)
                            if this_iou > iou_threshold:
                                if this_iou > max_iou:
                                    max_iou = this_iou
                                    max_box_id = pred_box_id
                                else:
                                    cls_fp_num[g_label] += 1 # not argmax case
                            else:
                                cls_fp_num[g_label] += 1 # small iou case
                        else:
                            continue
                if max_box_id != -1:
                    cls_tp_num[g_label] += 1
                else:
                    cls_fn_num[g_label] += 1
        prec_batch, rec_batch = 0, 0
        for cls_id in batch_cls_set:
            if cls_tp_num[cls_id] ==0:
                cls_prec = 0
                cls_rec = 0
            else:   
                cls_prec = cls_tp_num[cls_id] / (cls_tp_num[cls_id] + cls_fp_num[cls_id] )
                cls_rec = cls_tp_num[cls_id] / (cls_tp_num[cls_id] + cls_fn_num[cls_id] )
            prec_batch += cls_prec
            rec_batch += cls_rec
        prec_batch /= cls_num
        rec_batch /= cls_num
        #batch_bops = torch.cat(bops, 0)
        batch_bops  = torch.mean(bops)
        prec_batch = np.array([prec_batch], dtype=np.float32)
        rec_batch = np.array([rec_batch], dtype=np.float32)
        #print(prec_batch)
        #print(rec_batch)
        return prec_batch, rec_batch, batch_bops #single, single, single

    def ssd_validate_batch_single_prec_bops(self, images, anno_boxes, anno_labels, image_sizes, policy, predictor=None, eval_path='./test', config=None, conf_threshold=0.01):
        eval_path = pathlib.Path(eval_path)
        eval_path.mkdir(exist_ok=True) # Note this dataset is ""traindataset""
        # switch to evaluate mode
        self.model.eval()
        IMG_SIZE = config['image_size']
        batch_size = len(images)
        predictions, flops = predictor.predict_batch(images, policy=policy, prob_threshold=conf_threshold, return_flops=True)
        bops = self.calculate_bops(flops, self.args.arch_type)
        #print('policy in train')
        #print(policy)
        #print('flops')
        #print(flops)
        #print('bops')
        #print(bops)
        #input("enter")
        prec_batch = np.zeros(len(images))
        rec_batch = np.zeros(len(images)) 
        min_score_batch = np.zeros(len(images)) # jeesak
        iou_threshold = 0.5
        # difficult 가 1인 애들은 애초에 vocdataset에서 짤려서 들어옴.
        for img_id, (anno_box, anno_label) in enumerate(zip(anno_boxes, anno_labels)):
            height, width = image_sizes[img_id][1], image_sizes[img_id][2]
            anno_box = torch.from_numpy(anno_box).clone().detach()
            anno_label = torch.from_numpy(anno_label).clone().detach()
            anno_box[:,0] *= width # return to original size
            anno_box[:,1] *= height
            anno_box[:,2] *= width
            anno_box[:,3] *= height
            # anno_box 에서 class만 뽑아내서 set으로 만듬
            anno_label_list = anno_label.tolist()
            cur_cls_set = list(set(anno_label_list))
            #print(cur_cls_set)
            cls_num = len(cur_cls_set)
            cls_tp_num, cls_gt_num, cls_fp_num , cls_fn_num= {}, {}, {}, {}
            for _cls in cur_cls_set:
                cls_tp_num[_cls] = 0
                cls_gt_num[_cls] = 0
                cls_fp_num[_cls] = 0
                cls_fn_num[_cls] = 0
            # set을 iteration시켜서 class 별 tp / fp 구함
            # 이때 주의할 것, 하나의 GT에 여러 개의 detection이 있는 경우 가장 큰 iou를 가진 box 외에 나머지는 다 fp가 됨
            # labels는 0, 0 , 0, c, c, c, 3, 3, 3, 4, 6, 6, 6, .. 이런 식으로 되어 있고
            # box는 (x1,y1,x2,y2), (x1,y1,x2,y2), (x1,y1,x2,y2), ..., .., .., (x1,y1,x2,y2), (x1,y1,x2,y2),  .. 이런 식
            # prob는 0.56, 0.3, 0.6, , .., .., .., 0.4, 0.6, 0.2, ... 이런식
            pred_boxes, pred_labels, pred_probs = predictions[img_id]
            # import pdb; pdb.set_trace()
            for gid, g_label in enumerate(anno_label_list):
                gt_box = anno_box[gid]
                #print("gt_box")
                #print(gt_box)
                max_iou = 0
                max_box_id = -1
                with torch.no_grad():
                    for pred_box_id, pbox in enumerate(pred_boxes):
                        if pred_labels[pred_box_id] == g_label:
                            pbox[0] *= (width / IMG_SIZE) # return to original size
                            pbox[1] *= (height / IMG_SIZE)
                            pbox[2] *= (width / IMG_SIZE)
                            pbox[3] *= (height / IMG_SIZE)
                            this_iou = box_utils.iou_of(pbox, gt_box)
                            if this_iou > iou_threshold:
                                if this_iou > max_iou:
                                    max_iou = this_iou
                                    max_box_id = pred_box_id
                                else:
                                    cls_fp_num[g_label] += 1 # not argmax case
                            else:
                                cls_fp_num[g_label] += 1 # small iou case
                        else:
                            continue
                if max_box_id != -1:
                    cls_tp_num[g_label] += 1
                else:
                    cls_fn_num[g_label] += 1
                
            prec_img, rec_img = 0, 0
            for cls_id in cur_cls_set:
                if cls_tp_num[cls_id] ==0:
                    cls_prec = 0
                    cls_rec = 0
                else:   
                    cls_prec = cls_tp_num[cls_id] / (cls_tp_num[cls_id] + cls_fp_num[cls_id] )
                    cls_rec = cls_tp_num[cls_id] / (cls_tp_num[cls_id] + cls_fn_num[cls_id] )
                prec_img += cls_prec
                rec_img += cls_rec
            # breakpoint()
            prec_img /= cls_num
            rec_img /= cls_num
            prec_batch[img_id] = prec_img
            rec_batch[img_id] = rec_img
            if len(pred_probs) > 0: # jeesak
                min_score_batch[img_id] = torch.min(pred_probs) # jeesak
            else:
                min_score_batch[img_id] = 0. # jeesak
        return prec_batch, rec_batch, bops, torch.from_numpy(min_score_batch).cuda() # jeesak
    
    def ssd_test_with_policy(self, test_loader, dataset, predictor=None, eval_path='./test', config=None, conf_threshold=0.01, test_policy=False, test_policy_all_bit=None):
        batch_time = AverageMeter()
        avg_aps = AverageMeter()
        #avg_flops = AverageMeter()
        avg_flops = []
        eval_path = pathlib.Path(eval_path)
        eval_path.mkdir(exist_ok=True)
        class_names = [name.strip() for name in open('./voc-model-labels.txt').readlines()]
        # switch to evaluate mode
        self.model.eval()
        IMG_SIZE = config['image_size']
        true_case_stat, all_gb_boxes, all_difficult_cases = self.group_annotation_by_class(dataset)
        end = time.time()
        total_start = time.time()
        if self.args.distributed:
            print("distributed")
            test_loader.sampler.set_epoch(epoch)
        results = []
        target_list = []
        for k, data in enumerate(tqdm(test_loader)):
            images, image_sizes, anno_boxes, anno_labels, anno_is_difficult = data # read batch iamges
            images = images.cuda(non_blocking=True)
            # images = (images*255).round_()
            batch_size = len(images)
            image_sizes = image_sizes.cpu()
            if self.agent is not None:
                self.agent.eval() 
                #print("agent")
                _probs, _ = self.agent(images)
                _probs = _probs.clone().detach().cuda(non_blocking=True)
                max_ops = torch.argmax(_probs, dim=2)
                policy = torch.zeros(_probs.shape).cuda(non_blocking=True).scatter(2, max_ops.unsqueeze(2), 1.0)
            elif test_policy is True:
                if test_policy_all_bit == '8':
                    _policy = '0001000100010001000100010001000100010001000100010001' # All 8bit
                elif test_policy_all_bit == '4':
                    _policy = '0010001000100010001000100010001000100010001000100010' # All 4bit
                elif test_policy_all_bit == '2':
                    _policy = '0100010001000100010001000100010001000100010001000100' # All 2bit
                elif test_policy_all_bit == '1':
                    _policy = '1000100010001000100010001000100010001000100010001000' # All 1bit
                else:
                    raise NotImplementedError("Not existing arch_type")
                policy = self.str_to_policy(_policy, len(images), 13).cuda(non_blocking=True) 
                #print(policy)
            else:
                #print("agent is not loaded")
                policy = [] #shape = (input.shape[0], self.model.module.num_of_blocks, self.model.module.num_of_actions)
                get_random_cand = lambda:tuple(np.random.randint(self.model.module.base_net.num_of_actions) for i in range(self.model.module.base_net.num_of_blocks))
                for indx in range(images.shape[0]):
                    sampled_config = get_random_cand() # (1, 3, 5, .., 0) len(can) == self.module.num_of_blocks
                    tmp = np.zeros((self.model.module.base_net.num_of_blocks, self.model.module.base_net.num_of_actions))
                    for b, a in enumerate(sampled_config):
                        tmp[b,a] = 1
                    policy.append(torch.from_numpy(np.array(tmp)).long().cuda(non_blocking=True))
                policy = torch.stack([p for p in policy], dim=0)
            predictions, flops = predictor.predict_batch(images, policy=policy, top_k=-1, prob_threshold=conf_threshold, return_flops=True)
            #avg_flops.update(flops, images.size(0))
            avg_flops.append(flops.cpu().numpy())
            for i, (boxes, labels, probs) in enumerate(predictions): # return batch size
                if len(boxes) == 0:
                    continue
                else:
                    with torch.no_grad():
                        height, width = image_sizes[i][1], image_sizes[i][2]
                        boxes[:, 0] *= (width / IMG_SIZE) # return to original size
                        boxes[:, 1] *= (height / IMG_SIZE)
                        boxes[:, 2] *= (width / IMG_SIZE)
                        boxes[:, 3] *= (height / IMG_SIZE)
                        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * (k*int(1.5*self.args.batch_size) + i)
                        results.append(torch.cat([indexes.reshape(-1, 1),labels.reshape(-1, 1).float(),probs.reshape(-1, 1),boxes + 1.0], dim=1))
        results = torch.cat(results)
        for class_index, class_name in enumerate(class_names):
            if class_index == 0: continue  # ignore background
            prediction_path = eval_path / f"det_test_{class_name}.txt"
            with open(prediction_path, "w") as f:
                sub = results[results[:, 1] == class_index, :]
                for i in range(sub.size(0)):
                    prob_box = sub[i, 2:].numpy()
                    ids = int(sub[i,0])
                    try:
                        image_id = dataset.ids[int(sub[i, 0])]
                        print(image_id + " " + " ".join([str(v) for v in prob_box]), file=f)
                    except:
                        a = int(sub[i,0])
        aps = []
        print("\n\nAverage Precision Per-class:")
        for class_index, class_name in enumerate(class_names):
            if class_index == 0:
                continue
            prediction_path = eval_path / f"det_test_{class_name}.txt"
            ap = self.compute_average_precision_per_class(
                true_case_stat[class_index],
                all_gb_boxes[class_index],
                all_difficult_cases[class_index],
                prediction_path,
                config['iou_threshold'],
                True
            )
            aps.append(ap)
            print(f"{class_name}: {ap}")
            avg_ap = 100*sum(aps)/len(aps)            
            # measure error and record loss
            avg_aps.update(avg_ap, images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        avg_flops = np.mean(np.concatenate([f for f in avg_flops]))
        #print(avg_flops/1e9)
        
        return avg_aps.avg, avg_flops
    def group_annotation_by_class(self, dataset):
        true_case_stat = {}
        all_gt_boxes = {}
        all_difficult_cases = {}
        for i in range(len(dataset)):
            image_id, annotation = dataset.get_annotation(i)
            gt_boxes, classes, is_difficult = annotation
            gt_boxes = torch.from_numpy(gt_boxes)
            for i, difficult in enumerate(is_difficult):
                class_index = int(classes[i])
                gt_box = gt_boxes[i]
                if not difficult:
                    true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

                if class_index not in all_gt_boxes:
                    all_gt_boxes[class_index] = {}
                if image_id not in all_gt_boxes[class_index]:
                    all_gt_boxes[class_index][image_id] = []
                all_gt_boxes[class_index][image_id].append(gt_box)
                if class_index not in all_difficult_cases:
                    all_difficult_cases[class_index]={}
                if image_id not in all_difficult_cases[class_index]:
                    all_difficult_cases[class_index][image_id] = []
                all_difficult_cases[class_index][image_id].append(difficult)

        for class_index in all_gt_boxes:
            for image_id in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
        for class_index in all_difficult_cases:
            for image_id in all_difficult_cases[class_index]:
                all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
        return true_case_stat, all_gt_boxes, all_difficult_cases
    def compute_average_precision_per_class(self, num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
        with open(prediction_file) as f:
            image_ids = []
            boxes = []
            scores = []
            for line in f:
                t = line.rstrip().split(" ")
                image_ids.append(t[0])
                scores.append(float(t[1]))
                box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
                box -= 1.0  # convert to python format where indexes start from 0
                boxes.append(box)
            scores = np.array(scores)
            sorted_indexes = np.argsort(-scores)
            boxes = [boxes[i] for i in sorted_indexes]
            image_ids = [image_ids[i] for i in sorted_indexes]
            true_positive = np.zeros(len(image_ids))
            false_positive = np.zeros(len(image_ids))
            matched = set()
            for i, image_id in enumerate(image_ids):
                box = boxes[i]
                if image_id not in gt_boxes:
                    false_positive[i] = 1
                    continue

                gt_box = gt_boxes[image_id]
                ious = box_utils.iou_of(box, gt_box)
                max_iou = torch.max(ious).item()
                max_arg = torch.argmax(ious).item()
                if max_iou > iou_threshold:
                    if difficult_cases[image_id][max_arg] == 0:
                        if (image_id, max_arg) not in matched:
                            true_positive[i] = 1
                            matched.add((image_id, max_arg))
                        else:
                            false_positive[i] = 1
                else:
                    false_positive[i] = 1

        true_positive = true_positive.cumsum()
        false_positive = false_positive.cumsum()
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / num_true_cases
        if use_2007_metric:
            return measurements.compute_voc2007_average_precision(precision, recall)
        else:
            return measurements.compute_average_precision(precision, recall)

