import torch.nn as nn
import torch.nn.functional as F
import torch


import box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, conf_threshold
                 ):
        """Implement SSD Multibox Loss.
        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold # NMS iou threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.cuda(non_blocking=True)
        self.conf_threshold = conf_threshold # NMS confidence threshold
        # self.return_loss_per_image = return_loss_per_image

    def forward(self, confidence, predicted_locations, labels, gt_locations, 
                return_loss_per_image=False,
                normalize_loss_to_num=True,
                multibox_loss_mode = None):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
            return_loss_per_image : return loss per image
            multibox_loss_mode :
                None(default) : regression loss + classification loss( neg_mask, nms mask동시 사용 --> neg_mask가 nms_mask를 포함함)
                'reg-nms' : regression loss + classification loss(nms_mask 사용) 
                            --> reg/num_pos, cls/num_pos ( num_nms아닌, num_pos로 일단 normalization)
                'nms' : no regression loss, only classification loss(nms_mask) 
                            --> cls/num_nms 로 normalize
        Output:
            if return_loss_per_image : (batch_size, 1)
            else                     : (1) --> scalar
        NMS loss:
            hard negative mining은 8bit, 4bit에서 값이 비슷함
            smooth l1 loss도 target box의 갯수에 따라 loss값이 정해지므로 8/4bit에 따라서 값이 비슷함
            nms후의 결과를 반영하도록 하면
        
        """
        num_classes = confidence.size(2)
        batch_size = confidence.size(0)
        device = confidence.device
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0] # background에 대해서만 hardnegative mining
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        
            ## nms loss            
            ## Generate nms mask
            # if use_hard_nms_mining:                
            if multibox_loss_mode == 'reg-nms' or multibox_loss_mode == 'nms':
                mask_nms = torch.zeros_like(mask).bool().to(device) #(batch_size, 1602)
                # nms를 한 box만 loss 계산 하려고 했으나
                # pos_mask가 gt와 iot_threshold이상 겹치는 모든 prior에 대해 loss계산
                # 따라서 갯수가 gt보다 많음, neg_mask는 pos mask의 3배
                # nms를 진행해버리면 갯수가 줄어들것임
                mask_ids_picked = self.hard_nms_mining(confidence, predicted_locations, self.iou_threshold, self.conf_threshold)
                
                print()

                for b, b_ids in mask_ids_picked.items():
                    if len(b_ids) > 0:
                        mask_nms[b, b_ids] = True
                mask_box_id = mask.nonzero() # True인 곳의 index
                mask_nms_box_id = mask_nms.nonzero() # True인 곳의 index
                num_nms = len(mask_nms_box_id)
        
        # nms loss를 어떻게 조합 할 것인지
        if multibox_loss_mode == 'reg-nms' or multibox_loss_mode == 'nms':
            mask = mask | mask_nms        
            
        confidence = confidence[mask, :]
        if return_loss_per_image:
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes),
                                                  labels[mask], reduction='none')
            classification_loss = classification_loss.unsqueeze(1) # (num_of_positives, 1)
            
        else:
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False, reduction='none')
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)        
        ## 개별 이미지별로 처리되도록 만들어줘야함. #############################
        # locations : back ground를 제외한 box의 갯수
        # confidences : back graound까지 포함되어진 갯수, 갯수가 더 많음
        # pos_mask_idx : (num of boxes, 2)
        # - 0번째 : image id number
        # - 1번째 : one --> 나중에 probs를 곱해줌
        # conf_mask_idx : (num of posi and negh, 2)
        # - 0번째 : image id number
        # - 1번째 : one --> 나중에 probs를 곱해줌
        pos_mask_idx = torch.cat([torch.ones((1,pos_mask.shape[1],1))*i for i in range(pos_mask.shape[0])], dim=0).to(pos_mask.device) # (num_predicted_box, 1) batch size만큼 idx를 부여하도록 만듦
        # print('pos mask idx shape : ', pos_mask_idx.shape, predicted_locations.shape)
        pos_mask_idx = pos_mask_idx[pos_mask, :] # size : (num_predicted_box, 2)
        pos_mask_idx = torch.cat([pos_mask_idx, torch.ones_like(pos_mask_idx)], dim=1).to(pos_mask.device)
        # print('pos mask idx shape : ', pos_mask_idx.shape)
        # for i in range(len(mask)):
        #     r = len(mask[i, mask[i,:]==1])
        #     print('r : ', r)
        #     t = torch.ones((1, r))*i
        #     print('t', t)
        conf_mask_idx = torch.cat([torch.ones((len(mask[i, mask[i, :] == 1]),1)) *i 
                            for i in range(len(mask))], dim=0).to(pos_mask.device)        
        # print('[shape] conf_mask_idx : ', conf_mask_idx.shape)
        ####################################################################
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        if return_loss_per_image:
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='none')
        else:
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        
        num_pos = gt_locations.size(0)

        # print('num_pos, neg_ratio, total_mask', num_pos, self.neg_pos_ratio, mask.sum())
        
        # print('mask shape : ', mask.shape)
        # print('mask size : ', mask.sum())
        # print('pos mask shape : ', pos_mask.shape)
        # print('predicted_locations, gt : ', predicted_locations.shape, gt_locations.shape)
        # print('num_pos : ', num_pos)     

        # print('smooth_l1_loss, classification : ', smooth_l1_loss.size(), classification_loss.size())  

        # input('multiboxloss, press enter \n')

        ''' image 별로 loss를 구하도록 정렬'''
        if return_loss_per_image:
            
            smooth_l1_loss = torch.cat([smooth_l1_loss[pos_mask_idx[:, 0] == i, :].sum().view(-1, 1)
                                  for i in range(batch_size)]).to(device).squeeze(1)  # (batch_size, 1) --> (batch_size)

            classification_loss = torch.cat([classification_loss[conf_mask_idx[:, 0] == i, :].sum().view(-1, 1)
                          for i in range(batch_size)]).to(device).squeeze(1)  # (batch_size, 1) --> (batch_size)
        #     print('---- in multibox_loss.py')
            # print('loss_reg : \n', smooth_l1_loss/num_pos)
            # print('loss_cls : \n', classification_loss)
        #     print('--------------------------------------------')        

        if normalize_loss_to_num is False: # denormalize
            num_pos = 1.
            num_nms = 1.
        # print('num_pos : ', normalize_loss_to_num)
        # print('num_pos : ', num_pos)
        # input('press enter, multiboxloss\n')
        if return_loss_per_image:
            if multibox_loss_mode == 'nms':
                return torch.zeros_like(smooth_l1_loss).to(device), classification_loss/num_nms, pos_mask_idx, conf_mask_idx
            else:
                return smooth_l1_loss/num_pos, classification_loss/num_pos, pos_mask_idx, conf_mask_idx
            # return smooth_l1_loss, classification_loss, pos_mask_idx, conf_mask_idx
        else:
            return smooth_l1_loss/num_pos, classification_loss/num_pos

    @torch.no_grad()
    def hard_nms_mining(self, scores, boxes_batch, iou_thresh, prob_threshold):
        '''
        Input:
            scores : (batch, num_priors, num_classes)
            locations : (batch, num_priors, 4)
        Out : 
            return positive mask after NMS
        
        nms과정을 동일하게 할 것이냐
        아니면 threshold보다 큰것들을 골라내고
        그것들은 나중에 False Positive가 될 가능성이 크므로
        negative mining에서 했던 것 처럼
        '''

        scores_batch = F.softmax(scores, dim=2)     
        picked_ids = {i:[] for i in range(scores_batch.size(0))} # {'batch numbers' : box_ids picked}최종 선택된 box들의 priors 번호
        box_ids_ref = torch.arange(scores_batch.size(1)) # number of prior boxes ( 1602)

        for i in range(len(boxes_batch)):
            boxes = boxes_batch[i]  # [3000, 4], (x1, y1, x2, y2) ==> final layer
            scores = scores_batch[i]
            scores_max = torch.argmax(scores, dim=1)            

            # this version of nms is slower on GPU, so we move data to CPU.
            boxes = boxes.cpu()
            scores = scores.cpu()
            picked_box_probs = []
            picked_labels = []
            
            for class_index in range(1, scores.size(1)):
                probs = scores[:, class_index]
                mask = probs > prob_threshold # (batch_size, num_priors, 1)
                # print('scores : ', scores[0:10])
                # print('th = {}, class[{}] mask length : {}'.format(prob_threshold, class_index, mask.sum()))
                probs = probs[mask]
                
                # print('mask shape : ', mask.shape)
                if probs.size(0) == 0:
                    continue
                subset_boxes = boxes[mask, :]  # (sum of mask)
                box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
                '''
                print("class, prob_threshold, mask.sum() : ", class_index, prob_threshold, mask.sum())
                print("box_probs : ", box_probs.shape)
                print("## let's do nms!!")
                input("press enter")
                '''
                _, box_ids_mask = box_utils.hard_nms_train(box_probs, 
                                        iou_threshold=self.iou_threshold,
                                        top_k=-1,
                                        candidate_size=200)
                box_ids_picked = box_ids_ref[mask]              # probs에서 뽑인 box id를 추출함
                box_ids_picked = box_ids_picked[box_ids_mask]   # nms에서 살아 남은 box만 다시 추출
                picked_ids[i].append(box_ids_picked.long().unsqueeze(0)) # batch 별로 최대 class갯수 만큼의 길이를 갖는 list 생김                
                
            # remove same box ids  
            none_flag = 0
            for j in picked_ids[i]:
                none_flag += len(j)
            if none_flag == 0:
                picked_ids[i] = []
            else:
                picked_ids[i] = torch.cat(picked_ids[i], dim=1) # class별로 뽑힌 id를 하나의 tensor로 합침
                # print(f'picked_ids[{i}] : ', picked_ids[i], picked_ids[i].shape)
                picked_ids[i] = torch.unique(picked_ids[i], sorted=True)
        

        return picked_ids
