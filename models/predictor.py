import torch

import box_utils
from data_provider.data_preprocessing import PredictionTransform_1, PredictionTransform


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform_1(size, mean, std)
        self.transform_batch = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.size = size
        self.sigma = sigma
        self.net.eval()

    def predict(self, images, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        if len(images.shape) == 4:
            b, c, height, width = images.shape
        else:
            c, height, width = images.shape

        if height!=self.size or width!=self.size: # Batch로 받으려면 collate_fn들어갈 때 size동일해야함,  dataset에서 transform함
            images = self.transform(images) # np.array to tensor, [H x W x C] -> [3, 300, 300]
        images = images.unsqueeze(0) # KJ, -> [1, 3, 300, 300]
        images = images.cuda()

        with torch.no_grad():
            scores, boxes = self.net.forward(images) # scores: [1, 3000, 21], boxes: [1, 3000, 4]

        boxes = boxes[0] # [3000, 4], (x1, y1, x2, y2) ==> final layer
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma, # in case of soft-NMS
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        picked_box_probs = torch.cat(picked_box_probs)
        
        print(picked_box_probs)
        print("piced box probs")
        input("press enter")
        
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height

        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]


    def predict_batch(self, images, policy, top_k=-1, prob_threshold=None, return_flops=False):

        self.net.eval()
        cpu_device = torch.device("cpu")
        b, c, height, width = images.shape
        # Batch로 받으려면 collate_fn들어갈 때 size동일해야함,
        # open image는 dataloder를안거치므로 network입력전에 size를 맞춰줘야함
        if height!=self.size or width!=self.size:
            images = self.transform_batch(images) # np.array to tensor, [H x W x C] -> [3, 300, 300]
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            scores_batch, boxes_batch, _flops = self.net.forward(images, policy) # scores: [B, 3000, 21], boxes: [B, 3000, 4] #SSD forward
        predictions = []
        for i in range(len(boxes_batch)): #Batch_size
            boxes = boxes_batch[i] # [3000, 4], (x1, y1, x2, y2) ==> final layer
            scores = scores_batch[i] # [3000, 21]
            if not prob_threshold:
                prob_threshold = self.filter_threshold
            # this version of nms is slower on GPU, so we move data to CPU.
            boxes = boxes.to(cpu_device)
            scores = scores.to(cpu_device)
            picked_box_probs = []
            picked_labels = []
            for class_index in range(1, scores.size(1)):
                probs = scores[:, class_index]
                mask = probs > prob_threshold
                probs = probs[mask]
                if probs.size(0) == 0:
                    continue
                subset_boxes = boxes[mask, :]
                box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
                box_probs = box_utils.nms(box_probs, self.nms_method,
                                          score_threshold=prob_threshold,
                                          iou_threshold=self.iou_threshold,
                                          sigma=self.sigma,
                                          top_k=top_k,
                                          candidate_size=self.candidate_size)
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.size(0))
            if not picked_box_probs:
                predictions.append((torch.tensor([]), torch.tensor([]), torch.tensor([])))
            else:
                picked_box_probs = torch.cat(picked_box_probs)
                picked_box_probs[:, 0] *= width
                picked_box_probs[:, 1] *= height
                picked_box_probs[:, 2] *= width
                picked_box_probs[:, 3] *= height
                predictions.append((picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]))
        if return_flops:
            return predictions, _flops
        assert(len(predictions) == len(images))
        return predictions

    def predict_batch_thresholding(self, images, top_k=-1, prob_threshold=None, entropy_mode=False):

        self.net.eval()
        cpu_device = torch.device("cpu")
        b, c, height, width = images.shape
        if height!=self.size or width!=self.size: # Batch로 받으려면 collate_fn들어갈 때 size동일해야함,  dataset에서 transform함
            images = self.transform_batch(images) # np.array to tensor, [H x W x C] -> [3, 300, 300]
        images = images.cuda()

        with torch.no_grad():
            scores_batch, boxes_batch = self.net.forward(images) # scores: [1, 3000, 21], boxes: [1, 3000, 4]
            ents_batch = torch.zeros_like(scores_batch).cuda()
            if entropy_mode is True: #entropy 계산을 위해 class probability 모두 필요함
                ents_batch = scores_batch.clone() # batch x 3000(anchors) x 21,,,--> including BACKGROUND
                ents_batch = -ents_batch*ents_batch.log()
                # score와 방향성을 맞추기 위해 1에서 빼줌       # batch x 3000(anchors) x 20
                # ent는 낮을수록 쉬운것, score는 낮을 수록 어려운 것이므로 '-'를 붙여서 동일하게
                ents_batch = ents_batch.sum(dim=2, keepdim=True) # batchx x anchors x 20 --> batch_size x 3000(anchors) x 1
        predictions = []

        for i in range(len(boxes_batch)):
            boxes = boxes_batch[i] # [3000, 4], (x1, y1, x2, y2) ==> final layer
            scores = scores_batch[i]
            ents = ents_batch[i]
            if not prob_threshold:
                prob_threshold = self.filter_threshold
            # this version of nms is slower on GPU, so we move data to CPU.
            boxes = boxes.to(cpu_device)
            scores = scores.to(cpu_device)
            ents = ents.to(cpu_device)
            picked_box_probs = []
            picked_labels = []
            picked_ents = []

            for class_index in range(1, scores.size(1)):
                probs = scores[:, class_index] # 해당 class 불러옴, 0.4, 0.6 class가 두 개 있을 때 두개 다 살아남으면?
                mask = probs > prob_threshold # threshol값 이하 mask
                probs = probs[mask]           # filterfing
                if probs.size(0) == 0:
                    continue
                subset_boxes = boxes[mask, :] # 해당 box만 불러옴
                box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
                box_probs, picked_indexes = box_utils.nms_thresholding(box_probs, self.nms_method,
                                              score_threshold=prob_threshold,
                                              iou_threshold=self.iou_threshold,
                                              sigma=self.sigma,
                                              top_k=top_k,
                                              candidate_size=self.candidate_size)

                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.size(0))
                ents_temp = ents[picked_indexes, :]
                picked_ents.append(ents[picked_indexes, :])

            if not picked_box_probs:
                predictions.append((torch.tensor([]), torch.tensor([]), torch.tensor([])))
            else:
                picked_box_probs = torch.cat(picked_box_probs)
                if entropy_mode is True:
                    picked_ents = torch.cat(picked_ents)
                    predictions.append((picked_box_probs[:, :4], torch.tensor(picked_labels), picked_ents))
                else:
                    predictions.append((picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]))

        return predictions


    def predict_batch_onnx(self, scores_batch, boxes_batch, top_k=-1, prob_threshold=None):

        cpu_device = torch.device("cpu")
        predictions = []

        for i in range(len(boxes_batch)):
            boxes = boxes_batch[i] # [3000, 4], (x1, y1, x2, y2) ==> final layer
            scores = scores_batch[i]
            if not prob_threshold:
                prob_threshold = self.filter_threshold
            # this version of nms is slower on GPU, so we move data to CPU.
            boxes = boxes.to(cpu_device)
            scores = scores.to(cpu_device)
            picked_box_probs = []
            picked_labels = []

            for class_index in range(1, scores.size(1)):
                probs = scores[:, class_index]
                mask = probs > prob_threshold
                probs = probs[mask]
                if probs.size(0) == 0:
                    continue
                subset_boxes = boxes[mask, :]
                box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)

                print("class, prob_threshold, mask.sum() : ", class_index, prob_threshold, mask.sum())
                print("box_probs : ", box_probs.shape)
                print("## let's do nms!!")
                input("press enter")

                box_probs = box_utils.nms(box_probs, self.nms_method,
                                          score_threshold=prob_threshold,
                                          iou_threshold=self.iou_threshold,
                                          sigma=self.sigma,
                                          top_k=top_k,
                                          candidate_size=self.candidate_size)

                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.size(0))
                print("boxes after NMS for class {}\n".format(class_index), box_probs)
                input("nms, press enter")
            if not picked_box_probs:
                predictions.append((torch.tensor([]), torch.tensor([]), torch.tensor([])))
                #return torch.tensor([]), torch.tensor([]), torch.tensor([])
            else:
                picked_box_probs = torch.cat(picked_box_probs)
                picked_box_probs[:, 0] *= width
                picked_box_probs[:, 1] *= height
                picked_box_probs[:, 2] *= width
                picked_box_probs[:, 3] *= height
                predictions.append((picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]))

        return predictions
