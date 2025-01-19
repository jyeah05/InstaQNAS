import collections
import torch
import itertools
from typing import List
import math

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])


def generate_ssd_priors(specs: List[SSDSpec], image_size, clamp=True) -> torch.Tensor:
    """Generate SSD Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
            ]
        image_size: image size.
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            # small sized square box
            size = spec.box_sizes.min
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # big sized square box
            size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # change h/w ratio of the small sized box
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio)
                priors.append([
                    x_center,
                    y_center,
                    w * ratio,
                    h / ratio
                ])
                priors.append([
                    x_center,
                    y_center,
                    w / ratio,
                    h * ratio
                ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    #print("device in box_utils.py : locations, priors", locations.get_device(), priors.get_device())
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """

    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def iou_of_int(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """

    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    # return overlap_area / (area0 + area1 - overlap_area + eps)
    # print("boxees0, 1 :", boxes0, boxes1)
    # print("area0 :", area0)
    #mask = area0==0.
    mask = torch.ones_like(area0)
    out_int = 255*overlap_area / (area0 + area1 - overlap_area + eps)
    # print("shape : area0, area1, mask, out_int", area0.shape, area1.shape, mask.shape, out_int.shape)
    # input("press enter")
    out_int[mask] = 255
    return out_int
    #return 255*overlap_area / (area0 + area1 - overlap_area + eps)



def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1) # 해당 prior에서 가장 iou높은 target 선택
    #size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0) # 해당 target에서 가장 iou높은 prior 선택
    
    for target_index, prior_index in enumerate(best_prior_per_target_index): # targetdp 대해 iou가장 높은 prior index가져옴
        best_target_per_prior_index[prior_index] = target_index   # 해당 위치를 target(0 ~ target 갯수)의 index로 채움
    # 2.0 is used to make sure every target has a prior assigned
    # best_prior_per_target_index에 해당하는 index만 2로 채움.
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2) # index_fill : 해당 index를 2로 채움
    # size: num_priors
    # best_target_per_prior_index사이즈로 broadcasting됨, 해당 index에 있는 label을 gt_label에서 가져와서 채움
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index] # target이랑 가장 iou높은
    #print("==========================================================\n")
    return boxes, labels


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    --> dim=1로 해서 batch 단위로 돌아가도록 되어 있음
    --> 이 말은 positive, negative비율이 이미지 단위로 유지된다는 말인것 같음
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """

    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg

    # print('In box_utils')
    # print('[shape] pos_mask, labels, num_pos', pos_mask.shape, labels.shape, num_pos.shape)
    # print('[shape] indexes, orders ', indexes.shape, orders.shape)
    # print('neg_mask shape', neg_mask.shape)
    # print('neg_maks sum : ', neg_mask.sum(dim=1))
    # print('num_pos : ', num_pos) 
    # input('press enter\n')

    return pos_mask | neg_mask


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:]/2,
                     locations[..., :2] + locations[..., 2:]/2], locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
         boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """

    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]    
    #print("scores shape in hard_nms : ", scores.shape, len(indexes))
    #print("doing hard_nms : ")
    # print('indexes : ', indexes)
    while len(indexes) > 0: # indexes에 아무것도 없을 때까지 반복
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:] # 현재 뽑힌 box를 indexes에서 제외
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        ) # current box와 rest_boxes간의 iou를 비교
        '''
        print(" indexes : ", len(indexes))
        print(" current id : ", current)
        print(" current box : ", current_box)
        print(" iou of \n", iou)
        print(" piced : \n", picked)
        input(" hard_nms, press enter")
        '''
        # 남아 있는 indexes에 iou가 iou_threshold보다 크면 current와 겹치는 box이므로 제거
        indexes = indexes[iou <= iou_threshold]
    
    # print('current : ', current)

    return box_scores[picked, :]


def hard_nms_train(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """

    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    #print("scores shape in hard_nms : ", scores.shape, len(indexes))
    #print("doing hard_nms : ")
    # print('indexes : ', indexes)
    # print('scores : ', scores )
    while len(indexes) > 0:  # indexes에 아무것도 없을 때까지 반복
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]  # 현재 뽑힌 box를 indexes에서 제외
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )  # current box와 rest_boxes간의 iou를 비교
        '''
        print(" indexes : ", len(indexes))
        print(" current id : ", current)
        print(" current box : ", current_box)
        print(" iou of \n", iou)
        print(" piced : \n", picked)
        input(" hard_nms, press enter")
        '''
        # 남아 있는 indexes에 iou가 iou_threshold보다 크면 current와 겹치는 box이므로 제거
        indexes = indexes[iou <= iou_threshold]

    # print('current : ', current)
    # print('picked : ', picked)
    # print('picked shape : ', picked.shape)
    # print('---------------------------end of hard_nms_train')

    return box_scores[picked, :], picked

def hard_nms_int(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """

    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    #print("scores shape in hard_nms : ", scores.shape, len(indexes))
    #print("doing hard_nms : ")
    while len(indexes) > 0: # indexes에 아무것도 없을 때까지 반복
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:] # 현재 뽑힌 box를 indexes에서 제외
        rest_boxes = boxes[indexes, :]

        # print(" indexes : ", len(indexes))
        # print(" current id : ", current)
        # print(" current box : ", current_box)
        # print(" iou of \n", iou)
        # print(" piced : \n", picked)
        # input(" hard_nms, press enter")

        iou = iou_of_int(
            rest_boxes,
            current_box.unsqueeze(0),
        ) # current box와 rest_boxes간의 iou를 비교
        '''
        print(" indexes : ", len(indexes))
        print(" current id : ", current)
        print(" current box : ", current_box)
        print(" iou of \n", iou)
        print(" piced : \n", picked)
        input(" hard_nms, press enter")
        '''
        # 남아 있는 indexes에 iou가 iou_threshold보다 크면 current와 겹치는 box이므로 제거
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def hard_nms_thresholding(box_scores, iou_threshold, top_k=-1, candidate_size=200, use_nms=True):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """

    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    #print("scores shape in hard_nms : ", scores.shape, len(indexes))
    #print("doing hard_nms : ")
    #print("Indexes befor nms : \n", indexes)

    if use_nms:
        while len(indexes) > 0: # indexes에 아무것도 없을 때까지 반복
            #print(" indexes : ", len(indexes))
            current = indexes[0]
            picked.append(current.item())
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:] # 현재 뽑힌 box를 indexes에서 제외
            rest_boxes = boxes[indexes, :]
            iou = iou_of(
                rest_boxes,
                current_box.unsqueeze(0),
            ) # current box와 rest_boxes간의 iou를 비교
            indexes = indexes[iou <= iou_threshold] # 남아 있는 indexes에 iou가 iou_threshold보다 작으면
            # ioou가 threshold보다 높은 것은 현재 box랑 겹치는 것이므로 제외
            # threshold보다 낮으면 안겹치는 box이므로 살려둠
    else:
        for current in indexes:
            picked.append(current.item())

    #print("picked indexes after nms : \n", picked)

    return box_scores[picked, :], picked

def nms_thresholding(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    return hard_nms_thresholding(box_scores, iou_threshold, top_k, candidate_size=candidate_size)


def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)


def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])
