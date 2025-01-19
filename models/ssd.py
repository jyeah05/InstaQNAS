import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple
import box_utils
import pickle

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.Sequential, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, config=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.config = config
        self.priors = config["priors"].cuda()
        save_obj(self.priors, "priors_{}.pickle".format(config['image_size']))
    def train_PACT_param(self):
        for name, pg in self.named_parameters():
            if 'alpha' in name:
                pg.requires_grad = True
    def freeze_PACT_param(self):
        for name, pg in self.named_parameters():
            if 'alpha' in name:
                pg.requires_grad = False
    def freeze_base_param(self):
        for name, pg in self.named_parameters():
            if 'base_net' in name:
                pg.requires_grad = False

    def forward(self, x: torch.Tensor, policy:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        t_flops = torch.zeros(x.size(0), requires_grad=False).cuda().float()
        outputs, _flops = self.base_net.forward(x, policy, is_ssd=True) # 0:(0-12), 1:(0-14), 2:(0-16)
        t_flops += _flops
        for header_index, y in enumerate([outputs[0], outputs[1]]):
            confidence, location = self.compute_header(header_index, y)
            confidences.append(confidence)
            locations.append(location)
        x = outputs[1]
        for layer in self.extras:
            x = layer(x) #QconvBlock_BIAS
            header_index +=1
            confidence, location = self.compute_header(header_index, x) #Qconv_MERE
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.training:
            return confidences, locations, t_flops
        else:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config["center_variance"], self.config["size_variance"] # KJ
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes, t_flops # boxes = x1, y1, x2, y2 
    def forward_test_ops(self, x):
        confidences = []
        locations = []
        t_flops = torch.zeros(x.size(0), requires_grad=False).cuda().float()
        outputs, _flops = self.base_net.forward_test_ops(x, is_ssd=True) # 0:(0-12), 1:(0-14), 2:(0-16)
        t_flops += _flops
        assert(len(outputs)==2)
        for header_index, y in enumerate([outputs[0], outputs[1]]):
            confidence, location = self.compute_header(header_index, y)
            confidences.append(confidence)
            locations.append(location)
        x = outputs[1]
        for layer in self.extras:
            x = layer(x) #QconvBlock_BIAS
            header_index +=1
            confidence, location = self.compute_header(header_index, x) #Qconv_MERE
            confidences.append(confidence)
            locations.append(location)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        if self.training:
            return confidences, locations, t_flops
        else:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config["center_variance"], self.config["size_variance"] # KJ
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes, t_flops # boxes = x, y, h, w
           
    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        location  = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        print("Load FP extras, and headers")
        self.init_model()
        temp_state_dict= {}
        state_dict = torch.load(model)['model']
        for k, v in state_dict.items():
            if k.startswith('extras'):
                if 'weight' in k:
                    new_key = k.split('weight')[0] + 'conv_q.conv.weight'
                elif 'bias' in k:
                    new_key = k.split('bias')[0] + 'conv_q.conv.bias'
                temp_state_dict[new_key] = v
            elif 'headers' in k and 'weight' in k:
                new_key = k.split('weight')[0]+'conv_q.conv.weight'
                temp_state_dict[new_key] = v
            elif 'headers'in k and 'bias' in k:
                new_key = k.split('bias')[0]+'conv_q.conv.bias'
                temp_state_dict[new_key] = v
        model_dict = self.state_dict()
        self.load_state_dict(temp_state_dict, strict=False)

    def init_model(self):
        for _mod in [self.base_net, self.extras, self.classification_headers, self.regression_headers]:
            for m in _mod.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    stdv = 1. / math.sqrt(m.weight.size(1))
                    m.weight.data.uniform_(-stdv, stdv)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
