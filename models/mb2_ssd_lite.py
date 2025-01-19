from .base import QuantInvertedResBlock, Qconv_Seperable, Qconv_MERE
from .ssd import SSD
from .predictor import Predictor
#from .config import mobilenetv1_ssd_config as config # KJ Commented
from torch.nn import Sequential, ModuleList


def create_mobilenetv2_ssd_lite_aug(_base_net, num_classes, wbit=8, abit=8, head_wbit=8, head_abit=8, config=None, full_pretrain=False):
    base_net = _base_net  # disable dropout layer

    source_layer_indexes = [
        12,
        17,
    ]
    extras = ModuleList([
        Sequential(
            QuantInvertedResBlock(1280, 512, kernel=3, stride=2, expansion=0.2, wbit=wbit, abit=abit, full_pretrain=full_pretrain),
        ),
        Sequential(
            QuantInvertedResBlock(512, 256, kernel=3, stride=2, expansion=0.25, wbit=wbit, abit=abit, full_pretrain=full_pretrain),
        ),
        Sequential(
            QuantInvertedResBlock(256, 256, kernel=3, stride=2, expansion=0.5, wbit=wbit, abit=abit, full_pretrain=full_pretrain),
        ),
        Sequential(
            QuantInvertedResBlock(256, 64, kernel=3, stride=2, expansion=0.25, wbit=wbit, abit=abit, full_pretrain=full_pretrain),
        )
    ])
    regression_headers = ModuleList([
        QuantInvertedResBlock(576, 24, kernel=3, stride=1, expansion=1, wbit=head_wbit, abit=head_abit, full_pretrain=full_pretrain),
        QuantInvertedResBlock(1280, 24, kernel=3, stride=1, expansion=1, wbit=head_wbit, abit=head_abit, full_pretrain=full_pretrain),
        QuantInvertedResBlock(512, 24, kernel=3, stride=1,  expansion=1, wbit=head_wbit, abit=head_abit, full_pretrain=full_pretrain),
        QuantInvertedResBlock(256, 24, kernel=3, stride=1,  expansion=1, wbit=head_wbit, abit=head_abit, full_pretrain=full_pretrain),
        QuantInvertedResBlock(256, 24, kernel=3, stride=1,  expansion=1, wbit=head_wbit, abit=head_abit, full_pretrain=full_pretrain),
        Qconv_MERE(64, 24, kernel=1, stride=1, padding=0, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
    ])

    classification_headers = ModuleList([
        QuantInvertedResBlock(576, 6*num_classes, kernel=3, stride=1, expansion=1, wbit=head_wbit, abit=head_abit, full_pretrain=full_pretrain),
        QuantInvertedResBlock(1280, 6*num_classes, kernel=3, stride=1, expansion=1, wbit=head_wbit, abit=head_abit, full_pretrain=full_pretrain),
        QuantInvertedResBlock(512, 6*num_classes, kernel=3, stride=1,  expansion=1, wbit=head_wbit, abit=head_abit, full_pretrain=full_pretrain),
        QuantInvertedResBlock(256, 6*num_classes, kernel=3, stride=1,  expansion=1, wbit=head_wbit, abit=head_abit, full_pretrain=full_pretrain),
        QuantInvertedResBlock(256, 6*num_classes, kernel=3, stride=1,  expansion=1, wbit=head_wbit, abit=head_abit, full_pretrain=full_pretrain),
        Qconv_MERE(64,  6*num_classes, kernel=1, stride=1, padding=0, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
    ])


    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, config=config)


def create_mobilenetv2_ssd_predictor(net, config, candidate_size=200, nms_method=None, sigma=0.5):
    predictor = Predictor(net, config["image_size"], config["image_mean"],
                          config["image_std"],
                          nms_method=nms_method,
                          iou_threshold=config["iou_threshold"],
                          candidate_size=candidate_size,
                          sigma=sigma
                        )

    return predictor
