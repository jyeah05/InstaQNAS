from .base import QconvBlock_BIAS, Qconv_MERE, Qconv_Seperable_MBv1
from .ssd import SSD
from .predictor import Predictor
#from .config import mobilenetv1_ssd_config as config # KJ Commented
from torch.nn import Sequential, ModuleList


def create_mobilenetv1_ssd_lite_aug(_base_net, num_classes, wbit=8, abit=8, head_wbit=8, head_abit=8, config=None, full_pretrain=False):
    base_net = _base_net  # disable dropout layer

    source_layer_indexes = [
        10,
        12,
    ]
    extras = ModuleList([
        Sequential(
            QconvBlock_BIAS(1024, 256, kernel=1, stride=1, padding=0, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
            Qconv_Seperable_MBv1(256, 512, kernel=3, stride=2, padding=1, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        ),
        Sequential(
            QconvBlock_BIAS(512, 128, kernel=1, stride=1, padding=0, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
            Qconv_Seperable_MBv1(128, 256, kernel=3, stride=2, padding=1, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        ),
        Sequential(
            QconvBlock_BIAS(256, 128, kernel=1, stride=1, padding=0, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
            Qconv_Seperable_MBv1(128, 256, kernel=3, stride=2, padding=1, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        ),
        Sequential(
            QconvBlock_BIAS(256, 128, kernel=1, stride=1, padding=0, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
            Qconv_Seperable_MBv1(128, 256, kernel=3, stride=2, padding=1, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        )
    ])

    regression_headers = ModuleList([
        Qconv_Seperable_MBv1(512, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        Qconv_Seperable_MBv1(1024, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        Qconv_Seperable_MBv1(512, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        Qconv_Seperable_MBv1(256, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        Qconv_Seperable_MBv1(256, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        Qconv_MERE(256, 24, kernel=1, stride=1, padding=0, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
    ])

    classification_headers = ModuleList([
        Qconv_Seperable_MBv1(512, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        Qconv_Seperable_MBv1(1024, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        Qconv_Seperable_MBv1(512, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        Qconv_Seperable_MBv1(256, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        Qconv_Seperable_MBv1(256, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
        Qconv_MERE(256, 6* num_classes, kernel=1, stride=1, padding=0, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain),
    ])
    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, config=config)


def create_mobilenetv1_ssd_predictor(net, config, candidate_size=200, nms_method=None, sigma=0.5):
    predictor = Predictor(net, config["image_size"], config["image_mean"],
                          config["image_std"],
                          nms_method=nms_method,
                          iou_threshold=config["iou_threshold"],
                          candidate_size=candidate_size,
                          sigma=sigma
                        )

    return predictor
