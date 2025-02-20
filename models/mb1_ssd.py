from .base import QconvBlock_BIAS, Qconv_MERE#, LQconvBlock_BIAS, LQconv_MERE
from .ssd import SSD
from .predictor import Predictor
#from .config import mobilenetv1_ssd_config as config # KJ Commented
from torch.nn import Sequential, ModuleList


def create_mobilenetv1_ssd_aug(_base_net, num_classes, wbit=8, abit=8, head_wbit=8, head_abit=8, config=None, full_pretrain=False, ActQ='PACT'):
    base_net = _base_net  # disable dropout layer

    source_layer_indexes = [
        10,
        12,
    ]
    extras = ModuleList([
        Sequential(
            QconvBlock_BIAS(1024, 256, kernel=1, stride=1, padding=0, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='extra1_1', ActQ=ActQ),
            QconvBlock_BIAS(256, 512, kernel=3, stride=2, padding=1, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='extra1_2', ActQ=ActQ),
        ),
        Sequential(
            QconvBlock_BIAS(512, 128, kernel=1, stride=1, padding=0, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='extra2_1', ActQ=ActQ),
            QconvBlock_BIAS(128, 256, kernel=3, stride=2, padding=1, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='extra2_2', ActQ=ActQ),
        ),
        Sequential(
            QconvBlock_BIAS(256, 128, kernel=1, stride=1, padding=0, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='extra3_1', ActQ=ActQ),
            QconvBlock_BIAS(128, 256, kernel=3, stride=2, padding=1, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='extra3_2', ActQ=ActQ),
        ),
        Sequential(
            QconvBlock_BIAS(256, 128, kernel=1, stride=1, padding=0, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='extra4_1', ActQ=ActQ),
            QconvBlock_BIAS(128, 256, kernel=3, stride=2, padding=1, wbit=wbit, abit=abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='extra4_2', ActQ=ActQ),
        )
    ])

    regression_headers = ModuleList([
        Qconv_MERE(512, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='reg1', ActQ=ActQ),
        Qconv_MERE(1024, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='reg2', ActQ=ActQ),
        Qconv_MERE(512, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='reg3', ActQ=ActQ),
        Qconv_MERE(256, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='reg4', ActQ=ActQ),
        Qconv_MERE(256, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='reg5', ActQ=ActQ),
        Qconv_MERE(256, 24, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='reg6', ActQ=ActQ),
    ])
    classification_headers = ModuleList([
        Qconv_MERE(512, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='cls1', ActQ=ActQ),
        Qconv_MERE(1024, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='cls2', ActQ=ActQ),
        Qconv_MERE(512, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='cls3', ActQ=ActQ),
        Qconv_MERE(256, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='cls4', ActQ=ActQ),
        Qconv_MERE(256, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='cls5', ActQ=ActQ),
        Qconv_MERE(256, 6 * num_classes, kernel=3, stride=1, padding=1, wbit=head_wbit, abit=head_abit, weight_only=False, same_padding=False, full_pretrain=full_pretrain, layer_name='cls6', ActQ=ActQ), # TODO: change to kernel_size=1, padding=0?
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
