import numpy as np

from box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors
from collections import OrderedDict

class generate_mb2_ssd_config():
    def __init__(self, image_size, iou_threshold=0.45, center_variance=0.1, size_variance=0.2):
        """Generate config for difference image size.
        """
        super(generate_mb2_ssd_config, self).__init__()

        if image_size == 112:
            self.specs = [
                SSDSpec(7, 16, SSDBoxSizes(22, 39), [2, 3]),
                SSDSpec(4, 28, SSDBoxSizes(39, 56), [2, 3]),
                SSDSpec(2, 56, SSDBoxSizes(56, 72), [2, 3]),
                SSDSpec(1, 112, SSDBoxSizes(72, 89), [2, 3]),
                SSDSpec(1, 112, SSDBoxSizes(89, 106), [2, 3]),
                SSDSpec(1, 112, SSDBoxSizes(106, 123), [2, 3])
            ]
        elif image_size == 224:
            self.specs = [
                SSDSpec(14, 16, SSDBoxSizes(44, 78), [2, 3]),
                SSDSpec(7, 32, SSDBoxSizes(78, 112), [2, 3]),
                SSDSpec(4, 56, SSDBoxSizes(112, 145), [2, 3]),
                SSDSpec(2, 112, SSDBoxSizes(145, 179), [2, 3]),
                SSDSpec(1, 224, SSDBoxSizes(179, 212), [2, 3]),
                SSDSpec(1, 224, SSDBoxSizes(212, 246), [2, 3])
            ]

        elif image_size == 300:
            self.specs = [
                SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
                SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
                SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
            ]

        else :
            print("## There is no config for input size {}".format(image_size))
            exit()

        self.priors = generate_ssd_priors(self.specs, image_size)

        self.image_size = image_size
        self.image_mean= np.array([0, 0, 0])  # RGB layout
        self.image_std = 1.0
        self.iou_threshold = iou_threshold
        self.center_variance = center_variance
        self.size_variance = size_variance
        ## make dict for matching widh SSD class input format
        self.config = OrderedDict()
        self.config['image_size'] = image_size
        self.config['image_mean']= self.image_mean
        self.config['image_std'] = self.image_std
        self.config['iou_threshold'] = self.iou_threshold
        self.config['center_variance'] = self.center_variance
        self.config['size_variance'] = self.size_variance
        self.config['specs'] = self.specs
        self.config['priors'] = self.priors

        print("## config for {} has generated.".format(image_size))
        print("## specs : \n", self.specs)
        #input("press enter")
