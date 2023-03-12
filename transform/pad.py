# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
import math
from .abstract_transform import AbstractTransform
from .image_process import *


class Pad(AbstractTransform):
    """
    Pad the image (shape [C, D, H, W] or [C, H, W]) to an new spatial shape, 
    the real output size will be max(image_size, output_size)
    """

    def __init__(self, output_size, ceil_mode=False):
        """
        output_size (tuple/list): the size along each spatial axis. 
        ceil_mode (bool): if true, the real output size is integer multiples of output_size.
        """
        self.output_size = output_size
        self.ceil_mode = ceil_mode

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        assert (len(self.output_size) == input_dim)
        if (self.ceil_mode):
            multiple = [int(math.ceil(float(input_shape[1 + i]) / self.output_size[i])) \
                        for i in range(input_dim)]
            output_size = [multiple[i] * self.output_size[i] \
                           for i in range(input_dim)]
        else:
            output_size = self.output_size
        margin = [max(0, output_size[i] - input_shape[1 + i]) for i in range(input_dim)]
        margin_lower = [int(margin[i] / 2) for i in range(input_dim)]
        margin_upper = [margin[i] - margin_lower[i] for i in range(input_dim)]
        pad = [(margin_lower[i], margin_upper[i]) for i in range(input_dim)]
        pad = tuple([(0, 0)] + pad)

        image_t = np.pad(image, pad, 'constant', constant_values=0) if (max(margin) > 0) else image

        sample['image'] = image_t

        if 'ctr' in sample:
            sample['ctr'] = sample['ctr'].copy() + margin_lower

        return sample


class MaskPad(AbstractTransform):
    """
    Pad the image (shape [C, D, H, W] or [C, H, W]) to an new spatial shape, 
    the real output size will be max(image_size, output_size)
    """

    def __init__(self, output_size, ceil_mode=False):
        """
        output_size (tuple/list): the size along each spatial axis. 
        ceil_mode (bool): if true, the real output size is integer multiples of output_size.
        """
        self.output_size = output_size
        self.ceil_mode = ceil_mode

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        assert (len(self.output_size) == input_dim)
        if (self.ceil_mode):
            multiple = [int(math.ceil(float(input_shape[1 + i]) / self.output_size[i])) \
                        for i in range(input_dim)]
            output_size = [multiple[i] * self.output_size[i] \
                           for i in range(input_dim)]
        else:
            output_size = self.output_size
        margin = [max(0, output_size[i] - input_shape[1 + i]) for i in range(input_dim)]
        margin_lower = [int(margin[i] / 2) for i in range(input_dim)]
        margin_upper = [margin[i] - margin_lower[i] for i in range(input_dim)]
        pad = [(margin_lower[i], margin_upper[i]) for i in range(input_dim)]
        pad = tuple([(0, 0)] + pad)

        image_t = np.pad(image, pad, 'constant', constant_values=0) if (max(margin) > 0) else image
        mask_t = np.pad(mask, pad, 'constant', constant_values=0) if (max(margin) > 0) else mask

        sample['image'] = image_t
        sample['mask'] = mask_t

        return sample