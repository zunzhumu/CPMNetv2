# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
import random
from .abstract_transform import AbstractTransform
from .image_process import *


class RandomCrop0(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W])
    """

    def __init__(self, output_size):
        """

        """
        self.output_size = output_size

        assert isinstance(self.output_size, (list, tuple))

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1

        assert (input_dim == len(self.output_size))
        crop_margin = [input_shape[i + 1] - self.output_size[i] \
                       for i in range(input_dim)]

        bb_min = [0] * (input_dim + 1)
        bb_max = image.shape
        bb_min, bb_max = bb_min[1:], bb_max[1:]
        crop_min = [random.randint(bb_min[i], bb_max[i]) - int(self.output_size[i] / 2) \
                    for i in range(input_dim)]
        crop_min = [max(0, item) for item in crop_min]
        crop_min = [min(crop_min[i], input_shape[i + 1] - self.output_size[i]) for i in range(input_dim)]

        crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]

        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        sample['image'] = image_t

        if 'ctr' in sample:
            sample['ctr'] = sample['ctr'].copy() - crop_min[1:]

        return sample


class RandomCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W])
    """

    def __init__(self, output_size, pos_ratio=0, label_down=4):
        """

        """
        self.output_size = output_size
        self.pos_ratio = pos_ratio
        self.label_down = label_down

        assert isinstance(self.output_size, (list, tuple))

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1

        assert (input_dim == len(self.output_size))
        crop_margin = [input_shape[i + 1] - self.output_size[i] \
                       for i in range(input_dim)]

        bb_min = [0] * (input_dim + 1)
        bb_max = image.shape
        bb_min, bb_max = bb_min[1:], bb_max[1:]

        if self.pos_ratio > 0 and sample['ctr'].size > 0:
            bb_min = sample['ctr'].min(0) - np.array(self.output_size) + 10
            bb_max = sample['ctr'].max(0) + np.array(self.output_size) - 10
            bb_min = np.clip(bb_min, a_min=0, a_max=None).astype('int16')
            bb_max = np.clip(bb_max, a_min=None, a_max=image.shape[1:]).astype('int16')

        crop_min = [random.randint(bb_min[i], max(bb_min[i], bb_max[i] - int(self.output_size[i]))) for i in
                    range(input_dim)]
        crop_min = [min(crop_min[i], input_shape[i + 1] - self.output_size[i]) for i in range(input_dim)]

        crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]

        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        sample['image'] = image_t

        crop_max_label = [ind // self.label_down for ind in crop_max]
        crop_max_label[0] = crop_max[0]
        crop_min_label = [ind // self.label_down for ind in crop_min]
        crop_min_label[0] = crop_min[0]

        if 'ctr' in sample:
            sample['ctr'] = sample['ctr'].copy() - crop_min[1:]


        return sample



class RandomMaskCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W])
    """

    def __init__(self, output_size, pos_ratio=0, label_down=4):
        """

        """
        self.output_size = output_size
        self.pos_ratio = pos_ratio
        self.label_down = label_down

        assert isinstance(self.output_size, (list, tuple))

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        input_shape = image.shape
        input_dim = len(input_shape) - 1

        assert (input_dim == len(self.output_size))
        crop_margin = [input_shape[i + 1] - self.output_size[i] \
                       for i in range(input_dim)]

        bb_min = [0] * (input_dim + 1)
        bb_max = image.shape
        bb_min, bb_max = bb_min[1:], bb_max[1:]

        ctr = np.array(image.shape[1:]) / 2

        if self.pos_ratio > 0:
            bb_min = ctr - np.array(self.output_size) + 10
            bb_max = ctr + np.array(self.output_size) - 10
            bb_min = np.clip(bb_min, a_min=0, a_max=None).astype('int16')
            bb_max = np.clip(bb_max, a_min=None, a_max=image.shape[1:]).astype('int16')

        crop_min = [random.randint(bb_min[i], max(bb_min[i], bb_max[i] - int(self.output_size[i]))) for i in
                    range(input_dim)]
        crop_min = [min(crop_min[i], input_shape[i + 1] - self.output_size[i]) for i in range(input_dim)]

        crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]

        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        mask_t = crop_ND_volume_with_bounding_box(mask, crop_min, crop_max)
        sample['image'] = image_t
        sample['mask'] = mask_t
        return sample
