# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
from .abstract_transform import AbstractTransform
from .image_process import *
import random


class RandomRescale(AbstractTransform):
    """Rescale the image in a sample to a given size."""

    def __init__(self, scale_range, label_down=4, p=0.3):
        """

        """
        self.scale_range = scale_range
        self.label_down = label_down
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1

        if random.random() < self.p:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            scale = [1] + [scale] * 3
            image_t = ndimage.interpolation.zoom(image, scale, order=1)

            sample['image'] = image_t

            if 'ctr' in sample:
                sample['ctr'] = sample['ctr'].copy() * scale[1:]

        return sample
