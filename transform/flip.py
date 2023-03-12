# -*- coding: utf-8 -*-
from __future__ import print_function, division
import random
import numpy as np
from .abstract_transform import AbstractTransform


class RandomFlip(AbstractTransform):
    """ random flip the image (shape [C, D, H, W] or [C, H, W]) """

    def __init__(self, flip_depth=True, flip_height=True, flip_width=True,p=0.5):
        """
            flip_depth (bool) : random flip along depth axis or not, only used for 3D images
            flip_height (bool): random flip along height axis or not
            flip_width (bool) : random flip along width axis or not
        """
        self.flip_depth = flip_depth
        self.flip_height = flip_height
        self.flip_width = flip_width
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        flip_axis = []
        if self.flip_width:
            if random.random() < self.p:
                flip_axis.append(-1)
        if self.flip_height:
            if random.random() < self.p:
                flip_axis.append(-2)
        if input_dim == 3 and self.flip_depth:
            if random.random() < self.p:
                flip_axis.append(-3)

        if len(flip_axis) > 0:
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            image_t = np.flip(image, flip_axis).copy()
            sample['image'] = image_t

            if 'ctr' in sample:
                coord = sample['ctr'].copy()
                for axis in flip_axis:
                    coord[:, axis] = input_shape[axis] - 1 - coord[:, axis]
                sample['ctr'] = coord

        return sample



class RandomMaskFlip(AbstractTransform):
    """ random flip the image (shape [C, D, H, W] or [C, H, W]) """

    def __init__(self, flip_depth=True, flip_height=True, flip_width=True,p=0.5):
        """
            flip_depth (bool) : random flip along depth axis or not, only used for 3D images
            flip_height (bool): random flip along height axis or not
            flip_width (bool) : random flip along width axis or not
        """
        self.flip_depth = flip_depth
        self.flip_height = flip_height
        self.flip_width = flip_width
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        flip_axis = []
        if self.flip_width:
            if random.random() < self.p:
                flip_axis.append(-1)
        if self.flip_height:
            if random.random() < self.p:
                flip_axis.append(-2)
        if input_dim == 3 and self.flip_depth:
            if random.random() < self.p:
                flip_axis.append(-3)

        if len(flip_axis) > 0:
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            image_t = np.flip(image, flip_axis).copy()
            mask = sample['mask']
            mask_t = np.flip(mask, flip_axis).copy()
            sample['image'] = image_t
            sample['mask'] = mask_t
        
        return sample
