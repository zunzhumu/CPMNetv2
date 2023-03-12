# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
from .abstract_transform import AbstractTransform
from .image_process import *
import random


def rotate_vecs_3d(vec, angle, axis):
    rad = np.deg2rad(angle)
    rotated_vec = vec.copy()
    rotated_vec[::, axis[0]] = vec[::, axis[0]] * np.cos(rad) - vec[::, axis[1]] * np.sin(rad)
    rotated_vec[::, axis[1]] = vec[::, axis[0]] * np.sin(rad) + vec[::, axis[1]] * np.cos(rad)
    return rotated_vec


class RandomRotate(AbstractTransform):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W]) 
    """

    def __init__(self, angle_range_d, angle_range_h, angle_range_w, only_one=True, reshape=True, p=0.3):
        """
        angle_range_d (tuple/list/None) : rorate angle range along depth axis (degree),
               only used for 3D images
        angle_range_h (tuple/list/None) : rorate angle range along height axis (degree)
        angle_range_w (tuple/list/None) : rorate angle range along width axis (degree)
        """
        self.angle_range_d = angle_range_d
        self.angle_range_h = angle_range_h
        self.angle_range_w = angle_range_w
        self.only_one = only_one
        self.reshape = reshape
        self.p = p

    def __apply_transformation(self, image, transform_param_list, order=1, cval=0):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            image = ndimage.rotate(image, angle, axes, reshape=self.reshape, order=order, cval=cval)

        return image

    def __apply_transformation_coord(self, image, coord, transform_param_list, order=1, cval=0):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            org_center = (np.array(image.shape[-3:]) - 1) / 2.
            image = ndimage.rotate(image, angle, axes, reshape=self.reshape, order=order, cval=cval)
            rot_center = (np.array(image.shape[-3:]) - 1) / 2.

            org = coord - org_center
            new = rotate_vecs_3d(org, angle, axes)
            coord = new + rot_center

        return image, coord

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1

        transform_param_list = []

        if (self.angle_range_d is not None) and random.random() < self.p:
            angle_d = np.random.uniform(self.angle_range_d[0], self.angle_range_d[1])
            transform_param_list.append([angle_d, (-2, -1)])
        if (self.angle_range_h is not None) and random.random() < self.p:
            angle_h = np.random.uniform(self.angle_range_h[0], self.angle_range_h[1])
            transform_param_list.append([angle_h, (-3, -1)])
        if (self.angle_range_w is not None) and random.random() < self.p:
            angle_w = np.random.uniform(self.angle_range_w[0], self.angle_range_w[1])
            transform_param_list.append([angle_w, (-3, -2)])

        if self.only_one and len(transform_param_list) > 0:
            transform_param_list = random.sample(transform_param_list, 1)

        if len(transform_param_list) > 0:
            if 'ctr' in sample:
                image_t, coord = self.__apply_transformation_coord(image, sample['ctr'].copy(), transform_param_list,
                                                                   1)
                sample['ctr'] = coord
            else:
                image_t = self.__apply_transformation(image, transform_param_list, 1)
            sample['image'] = image_t

        return sample


class RandomTranspose(AbstractTransform):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W])
    """

    def __init__(self, trans_xy=True, trans_zx=False, trans_zy=False, p=0.5):
        self.trans_xy = trans_xy
        self.trans_zx = trans_zx
        self.trans_zy = trans_zy
        self.p = p

    def __call__(self, sample):
        transpose_list = []

        if self.trans_zy and random.random() < self.p:
            transpose_list.append((0, 2, 1, 3))
        if self.trans_xy and random.random() < self.p:
            transpose_list.append((0, 1, 3, 2))
        if self.trans_zx and random.random() < self.p:
            transpose_list.append((0, 3, 2, 1))

        if len(transpose_list) > 0:
            ctr_t = sample['ctr'].copy()
            image_t = sample['image']
            for transpose in transpose_list:
                temp = ctr_t.copy()
                ctr_t[:, 0] = temp[:, transpose[1] - 1]
                ctr_t[:, 1] = temp[:, transpose[2] - 1]
                ctr_t[:, 2] = temp[:, transpose[3] - 1]
                image_t = np.transpose(image_t, transpose)

            sample['image'] = image_t
            sample['ctr'] = ctr_t

        return sample



class RandomMaskTranspose(AbstractTransform):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W])
    """

    def __init__(self, trans_xy=True, trans_zx=False, trans_zy=False, p=0.5):
        self.trans_xy = trans_xy
        self.trans_zx = trans_zx
        self.trans_zy = trans_zy
        self.p = p

    def __call__(self, sample):
        transpose_list = []

        if self.trans_zy and random.random() < self.p:
            transpose_list.append((0, 2, 1, 3))
        if self.trans_xy and random.random() < self.p:
            transpose_list.append((0, 1, 3, 2))
        if self.trans_zx and random.random() < self.p:
            transpose_list.append((0, 3, 2, 1))

        if len(transpose_list) > 0:
            image_t = sample['image']
            mask_t = sample['mask']
            for transpose in transpose_list:
                image_t = np.transpose(image_t, transpose)
                mask_t = np.transpose(mask_t, transpose)

            sample['image'] = image_t
            sample['mask'] = mask_t

        return sample


class RandomMaskRotate(AbstractTransform):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W]) 
    """

    def __init__(self, angle_range_d, angle_range_h, angle_range_w, reshape=True, p=0.3):
        """
        angle_range_d (tuple/list/None) : rorate angle range along depth axis (degree),
               only used for 3D images
        angle_range_h (tuple/list/None) : rorate angle range along height axis (degree)
        angle_range_w (tuple/list/None) : rorate angle range along width axis (degree)
        """
        self.angle_range_d = angle_range_d
        self.angle_range_h = angle_range_h
        self.angle_range_w = angle_range_w
        self.reshape = reshape
        self.p = p

    def __apply_transformation(self, image, transform_param_list, order=1, cval=0):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            image = ndimage.rotate(image, angle, axes, reshape=self.reshape, order=order, cval=cval)
        return image

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        transform_param_list = []

        if (self.angle_range_d is not None) and random.random() < self.p:
            angle_d = np.random.uniform(self.angle_range_d[0], self.angle_range_d[1])
            transform_param_list.append([angle_d, (-2, -1)])
        if (self.angle_range_h is not None) and random.random() < self.p:
            angle_h = np.random.uniform(self.angle_range_h[0], self.angle_range_h[1])
            transform_param_list.append([angle_h, (-3, -1)])
        if (self.angle_range_w is not None) and random.random() < self.p:
            angle_w = np.random.uniform(self.angle_range_w[0], self.angle_range_w[1])
            transform_param_list.append([angle_w, (-3, -2)])

        if len(transform_param_list) > 0:
            image_t = self.__apply_transformation(image, transform_param_list, 1)
            mask_t = self.__apply_transformation(mask, transform_param_list, 0)
            sample['image'] = image_t
            sample['mask'] = mask_t

        return sample