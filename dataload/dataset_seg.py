# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import torch
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from torch.utils.data import Dataset

lesion_label_default = ['aneurysm']

class DetDatasetPath(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]

    Arguments:
        roots (list): list of dirs of the dataset
        transform_post: transform object after cropping
        crop_fn: cropping function
        lesion_label (list): label names of lesion, such as ['Aneurysm']

    """

    def __init__(self, root, transform_post=None, train=False, target_shape=(16, 64, 64)):

        self.image_list = []
        id_list = os.listdir(root)
        if train:
            self.image_list = [os.path.join(root, id) for id in id_list if 'image' in id]
        else:
            self.image_list = [os.path.join(root, id) for id in id_list if 'image_gt' in id]

        self.transform_post = transform_post
        self.target_shape = target_shape


    def __len__(self):
        return len(self.image_list)
    
    def __rescale__(self, image, mask, target_shape):
        current_shape = image.shape
        rescale = [t/c for t, c in zip(target_shape, current_shape)]
        image_r = ndimage.interpolation.zoom(image, rescale, order=1)
        mask_r = ndimage.interpolation.zoom(mask, rescale, order=0)
        return image_r, mask_r 

    def __getitem__(self, idx):
        image_dir = self.image_list[idx]
        if 'image_gt' in image_dir.split('/')[-1]:
            label_dir  = image_dir.replace('image_gt', 'label_gt')
        elif 'image_pred' in image_dir.split('/')[-1]:
            label_dir  = image_dir.replace('image_pred', 'label_pred')
        image = sitk.ReadImage(image_dir)
        image = sitk.GetArrayFromImage(image).astype('float32')# z, y, x
        mask = sitk.ReadImage(label_dir)
        mask = sitk.GetArrayFromImage(mask)
        image, mask = self.__rescale__(image, mask, self.target_shape)
        sample = {}
        sample['image'] = image[np.newaxis, ...]
        sample['mask'] = mask[np.newaxis, ...]
        if self.transform_post:
                sample = self.transform_post(sample)

        return sample



def collate_fn_dict(batches):
    batch = []
    [batch.extend(b) for b in batches]
    imgs = [s['image'] for s in batch]
    imgs = np.stack(imgs)
    annots = [s['annot'] for s in batch]
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = np.ones((len(annots), max_num_annots, 7), dtype='float32') * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = np.ones((len(annots), 1, 7), dtype='float32') * -1

    return {'image': torch.tensor(imgs), 'annot': torch.tensor(annot_padded)}
