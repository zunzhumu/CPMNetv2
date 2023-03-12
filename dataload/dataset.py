# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

lesion_label_default = ['aneurysm']

class DetDatasetCSVR(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]

    Arguments:
        roots (list): list of dirs of the dataset
        transform_post: transform object after cropping
        crop_fn: cropping function
        lesion_label (list): label names of lesion, such as ['Aneurysm']

    """

    def __init__(self, roots, transform_post=None, crop_fn=None, lesion_label=None, csv_file='./train.csv'):

        if lesion_label is None:
            self.lesion_label = lesion_label_default
        else:
            self.lesion_label = lesion_label

        self.ct_list = []
        self.csv_list = []
        self.csv_file = np.array(pd.read_csv(csv_file))
        for root in roots:
            id_list = os.listdir(root)
            ct_list = [os.path.join(root, id) for id in id_list]
            csv_list = [self.csv_file[self.csv_file[:, 0] == id, 1:] for id in id_list]
            self.ct_list.extend(ct_list)
            self.csv_list.extend(csv_list)

        self.transform_post = transform_post
        self.crop_fn = crop_fn

    def __len__(self):
        return len(self.ct_list)
    
    def __norm__(self, data):
        max_value = np.percentile(data, 99)
        min_value = 0.
        data[data>max_value] = max_value
        data[data<min_value] = min_value
        data = data/max_value
        return data

    def __getitem__(self, idx):
        ct_dir = self.ct_list[idx]
        # print('ct dir:', ct_dir)
        # inputImage = sitk.ReadImage(ct_dir)
        image = sitk.ReadImage(ct_dir)
        # N4 bias Field Correction
        # maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
        # inputImage = sitk.Cast(inputImage,sitk.sitkFloat32)
        # corrector = sitk.N4BiasFieldCorrectionImageFilter()
        # image = corrector.Execute(inputImage,maskImage)

        image_spacing = image.GetSpacing()[::-1] # z, y, x
        image = sitk.GetArrayFromImage(image).astype('float32')# z, y, x
        image = self.__norm__(image) # normalized
        csv_label = self.csv_list[idx]
        all_loc = csv_label[:, 0:3].astype('float32') # x,y,z
        all_loc = all_loc[:,::-1] # convert z,y,x
        all_rad = csv_label[:, 3:6].astype('float32') # w,h,d
        all_rad = all_rad[:,::-1] # convert d,h,w
        lesion_index = np.sum([csv_label[:, -1] == label for label in self.lesion_label], axis=0, dtype='bool')
        all_cls = np.ones(shape=(all_loc.shape[0]), dtype='int8') * (-1)
        all_cls[lesion_index] = 0

        data = {}
        data['image'] = image
        data['all_loc'] = all_loc
        data['all_rad'] = all_rad
        data['all_cls'] = all_cls
        data['file_name'] = self.ct_list[idx]
        samples = self.crop_fn(data, image_spacing)
        random_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            if self.transform_post:
                sample = self.transform_post(sample)
            sample['image'] = (sample['image'] * 2.0 - 1.0) # normalized to -1 ~ 1
            random_samples.append(sample)

        return random_samples



class DetDatasetCSVRTest(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]
    """

    def __init__(self, roots, SplitComb, csv_file='./val.csv', lesion_label=None):
        if lesion_label is None:
            self.lesion_label = lesion_label_default
        else:
            self.lesion_label = lesion_label
        self.ct_list = []
        self.csv_list = []
        self.splitcomb = SplitComb
        self.csv_file = np.array(pd.read_csv(csv_file))
        for root in roots:
            id_list = os.listdir(root)
            ct_list = [os.path.join(root, id) for id in id_list]
            csv_list = [self.csv_file[self.csv_file[:, 0] == id, 1:] for id in id_list]
            self.ct_list.extend(ct_list)
            self.csv_list.extend(csv_list)

    def __len__(self):
        return len(self.ct_list)
    
    def __norm__(self, data):
        max_value = np.percentile(data, 99)
        min_value = 0.
        data[data>max_value] = max_value
        data[data<min_value] = min_value
        data = data/max_value
        return data

    def __getitem__(self, idx):
        ct_dir = self.ct_list[idx]
        image = sitk.ReadImage(ct_dir)
        # inputImage = sitk.ReadImage(ct_dir)
        # # N4 bias Field Correction
        # maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
        # inputImage = sitk.Cast(inputImage,sitk.sitkFloat32)
        # corrector = sitk.N4BiasFieldCorrectionImageFilter()
        # image = corrector.Execute(inputImage,maskImage)

        image_spacing = image.GetSpacing()[::-1] # z, y, x
        image = sitk.GetArrayFromImage(image).astype('float32')# z, y, x
        image = self.__norm__(image) # normalized
        csv_label = self.csv_list[idx]
        all_loc = csv_label[:, 0:3].astype('float32') # x,y,z
        all_loc = all_loc[:,::-1] # convert z,y,x
        all_rad = csv_label[:, 3:6].astype('float32') # w,h,d
        all_rad = all_rad[:,::-1] # convert d,h,w
        lesion_index = np.sum([csv_label[:, -1] == label for label in self.lesion_label], axis=0, dtype='bool')

        all_cls = np.ones(shape=(all_loc.shape[0]), dtype='int8') * (-1)
        all_cls[lesion_index] = 0

        data = {}
        # convert to -1 ~ 1  note ste pad_value to -1 for SplitComb
        image = image * 2.0 - 1.0
        # split_images [N, 1, crop_z, crop_y, crop_x]
        split_images, nzhw = self.splitcomb.split(image)
        data['split_images'] = np.ascontiguousarray(split_images)
        # data['all_loc'] = np.ascontiguousarray(all_loc) # index z, y, x
        # data['all_rad'] = np.ascontiguousarray(all_rad) # mm    d, h, w
        # data['all_cls'] = np.ascontiguousarray(all_cls)
        data['file_name'] = self.ct_list[idx].split('/')[-1]
        data['nzhw'] = nzhw
        data['spacing'] = image_spacing

        return data



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
