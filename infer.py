# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import math
import os
import logging
import numpy as np
from torch import nn
import tqdm
###network###
from networks.ResNet_3D_CPM import resnet18, Detection_Postprocess
###data###
from dataload.split_combine import SplitComb
###postprocessing###
from utils.box_utils import nms_3D
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage as ndimage
import skimage.measure as measure

parser = argparse.ArgumentParser(description='anchorfree_3D_aneurysm_infer')
parser.add_argument('--batch-size', type=int, default=6, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--no-cuda', default=False,help='enables CUDA training')
parser.add_argument('--load', default=True, help='enables load weights')
parser.add_argument('--load_model', type=str, default='', metavar='str', help='the parameters of net')
parser.add_argument('--num_workers', type=int, default=4, metavar='S', help='num_workers (default: 1)')
parser.add_argument('--root', type=str, default='', metavar='str', help='folder that contains data (default: test dataset)')
parser.add_argument('--save_dir', type=str, default='', metavar='str', help='folder that save results')
parser.add_argument('--norm_type', type=str, default='batchnorm', metavar='N', help='norm type of backbone')
parser.add_argument('--head_norm', type=str, default='batchnorm', metavar='N', help='norm type of head')
parser.add_argument('--act_type', type=str, default='ReLU', metavar='N', help='act type of network')
parser.add_argument('--se', default=False, help='using se block')
parser.add_argument('--gpu', type=str, default='0', metavar='N',help='use gpu')
parser.add_argument('--log', type=str, default="logs/infer.log",help='save training log to file')
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print("The number of GPUs:", torch.cuda.device_count())
# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# set a log
if args.log:
    # crate a fileHandler
    fh = logging.FileHandler(args.log)
logger.addHandler(fh)

# CROP_SIZE = [48, 192, 192]
# OVERLAP_SIZE = [12, 48, 48]
CROP_SIZE = [64, 128, 128]
OVERLAP_SIZE = [16, 32, 32]

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('save_dir:', args.save_dir)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if args.cuda else {}

logger.info('The batch size:{}'.format(args.batch_size))
logger.info('norm type:{}, head norm:{}, act_type:{}, using se block:{}'
.format(args.norm_type, args.head_norm, args.act_type, args.se))
###############  ###bulid model###################
model = resnet18(n_channels=1, n_blocks=[2, 3, 3, 3], n_filters=[64, 96, 128, 160], stem_filters=32,norm_type=args.norm_type, 
                head_norm=args.head_norm, act_type=args.act_type, se=args.se, first_stride=(1, 2, 2))
# set threshold 0.8 for 1.5 FPs
detection_postprocess = Detection_Postprocess(topk=60, threshold=0.8, nms_threshold=0.05, num_topk=20, crop_size=CROP_SIZE)

# model = nn.DataParallel(model)
model.to(device)

if args.load:
    model.load_state_dict(torch.load(args.load_model))
    print("load model successful", args.load_model)


def GetBoundingBox_From_Coords(coords:list)-> float:
    '''
    coords is a list in [[z, y, x], [z, y, x], ....] type
    spacing is a list for x, y, z order
    '''
    coords = np.array(coords)
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    z_min, z_max = min(z), max(z)
    y_min, y_max = min(y), max(y)
    x_min, x_max = min(x), max(x)
    center_z, center_y, center_x = (z_min + z_max)/2, (y_min + y_max)/2, (x_min + x_max)/2
    d, h, w = (z_max - z_min + 1), (y_max - y_min + 1), (x_max - x_min + 1)
    return center_z, center_y, center_x, d, h, w


def norm(data):
    max_value = np.percentile(data, 99)
    min_value = 0.
    data[data>max_value] = max_value
    data[data<min_value] = min_value
    data = data/max_value
    return data

def findCube(z, y, x, d, h, w, shape, stride=4):
    ## extend edge
    # d, h, w = round(d+20), round(h+40), round(w+40)
    ## extend edge
    d, h, w = math.ceil(2 * d), round(2 * h), round(2 * w)
    d, h, w = math.ceil(d / stride) * stride, math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
    z1 = max(round(z-d/2), 0)
    y1 = max(round(y-h/2), 0)
    x1 = max(round(x-w/2), 0)

    z2 = min(round(z+d/2), shape[0])
    y2 = min(round(y+h/2), shape[1])
    x2 = min(round(x+w/2), shape[2])

    return z1, y1, x1, z2, y2, x2


def infer(save_dir, model, ct_dir, label_dir):
    global N
    model.eval().to(device)
    split_comber = SplitComb(crop_size=CROP_SIZE, overlap=OVERLAP_SIZE, pad_value=-1)
    simage = sitk.ReadImage(ct_dir)
    # spacing = torch.from_numpy(np.array(simage.GetSpacing()[::-1])) # z, y, x
    image = sitk.GetArrayFromImage(simage).astype('float32')# z, y, x
    image = norm(image) # normalized
    # convert to -1 ~ 1  note ste pad_value to -1 for SplitComb
    image = image * 2.0 - 1.0
    # split_images [N, 1, crop_z, crop_y, crop_x]
    split_images, nzhw = split_comber.split(image)
    batch_size = 8 * args.batch_size
    top_k = 40
    data = torch.from_numpy(split_images)
    outputlist = []
    for i in range(int(math.ceil(data.size(0) / batch_size))):
        end = (i+1) * batch_size
        if end > data.size(0):
            end = data.size(0)
        input = data[i*batch_size:end].to(device)
        with torch.no_grad():
            output = model(input)
            output = detection_postprocess(output, device=device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
        outputlist.append(output.data.cpu().numpy())
    output = np.concatenate(outputlist, 0)
    output = split_comber.combine(output, nzhw=nzhw)
    output = torch.from_numpy(output).view(-1, 8)
    object_ids = output[:, 0] != -1.0
    output = output[object_ids]
    if len(output) > 0:
        keep = nms_3D(output[:, 1:], overlap=0.05, top_k=top_k)
        output = output[keep]
    output = output.numpy()


    mask = sitk.ReadImage(label_dir)
    mask_arr = sitk.GetArrayFromImage(mask)
    shape = mask_arr.shape
    labeled_array, num_features = ndimage.measurements.label(mask_arr, structure=ndimage.generate_binary_structure(3,3))
    region = measure.regionprops(labeled_array)
    for j in range(num_features):
        coords = region[j].coords
        z, y, x, d, h, w = GetBoundingBox_From_Coords(coords)
        z1, y1, x1, z2, y2, x2 = findCube(z, y, x, d, h, w, shape)
        crop_image = image[z1:z2, y1:y2, x1:x2]
        crop_mask = mask_arr[z1:z2, y1:y2, x1:x2]
        crop_image = sitk.GetImageFromArray(crop_image)
        crop_mask = sitk.GetImageFromArray(crop_mask)
        sitk.WriteImage(crop_image, os.path.join(save_dir, 'image_gt_{}.nii.gz'.format(N)))
        sitk.WriteImage(crop_mask, os.path.join(save_dir, 'label_gt_{}.nii.gz'.format(N)))
        N += 1


    for s in range(len(output)):
        p_z, p_y, p_x, p_d, p_h, p_w = output[s][2:]
        z1, y1, x1, z2, y2, x2 = findCube(p_z, p_y, p_x, p_d, p_h, p_w, shape)
        crop_image = image[z1:z2, y1:y2, x1:x2]
        crop_mask = mask_arr[z1:z2, y1:y2, x1:x2]
        crop_image = sitk.GetImageFromArray(crop_image)
        crop_mask = sitk.GetImageFromArray(crop_mask)
        sitk.WriteImage(crop_image, os.path.join(save_dir, 'image_pred_{}.nii.gz'.format(N)))
        sitk.WriteImage(crop_mask, os.path.join(save_dir, 'label_pred_{}.nii.gz'.format(N)))
        N += 1

if __name__ == '__main__':
    images_root = '/home/datas/imagesTs'
    labels_root = '/home/datas/labelsTs'
    N = 1
    for i in tqdm.tqdm(os.listdir(images_root)):
        image_dir = os.path.join(images_root, i)
        label_dir = os.path.join(labels_root, i.replace('_0000.nii.gz', '.nii.gz'))
        infer(args.save_dir, model, image_dir, label_dir)

