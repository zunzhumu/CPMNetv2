import os
import SimpleITK as sitk
import scipy.ndimage as ndimage
import skimage.measure as measure
import numpy as np
import pandas as pd
import tqdm
import random

def GetBoundingBox_From_Coords(coords:list, spacing:list)-> float:
    '''
    coords is a list in [[z, y, x], [z, y, x], ....] type
    spacing is a list for x, y, z order
    '''
    coords = np.array(coords)
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    z_spacing, y_spacing, x_spacing = spacing[2], spacing[1], spacing[0]
    z_min, z_max = min(z), max(z)
    y_min, y_max = min(y), max(y)
    x_min, x_max = min(x), max(x)
    center_z, center_y, center_x = (z_min + z_max)/2, (y_min + y_max)/2, (x_min + x_max)/2
    d, h, w = (z_max - z_min + 1) * z_spacing, (y_max - y_min + 1) * y_spacing, (x_max - x_min + 1) * x_spacing
    return center_z, center_y, center_x, d, h, w

train_image_path = 'imagesTs'
train_label_path = 'labelsTs'
image_list = os.listdir(train_image_path)
data_information = []
for i in tqdm.tqdm(image_list):
    image = sitk.ReadImage(os.path.join(train_image_path, i)) # x, y, z
    spacing = image.GetSpacing()
    mask = sitk.ReadImage(os.path.join(train_label_path, i.replace('_0000.nii.gz', '.nii.gz')))
    mask_arr = sitk.GetArrayFromImage(mask)
    labeled_array, num_features = ndimage.measurements.label(mask_arr, structure=ndimage.generate_binary_structure(3,3))
    region = measure.regionprops(labeled_array)
    for j in range(num_features):
        coords = region[j].coords
        z, y, x, d, h, w = GetBoundingBox_From_Coords(coords, spacing)
        data_information.append([i, x, y, z, w, h, d, 'ribfra'])
df = pd.DataFrame(data=data_information, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'lesion'])
df.to_csv('train.csv', index=False)