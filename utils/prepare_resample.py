import numpy as np
import pandas as pd
import tqdm
import os
import SimpleITK as sitk

train_image_path = 'images_paint'
train_label_path = 'labels_paint'
save_image_path = 'imagesTs'
save_label_path = 'labelsTs'
image_list = os.listdir(train_image_path)


def resample_simg(itkimage:sitk.Image, newSpacing=(1.0, 1.0, 1.0), label=False)->sitk.Image:
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if label:
        resamplemethod=sitk.sitkNearestNeighbor
    else:
        resamplemethod=sitk.sitkLinear
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

target_spacing = (0.70507812,  0.70507812,  2.0)

for i in tqdm.tqdm(image_list):
    image = sitk.ReadImage(os.path.join(train_image_path, i)) # x, y, z
    mask = sitk.ReadImage(os.path.join(train_label_path, i.replace('_0000.nii.gz', '.nii.gz')))
    spacing = image.GetSpacing()
    if (np.abs(spacing[0] - target_spacing[0]) <= 0.01) and (np.abs(spacing[2] - target_spacing[2]) <= 0.01):
        print(spacing, 'continue')
        sitk.WriteImage(image, os.path.join(save_image_path, i))
        sitk.WriteImage(mask, os.path.join(save_label_path, i.replace('.nii.gz', '_0000.nii.gz')))
    else:
        print('resample before', spacing, image.GetSize())
        image_resampled = resample_simg(itkimage=image, newSpacing=target_spacing, label=False)
        mask_resampled = resample_simg(itkimage=mask, newSpacing=target_spacing, label=True)
        new_spacing = image_resampled.GetSpacing()
        assert image_resampled.GetSize() == mask_resampled.GetSize()
        print('resample after', new_spacing, image_resampled.GetSize())
        sitk.WriteImage(image_resampled, os.path.join(save_image_path, i))
        sitk.WriteImage(mask_resampled, os.path.join(save_label_path, i.replace('.nii.gz', '_0000.nii.gz')))