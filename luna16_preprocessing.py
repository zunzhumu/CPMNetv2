import os
import shutil
import numpy as np
from scipy.io import loadmat
import numpy as np
import pandas
import scipy
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial

import warnings


def resample_simg(itkimage, newSpacing=(1.0, 1.0, 1.0), resamplemethod=sitk.sitkNearestNeighbor):
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
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def ITKReDirection(itkimg, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
    # target direction should be orthognal, i.e. (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # permute axis
    tmp_target_direction = np.abs(np.round(np.array(target_direction))).reshape(3, 3).T
    current_direction = np.abs(np.round(itkimg.GetDirection())).reshape(3, 3).T

    permute_order = []
    if not np.array_equal(tmp_target_direction, current_direction):
        for i in range(3):
            for j in range(3):
                if np.array_equal(tmp_target_direction[i], current_direction[j]):
                    permute_order.append(j)
                    break
        redirect_img = sitk.PermuteAxes(itkimg, permute_order)
    else:
        redirect_img = itkimg
    # flip axis
    current_direction = np.round(np.array(redirect_img.GetDirection())).reshape(3, 3).T
    current_direction = np.max(current_direction, axis=1)

    tmp_target_direction = np.array(target_direction).reshape(3, 3).T
    tmp_target_direction = np.max(tmp_target_direction, axis=1)
    flip_order = ((tmp_target_direction * current_direction) != 1)
    fliped_img = sitk.Flip(redirect_img, [bool(flip_order[0]), bool(flip_order[1]), bool(flip_order[2])])

    return fliped_img


def process_mask(mask):
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(mask, structure=struct, iterations=10)
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def savenpy_luna(id, annos, filelist, luna_segment, luna_data, savepath):
    name = filelist[id]
    simg = sitk.ReadImage(os.path.join(luna_data, name + '.mhd'))
    smask = sitk.ReadImage(os.path.join(luna_segment, name + '.mhd'))
    if np.abs(np.array(smask.GetDirection()) - np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))).sum() > 1:
        print(name, 'redirection')
        simg = ITKReDirection(simg)
        smask = ITKReDirection(smask)

    spacing = simg.GetSpacing()[::-1] # z, y, x
    origin = simg.GetOrigin()[::-1] # z, y, x
    print(spacing)
    # resample
    new_spacing = (1., 1., 1.)
    rsimg = resample_simg(simg, new_spacing, resamplemethod=sitk.sitkLinear)  # sitk.sitkBSpline
    rsmask = resample_simg(smask, new_spacing, resamplemethod=sitk.sitkNearestNeighbor)

    img_array = sitk.GetArrayFromImage(rsimg)
    mask_array = sitk.GetArrayFromImage(rsmask)
    mask_array = mask_array.astype('uint8')
    newshape = mask_array.shape
    
    m1, m2 = mask_array == 3, mask_array == 4
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1 + dm2
    Mask = m1 + m2

    zz, yy, xx = np.where(Mask)
    box = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]])
    margin = 10
    extendbox = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + margin], axis=0).T]).T
    
    img_array = lumTrans(img_array)
    img_array = img_array * dilatedMask
    crop_image_arr = img_array[extendbox[0, 0]:extendbox[0, 1],
                   extendbox[1, 0]:extendbox[1, 1],
                   extendbox[2, 0]:extendbox[2, 1]]

    np.save(os.path.join(savepath, name + '_clean.npy'), crop_image_arr[np.newaxis,...])
    np.save(os.path.join(savepath, name + '_spacing.npy'), spacing)
    np.save(os.path.join(savepath, name + '_extendbox.npy'), extendbox)
    np.save(os.path.join(savepath, name + '_origin.npy'), origin)
    np.save(os.path.join(savepath, name + '_mask.npy'), Mask)
    # label
    this_annos = np.copy(annos[annos[:, 0] == (name)])
    label = []
    if len(this_annos) > 0:
        for c in this_annos:
            pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
            label.append(np.concatenate([pos, [c[4]]]))

    label = np.array(label)
    if len(label) == 0:
        label2 = np.array([[0, 0, 0, 0]])
    else:
        label2 = np.copy(label).T
        label2[:3] = label2[:3] * np.expand_dims(spacing, 1)  / np.expand_dims(new_spacing, 1)
        label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
        label2 = label2[:4].T
    np.save(os.path.join(savepath, name + '_label.npy'), label2)
    print(name)


def preprocess_luna():
    luna_data = '/xxxx/xxxx/xxxx/LUNA'
    luna_segment = luna_data + '/seg-lungs-LUNA16'
    savepath = luna_data + '/process'
    luna_label = luna_data + '/annotations.csv'
    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    # if not os.path.exists(finished_flag):
    annos = np.array(pandas.read_csv(luna_label))
   

    pool = Pool()
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for setidx in range(10):
        print('process subset', setidx)
        filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data + '/subset' + str(setidx)) if
                    f.endswith('.mhd')]
        if not os.path.exists(savepath + '/subset' + str(setidx)):
            os.mkdir(savepath + '/subset' + str(setidx))
        partial_savenpy_luna = partial(savenpy_luna, annos=annos, filelist=filelist,
                                   luna_segment=luna_segment, luna_data=luna_data+'/subset'+str(setidx)+'/',
                                   savepath=savepath+'/subset'+str(setidx)+'/')
        N = len(filelist)
        _ = pool.map(partial_savenpy_luna,range(N))
    pool.close()
    pool.join()
    print('end preprocessing luna')
    open(finished_flag, "w+")


if __name__ == '__main__':
    preprocess_luna()

