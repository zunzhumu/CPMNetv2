from __future__ import print_function, division

from .abstract_transform import AbstractTransform
from .image_process import *


class CoordToAnnot(AbstractTransform):
    """Convert one-channel label map to one-hot multi-channel probability map"""

    def __init__(self, blank_side=1):
        """
        """
        self.blank_side = blank_side

    def __call__(self, sample):
        ctr = sample['ctr']
        rad = sample['rad']
        cls = sample['cls']
    
        annot = np.concatenate([ctr, rad.reshape(-1, 3), cls.reshape(-1, 1)], axis=-1).astype('float32')

        sample['annot'] = annot

        return sample
