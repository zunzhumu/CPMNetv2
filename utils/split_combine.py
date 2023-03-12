import torch
import numpy as np


class SplitComb():
    def __init__(self, side_len, max_stride, stride, margin, pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value

    def split(self, data, side_len=None, max_stride=None, margin=None):
        if side_len == None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin

        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))

        nzhw = [nz, nh, nw]
        self.nzhw = nzhw
        pad = [[0, 0],
                [0, int(nz * side_len - z + 2 * margin)],
                [0, int(nh * side_len - h + 2 * margin)],
                [0, int(nw * side_len - w + 2 * margin)]]

        data = np.pad(data, pad, 'edge')  # change

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = int(iz * side_len)
                    ez = int((iz + 1) * side_len + 2 * margin)
                    sh = int(ih * side_len)
                    eh = int((ih + 1) * side_len + 2 * margin)
                    sw = int(iw * side_len)
                    ew = int((iw + 1) * side_len + 2 * margin)

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits, nzhw

    def combine(self, output, nzhw=None, side_len=None, stride=None, margin=None):

        if side_len == None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        assert nzhw is not None
        nz,nh,nw = nzhw
        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = int(iz * side_len)
                    sh = int(ih * side_len)
                    sw = int(iw * side_len)
                    # num 400 6
                    # 6-> id, prob, z_min, y_min, x_min, d
                    output[idx][:, 2] += sz
                    output[idx][:, 3] += sh
                    output[idx][:, 4] += sw
                    idx += 1

        return output