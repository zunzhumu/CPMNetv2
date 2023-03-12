import torch
import numpy as np
import math

def nms_3D(dets, overlap=0.5, top_k=200):
    # det {prob, ctr_z, ctr_y, ctr_x, d, h, w}
    dd, hh, ww = dets[:, 4], dets[:, 5], dets[:, 6]
    z1 = dets[:, 1] - 0.5 * dd
    y1 = dets[:, 2] - 0.5 * hh
    x1 = dets[:, 3] - 0.5 * ww
    z2 = dets[:, 1] + 0.5 * dd
    y2 = dets[:, 2] + 0.5 * hh
    x2 = dets[:, 3] + 0.5 * ww
    scores = dets[:, 0]
    areas = dd * hh * ww
    _, idx = scores.sort(0, descending=True)
    keep = []
    while idx.size(0) > 0:
        i = idx[0]
        keep.append(int(i.cpu().numpy()))
        if idx.size(0) == 1 or len(keep) == top_k:
            break
        xx1 = torch.max(x1[idx[1:]], x1[i].expand(len(idx)-1))
        yy1 = torch.max(y1[idx[1:]], y1[i].expand(len(idx)-1))
        zz1 = torch.max(z1[idx[1:]], z1[i].expand(len(idx)-1))

        xx2 = torch.min(x2[idx[1:]], x2[i].expand(len(idx)-1))
        yy2 = torch.min(y2[idx[1:]], y2[i].expand(len(idx)-1))
        zz2 = torch.min(z2[idx[1:]], z2[i].expand(len(idx)-1))

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        d = torch.clamp(zz2 - zz1, min=0.0)

        inter = w*h*d
        IoU = inter / (areas[i] + areas[idx[1:]] - inter)
        inds = IoU <= overlap
        idx = idx[1:][inds]
    return torch.from_numpy(np.array(keep))


def iou_3D(box1, box2):
    # need z_ctr, y_ctr, x_ctr, d
    z1 = np.maximum(box1[0] - 0.5 * box1[3], box2[0] - 0.5 * box2[3])
    y1 = np.maximum(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    x1 = np.maximum(box1[2] - 0.5 * box1[3], box2[2] - 0.5 * box2[3])

    z2 = np.minimum(box1[0] + 0.5 * box1[3], box2[0] + 0.5 * box2[3])
    y2 = np.minimum(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3])
    x2 = np.minimum(box1[2] + 0.5 * box1[3], box2[2] + 0.5 * box2[3])

    w = np.maximum(x2 - x1, 0.)
    h = np.maximum(y2 - y1, 0.)
    d = np.maximum(z2 - z1, 0.)

    inters = w * h * d
    uni = box1[3] * box1[3] * box1[3] + box2[3] * box2[3] * box2[3] - inters
    uni = np.maximum(uni, 1e-8)
    ious = inters / uni
    return ious

