# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import math
import os
import logging
import numpy as np
from torch import nn
###network###
from networks.ResNet_3D_CPM import resnet18, Detection_Postprocess, Detection_loss
###data###
from dataload.dataset_rib import DetDatasetCSVR, DetDatasetCSVRTest, collate_fn_dict
from dataload.crop import InstanceCrop
from dataload.split_combine import SplitComb
from torch.utils.data import DataLoader
import transform
import torchvision
###optimzer###
from optimizer.optim import AdamW
from optimizer.scheduler import GradualWarmupScheduler
from sync_batchnorm import convert_model
import optimizer.solver as solver
###postprocessing###
from utils.box_utils import nms_3D
from evaluationScript.detectionCADEvalutionIOU import noduleCADEvaluation
import pandas as pd

parser = argparse.ArgumentParser(description='anchorfree_3D_aneurysm_refine')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', default=False,help='enables CUDA training')
parser.add_argument('--load', default=False, help='enables load weights')
parser.add_argument('--resume', default=True,help='resume training from epoch n')
parser.add_argument('--load_model', type=str, default='', metavar='str', help='the parameters of net')
parser.add_argument('--seed', type=int, default=233, metavar='S',help='random seed (default: 1)')
parser.add_argument('--num_workers', type=int, default=0, metavar='S', help='num_workers (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--root', type=str, default='', metavar='str', help='folder that contains data (default: test dataset)')
parser.add_argument('--train_csv', type=str, default='test.csv', help='train_list')
parser.add_argument('--val_csv', type=str, default=' ', help='val_list')
parser.add_argument('--save_model_dir', type=str, default='', metavar='str', help='folder that save model')
parser.add_argument('--save_FrocResult_dir', type=str, default='', metavar='str', help='folder that save FrocResult')
parser.add_argument('--lr', type=float, default=0.01, help='the learning rate')
parser.add_argument('--lambda_cls', type=float, default=4.0, help='weights of seg')
parser.add_argument('--lambda_offset', type=float, default=1.0,help='weights of offset')
parser.add_argument('--lambda_shape', type=float, default=0.1, help='weights of reg')
parser.add_argument('--lambda_iou', type=float, default=1.0, help='weights of iou loss')
parser.add_argument('--topk', type=int, default=5, metavar='N', help='topk grids assigned as positives')
parser.add_argument('--num_sam', type=int, default=1, metavar='N', help='sampling batch number in per sample')
parser.add_argument('--norm_type', type=str, default='batchnorm', metavar='N', help='norm type of backbone')
parser.add_argument('--head_norm', type=str, default='batchnorm', metavar='N', help='norm type of head')
parser.add_argument('--act_type', type=str, default='ReLU', metavar='N', help='act type of network')
parser.add_argument('--se', default=False, help='using se block')
parser.add_argument('--gpu', type=str, default='0', metavar='N',help='use gpu')
parser.add_argument('--log', type=str, default="logs/train.log",help='save training log to file')
parser.add_argument('--outputDir', type=str, default='bbox01_aneurysm', metavar='str',help='output')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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

CROP_SIZE = [64, 128, 128]
OVERLAP_SIZE = [16, 32, 32]
SPACING = [2.0, 0.70507812, 0.70507812]

train_roots = [args.root + '/imagesTr']
val_roots = [args.root + '/imagesVa']
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if args.cuda else {}

logger.info('The learning rate:{}'.format(args.lr))
logger.info('The batch size:{}'.format(args.batch_size))
logger.info('The Crop Size:[{}, {}, {}]'.format(CROP_SIZE[0], CROP_SIZE[1], CROP_SIZE[2]))
logger.info('topk:{}, lambda_cls:{}, lambda_shape:{}, lambda_offset:{}, lambda_iou:{},, num_sam:{}'
.format(args.topk, args.lambda_cls, args.lambda_shape, args.lambda_offset, args.lambda_iou, args.num_sam))
logger.info('norm type:{}, head norm:{}, act_type:{}, using se block:{}'
.format(args.norm_type, args.head_norm, args.act_type, args.se))
##################bulid model###################
detection_loss = Detection_loss(crop_size=CROP_SIZE, topk=args.topk, spacing=SPACING)
model = resnet18(n_channels=1, n_blocks=[2, 3, 3, 3], n_filters=[64, 96, 128, 160], stem_filters=32,norm_type=args.norm_type, 
                head_norm=args.head_norm, act_type=args.act_type, se=args.se, first_stride=(1, 2, 2), detection_loss=detection_loss, device=device)
detection_postprocess = Detection_Postprocess(topk=60, threshold=0.15, nms_threshold=0.05, num_topk=20, crop_size=CROP_SIZE)

if not os.path.exists(args.save_model_dir):
    os.makedirs(args.save_model_dir)

if torch.cuda.device_count() > 1:
    print("Using MultiGPUs")
    model = nn.DataParallel(model)
    model = convert_model(model)

model.to(device)

if args.load:
    model.load_state_dict(torch.load(args.load_model))
    print("load model successful", args.load_model)
##################set optimzer#####################
optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler_reduce = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-6)
scheduler_warm = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=2, after_scheduler=scheduler_reduce)
##################data load########################
def training_data_prepare(crop_size=CROP_SIZE, blank_side=0):
    transform_list_train = [transform.RandomFlip(flip_depth=True, flip_height=True, flip_width=True, p=0.5),
                            transform.RandomTranspose(p=0.5, trans_xy=True, trans_zx=False, trans_zy=False),
                            transform.Pad(output_size=crop_size),
                            transform.RandomCrop(output_size=crop_size, pos_ratio=0.9),
                            transform.CoordToAnnot(blank_side=blank_side)]
    train_transform = torchvision.transforms.Compose(transform_list_train)

    crop_fn_train = InstanceCrop(crop_size=crop_size, tp_ratio=0.75, spacing=SPACING,
                                        rand_trans=[10, 20, 20], rand_rot=[20, 0, 0], rand_space=[0.9, 1.2],
                                        sample_num=args.num_sam, blank_side=blank_side, instance_crop=True)

    train_dataset = DetDatasetCSVR(roots=train_roots,crop_fn=crop_fn_train,transform_post=train_transform,csv_file=os.path.join(args.root, args.train_csv))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn_dict,num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print("Train Data : ", len(train_loader.dataset))
    return train_loader

def test_val_data_prepare(roots, csv_file):
    split_comber = SplitComb(crop_size=CROP_SIZE, overlap=OVERLAP_SIZE, pad_value=-1)
    test_dataset = DetDatasetCSVRTest(roots=roots, SplitComb=split_comber, csv_file=os.path.join(args.root, csv_file))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print("Test or Val Data : ", len(test_loader.dataset))
    return test_loader

#######################training##########################################
iteration = 0
def train(epoch, train_loader, model=None):
    global iteration
    assert model is not None
    model.train().to(device)
    total_cls_loss = 0
    total_shape_loss = 0
    total_offset_loss = 0
    total_iou_loss = 0
    scheduler_warm.step() # lr_policy
    for batch_idx, sample in enumerate(train_loader):
        iteration += 1
        data = sample['image'].to(device)
        labels = sample['annot'].to(device) # z, y, x, d, h, w, type[-1, 0]
        optimizer.zero_grad()
        cls_loss, shape_loss, offset_loss, iou_loss = model([data, labels])
        cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        loss = args.lambda_cls * cls_loss + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
        total_cls_loss += args.lambda_cls * cls_loss.item()
        total_shape_loss += args.lambda_shape * shape_loss.item()
        total_offset_loss += args.lambda_offset * offset_loss.item()
        total_iou_loss += args.lambda_iou * iou_loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tcls_Loss: {:.6f} \tshape_loss:{:.6f} \toffset_loss:{:.6f} \tgiou_loss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),args.lambda_cls * cls_loss.item(), 
                args.lambda_shape * shape_loss.item(), args.lambda_offset * offset_loss.item(), args.lambda_iou * iou_loss.item()))

    logger.info('====> Epoch: {} train_cls_loss: {:.4f}'.format(epoch, total_cls_loss / (batch_idx + 1)))
    logger.info('====> Epoch: {} train_shape_loss: {:.4f}'.format(epoch, total_shape_loss / (batch_idx + 1)))
    logger.info('====> Epoch: {} train_offset_loss: {:.4f}'.format(epoch, total_offset_loss / (batch_idx + 1)))
    logger.info('====> Epoch: {} train_iou_loss: {:.4f}'.format(epoch, total_iou_loss / (batch_idx + 1)))

    model.eval().cpu()
    save_model_filename = 'Det_anchorfree' + "_epoch_" + str(epoch) + ".pth"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(model.state_dict(), save_model_path)
    logger.info("Done, trained model saved at {}".format(save_model_path))

def val(epoch, test_loader, save_dir, anno_filename, anno_ex_filename, suid_filename, model=None):

    def convert_to_standard_output(output, spacing, name):
        '''
        convert [id, prob, ctr_z, ctr_y, ctr_x, d, h, w] to
        ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
        '''
        AneurysmList = []
        spacing = np.array([spacing[0].numpy(), spacing[1].numpy(), spacing[2].numpy()]).reshape(-1, 3)
        for j in range(output.shape[0]):
            AneurysmList.append([name, output[j, 4], output[j, 3], output[j, 2], output[j, 1], output[j, 7], output[j, 6], output[j, 5]])
        return AneurysmList

    top_k = 40
    assert model is not None
    model.eval().to(device)
    split_comber = test_loader.dataset.splitcomb
    batch_size = 2 * args.batch_size * args.num_sam
    aneurysm_lists = []   
    for s, sample in enumerate(test_loader):
        data = sample['split_images'][0].to(device)
        nzhw = sample['nzhw']
        name = sample['file_name'][0]
        spacing = sample['spacing']
        outputlist = []
        for i in range(int(math.ceil(data.size(0) / batch_size))):
            end = (i+1) * batch_size
            if end > data.size(0):
                end = data.size(0)
            input = data[i*batch_size:end]
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
        # convert to ['seriesuid', 'coordX', 'coordY', 'coordZ', 'radius', 'probability']
        AneurysmList = convert_to_standard_output(output, spacing, name)
        aneurysm_lists.extend(AneurysmList)
    # save predict csv
    column_order = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
    df = pd.DataFrame(aneurysm_lists, columns=column_order)
    results_filename = os.path.join(save_dir, 'predict_epoch_{}.csv'.format(epoch))
    df.to_csv(results_filename, index=False)
    outputDir = os.path.join(save_dir, results_filename.split('/')[-1].split('.')[0])
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    try:
        FPS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        out_01 = noduleCADEvaluation(anno_filename, anno_ex_filename, suid_filename, results_filename, outputDir, 0.1)
        frocs = out_01[-1]
        logger.info('====> Epoch: {}'.format(epoch))
        for s in range(len(frocs)):
            logger.info('====> fps:{:.4f} iou 0.1 frocs:{:.4f}'.format(FPS[s], frocs[s]))
        logger.info('====> mean frocs:{:.4f}'.format(np.mean(np.array(frocs))))
    except:
        logger.info('====> Epoch: {} FROC compute error'.format(epoch))
        pass

def convert_to_standard_csv(csv_path, save_dir, state, spacing):
    '''
    convert [seriesuid	coordX	coordY	coordZ	w	h	d] to 
    'seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'
    spacing:[z, y, x]
    '''
    column_order = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd']
    gt_list = []
    csv_file = pd.read_csv(csv_path)
    seriesuid = csv_file['seriesuid']
    coordX, coordY, coordZ = csv_file['coordX'], csv_file['coordY'], csv_file['coordZ']
    w, h, d = csv_file['w'], csv_file['h'], csv_file['d']
    clean_seriesuid = []
    for j in range(seriesuid.shape[0]):
        if seriesuid[j] not in clean_seriesuid: clean_seriesuid.append(seriesuid[j])
        gt_list.append([seriesuid[j], coordX[j], coordY[j], coordZ[j], w[j]/spacing[2], h[j]/spacing[1], d[j]/spacing[0]])
    df = pd.DataFrame(gt_list, columns=column_order)
    df.to_csv(os.path.join(save_dir, 'annotation_{}.csv'.format(state)), index=False)
    df = pd.DataFrame(clean_seriesuid)
    df.to_csv(os.path.join(save_dir, 'seriesuid_{}.csv'.format(state)), index=False, header=None)


def main(epochs=None):
    assert epochs is not None
    val_loader = test_val_data_prepare(val_roots, args.val_csv)
    train_loader = training_data_prepare()
    save_dir = os.path.join(args.save_FrocResult_dir, args.outputDir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info("save_dir {}".format(save_dir))
    ###################preprocessing tool############################
    state = 'validate'
    if not os.path.exists(os.path.join(save_dir, 'annotation_{}.csv'.format(state))):
        convert_to_standard_csv(os.path.join(args.root, args.val_csv), save_dir, state, SPACING)
        logger.info("convert {} csv sucessful".format(state))
    
    if args.load:
        start_epoch = 60
    else:
        start_epoch = 200

    for epoch in range(1, epochs + 1):
        train(epoch=epoch, train_loader=train_loader, model=model)
        if epoch > start_epoch: 
            val(epoch, val_loader, save_dir, os.path.join(save_dir, 'annotation_validate.csv'), 'evaluationScript/annotations_excluded.csv', 
            os.path.join(save_dir, 'seriesuid_validate.csv'), model)



if __name__ == '__main__':
    main(args.epochs)



