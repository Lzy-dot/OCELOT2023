#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""
import warnings
warnings.filterwarnings("ignore")
# from torchmetrics import IoU
from torch.optim.lr_scheduler import LambdaLR
from monai.inferers import sliding_window_inference
import torch.nn.functional as F
from monai.data import decollate_batch, PILReader
from skimage import io, segmentation, morphology, measure, exposure
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage import feature
from pickle import FALSE
import sys
import argparse
import os
import cv2
join = os.path.join
import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import monai
from monai.data import decollate_batch, PILReader,NumpyReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    EnsureTyped,
    EnsureType,
    RandGaussianSharpend,
    ResizeWithPadOrCropd,
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import sys
# sys.path.append("/ssd/lizhaoyang/code/neurips22-cellseg_saltfish/code_saltfish")
from utils import gcio
from utils import Dice_Loss
from utils.miou import compute_miou, fast_hist, per_class_iu, compute_mIoU
# from torch.autograd import Variable
# from net import ResUNETR_s2_tissue, ResUNETR_s2_cell ,ResUNETR_s2widetiny,UnetModel, ResUNETR_s2
from net_sam import SAM_Model
os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '20000'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

print("Successfully imported all requirements!")
from pathlib import Path
output_json_path = Path("/ssd/lizhaoyang/code/neurips22-cellseg_saltfish/code_saltfish/work_dir/cell_classification.json")
import re
import numpy as np
### These are fixed, don't change!
DISTANCE_CUTOFF = 15
CLS_IDX_TO_NAME = {1: "BC", 2: "TC"}


def prepare_input(cell_patch, tissue_patch, mask_meta):
    # resize_to=(1024,1024)
    resize_to=(512,512)
    """This function prepares the cell patch array to be forwarded by
    the model

    Parameters
    ----------
    cell_patch: np.ndarray[uint8]
        Cell patch with shape [1024, 1024, 3] with values from 0 - 255

    Returns
    -------
        torch.tensor of shape [1, 3, 1024, 1024] where the first axis is the batch
        dimension
    """
    cell_min =torch.min(cell_patch)
    cell_max=torch.max(cell_patch)
    tissue_min=torch.min(tissue_patch)
    tissue_max=torch.max(tissue_patch)
    cell_patch = (cell_patch - cell_min) / (cell_max - cell_min)
    tissue_patch = (tissue_patch - tissue_min) / (tissue_max - tissue_min)
    cell_patch= F.interpolate(
                cell_patch, size=resize_to, mode="area"
        ).detach()

    # tissue_patch = tissue_patch / 255 # normalize [0-1]
    tissue_patch= F.interpolate(
                tissue_patch, size=resize_to, mode="area"
        ).detach()

    mask_meta= F.interpolate(
                mask_meta, size=resize_to, mode="nearest"
        ).detach()
    return cell_patch, tissue_patch, mask_meta


def post_process(logits):
    """This function applies some post processing to the
    output logits
    
    Parameters
    ----------
    logits: torch.tensor
        Outputs of U-Net

    Returns
    -------
        torch.tensor after post processing the logits
    """
    logits = F.interpolate(logits, size=(1024,1024),
            mode='bilinear', align_corners=False
        )
    return torch.softmax(logits, dim=1)


def find_cells(heatmap,id):
    """This function detects the cells in the output heatmap

    Parameters
    ----------
    heatmap: torch.tensor
        output heatmap of the model,  shape: [1, 3, 1024, 1024]

    Returns
    -------
        List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
    """
    arr = heatmap[0,:,:,:].cpu().detach().numpy()
    # arr = np.transpose(arr, (1, 2, 0)) # CHW -> HWC

    bg, pred_wo_bg = np.split(arr, (1,), axis=0) # Background and non-background channels
    bg = np.squeeze(bg, axis=0)
    obj = 1.0 - bg

    arr = cv2.GaussianBlur(obj, (0, 0), sigmaX=3)
    peaks = feature.peak_local_max(
        arr, min_distance=7, exclude_border=0, threshold_abs=0.0
    ) # List[y, x]

    maxval = np.max(pred_wo_bg, axis=0)
    maxcls_0 = np.argmax(pred_wo_bg, axis=0)

    # Filter out peaks if background score dominates
    peaks = np.array([peak for peak in peaks if bg[peak[0], peak[1]] < maxval[peak[0], peak[1]]])
    if len(peaks) == 0:
        return []
    # Get score and class of the peaks
    scores = maxval[peaks[:, 0], peaks[:, 1]]
    peak_class = maxcls_0[peaks[:, 0], peaks[:, 1]]

    predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]

    return predicted_cells



def _check_validity(inp):
    """ Check validity of algorithm output.

    Parameters
    ----------
    inp: List[Dict]
        List of cell predictions, each element corresponds a cell point.
        Each element is a dictionary with 3 keys, `name`, `point`, `probability`.
        Value of `name` key is `image_{idx}` where `idx` indicates the image index.
        Value of `point` key is a list of three elements, x, y, and cls.
        Value of `probability` key is a confidence score of a predicted cell.
    """
    for cell in inp:
        assert sorted(list(cell.keys())) == ["name", "point", "probability"]
        assert re.fullmatch(r'image_[0-9]+', cell["name"]) is not None
        assert type(cell["point"]) is list and len(cell["point"]) == 3
        assert type(cell["point"][0]) is int and 0 <= cell["point"][0] <= 1023
        assert type(cell["point"][1]) is int and 0 <= cell["point"][1] <= 1023
        assert type(cell["point"][2]) is int and cell["point"][2] in (1, 2)
        assert type(cell["probability"]) is float and 0.0 <= cell["probability"] <= 1.0


def _convert_format(pred_json, gt_json, num_images):
    """ Helper function that converts the format for easy score computation.

    Parameters
    ----------
    pred_json: List[Dict]
        List of cell predictions, each element corresponds a cell point.
        Each element is a dictionary with 3 keys, `name`, `point`, `probability`.
        Value of `name` key is `image_{idx}` where `idx` indicates the image index.
        Value of `point` key is a list of three elements, x, y, and cls.
        Value of `probability` key is a confidence score of a predicted cell.
    
    gt_json: List[Dict]
        List of cell ground-truths, each element corresponds a cell point.
        Each element is a dictionary with 3 keys, `name`, `point`, `probability`.
        Value of `name` key is `image_{idx}` where `idx` indicates the image index.
        Value of `point` key is a list of three elements, x, y, and cls.
        Value of `probability` key is always 1.0.
    
    num_images: int
        Number of images.
    
    Returns
    -------
    pred_after_convert: List[List[Tuple(int, int, int, float)]]
        List of predictions, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each predicted cell consist of x, y, cls, prob.
    
    gt_after_convert: List[List[Tuple(int, int, int, float)]]
        List of GT, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each GT cell consist of x, y, cls, prob (always 1.0).
    """
    pred_ids = []
    pred_after_convert = [[] for _ in range(num_images)]
    for pred_cell in pred_json:
        x, y, c = pred_cell["point"]
        prob = pred_cell["probability"]
        img_idx = int(pred_cell["name"].split("_")[-1])
        if img_idx not in pred_ids:
            pred_ids.append(img_idx)
        pred_after_convert[img_idx].append((x, y, c, prob))

    print("pred_ids",len(pred_ids))
    gt_after_convert = [[] for _ in range(num_images)]
    for gt_cell in gt_json:
        if int(gt_cell["name"].split("_")[-1]) in pred_ids:       
            x, y, c = gt_cell["point"]
            prob = gt_cell["probability"]
            img_idx = int(gt_cell["name"].split("_")[-1])
            gt_after_convert[img_idx].append((x, y, c, prob))
        else:
            continue
    return pred_after_convert, gt_after_convert


def _preprocess_distance_and_confidence(pred_all, gt_all):
    """ Preprocess distance and confidence used for F1 calculation.

    Parameters
    ----------
    pred_all: List[List[Tuple(int, int, int, float)]]
        List of predictions, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each predicted cell consist of x, y, cls, prob.

    gt_all: List[List[Tuple(int, int, int)]]
        List of GTs, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each GT cell consist of x, y, cls.

    Returns
    -------
    all_sample_result: List[List[Tuple(int, np.array, np.array)]]
        Distance (between pred and GT) and Confidence per class and sample.
    """

    all_sample_result = []

    for pred, gt in zip(pred_all, gt_all):
        one_sample_result = {}

        for cls_idx in sorted(list(CLS_IDX_TO_NAME.keys())):
            pred_cls = np.array([p for p in pred if p[2] == cls_idx], np.float32)
            gt_cls = np.array([g for g in gt if g[2] == cls_idx], np.float32)
            if len(gt_cls) == 0:
                gt_cls = np.zeros(shape=(0, 4))
            
            if len(pred_cls) == 0:
                distance = np.zeros([0, len(gt_cls)])
                confidence = np.zeros([0, len(gt_cls)])
            else:
                pred_loc = pred_cls[:, :2].reshape([-1, 1, 2])
                gt_loc = gt_cls[:, :2].reshape([1, -1, 2])
                distance = np.linalg.norm(pred_loc - gt_loc, axis=2)
                confidence = pred_cls[:, 2]

            one_sample_result[cls_idx] = (distance, confidence)

        all_sample_result.append(one_sample_result)

    return all_sample_result


def _calc_scores(all_sample_result, cls_idx, cutoff):
    """ Calculate Precision, Recall, and F1 scores for given class 
    
    Parameters
    ----------
    all_sample_result: List[List[Tuple(int, np.array, np.array)]]
        Distance (between pred and GT) and Confidence per class and sample.

    cls_idx: int
        1 or 2, where 1 and 2 corresponds Tumor (TC) and Background (BC) cells, respectively.

    cutoff: int
        Distance cutoff that used as a threshold for collecting candidates of 
        matching ground-truths per each predicted cell.

    Returns
    -------
    precision: float
        Precision of given class

    recall: float
        Recall of given class

    f1: float
        F1 of given class
    """
    
    global_num_gt = 0
    global_num_tp = 0
    global_num_fp = 0

    for one_sample_result in all_sample_result:
        distance, confidence = one_sample_result[cls_idx]
        num_pred, num_gt = distance.shape
        assert len(confidence) == num_pred

        sorted_pred_indices = np.argsort(-confidence)
        bool_mask = (distance <= cutoff)

        num_tp = 0
        num_fp = 0
        for pred_idx in sorted_pred_indices:
            gt_neighbors = bool_mask[pred_idx].nonzero()[0]
            if len(gt_neighbors) == 0:  # No matching GT --> False Positive
                num_fp += 1
            else:  # Assign neares GT --> True Positive
                gt_idx = min(gt_neighbors, key=lambda gt_idx: distance[pred_idx, gt_idx])
                num_tp += 1
                bool_mask[:, gt_idx] = False

        assert num_tp + num_fp == num_pred
        global_num_gt += num_gt
        global_num_tp += num_tp
        global_num_fp += num_fp
        
    precision = global_num_tp / (global_num_tp + global_num_fp + 1e-7)
    recall = global_num_tp / (global_num_gt + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def calmF1():
    """ Calculate mF1 score and save scores.

    Returns
    -------
    float
        A mF1 value which is average of F1 scores of BC and TC classes.
    """

    # Path where algorithm output is stored
    algorithm_output_path = "/ssd/lizhaoyang/code/neurips22-cellseg_saltfish/code_saltfish/work_dir/cell_classification.json"
    with open(algorithm_output_path, "r") as f:
        pred_json = json.load(f)["points"]
    
    # Path where GT is stored
    gt_path = "/ssd/lizhaoyang/code/neurips22-cellseg_saltfish/data/cell/cell_gt_train.json"
    with open(gt_path, "r") as f:
        data = json.load(f)
        gt_json = data["points"]
        num_images = data["num_images"]

    # Check the validity (e.g. type) of algorithm output
    _check_validity(pred_json)
    _check_validity(gt_json)

    # Convert the format of GT and pred for easy score computation
    pred_all, gt_all = _convert_format(pred_json, gt_json, num_images)

    # For each sample, get distance and confidence by comparing prediction and GT
    all_sample_result = _preprocess_distance_and_confidence(pred_all, gt_all)

    # Calculate scores of each class, then get final mF1 score
    scores = {}
    for cls_idx, cls_name in CLS_IDX_TO_NAME.items():
        precision, recall, f1 = _calc_scores(all_sample_result, cls_idx, DISTANCE_CUTOFF)
        scores[f"Pre/{cls_name}"] = precision
        scores[f"Rec/{cls_name}"] = recall
        scores[f"F1/{cls_name}"] = f1
    
    scores["mF1"] = sum([
        scores[f"F1/{cls_name}"] for cls_name in CLS_IDX_TO_NAME.values()
    ]) / len(CLS_IDX_TO_NAME)
    return scores
    print(scores)




class wL1Loss(nn.Module):
    def __init__(self):
        super(wL1Loss,self).__init__()
        
    def forward(self,inputs,weight):
        weight2 = (weight==0)
        weight1 = (weight>0)
        weight3 = (weight<0)
        num1 = torch.sum(weight2 )
        num2 = torch.sum(weight1 )
        num3 = torch.sum(weight3)
        loss = torch.sum(torch.abs(inputs*weight2))/(num1+1e-3)+torch.sum(torch.abs(inputs*weight1-weight*weight1))/(num2+1e-3)+torch.sum(torch.abs(inputs*weight3-weight*weight3))/(num3+1e-3)
        # print(num1,num2,num3,loss)
        return loss

def main():
    parser = argparse.ArgumentParser("Baseline for Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path_tissue",
        default="/ssd/lizhaoyang/code/neurips22-cellseg_saltfish/data/tissue_pre",
        type=str,
        help="training tissue data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--data_path_cell",
        # default="/ssd/lizhaoyang/code/neurips22-cellseg_saltfish/data/cell_pre",
        default="/ssd/lizhaoyang/code/ocelot23algo/cell_pre",
        type=str,
        help="training cell data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--weight_path",
        default="/ssd/lizhaoyang/code/neurips22-cellseg_saltfish/data/cell/weights",
        type=str,
        help="training cell distance transform weight path",
    )
    parser.add_argument(
        "--work_dir", default="./work_dir", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2023, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", default="unetmodel", help="select mode: res50_unetr, res50wt_unetr,"
    )
    parser.add_argument("--num_class", default=2, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=512, type=int, help="segmentation classes"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=500, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=100, type=int)
    parser.add_argument("--initial_lr", type=float, default=6e-4, help="learning rate")
    parser.add_argument("--warmup", type=bool, default=False, help="learning rate")
    parser.add_argument("--warmup_period", type=int, default=50, help="learning rate")
    parser.add_argument('--lr_exp', type=float, default=0.85, help='The learning rate decay expotential')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    ### 获取gpu id
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", init_method='env://', rank=local_rank, world_size=4)
    monai.config.print_config()
    np.random.seed(args.seed)

    model_path = join(args.work_dir, args.model_name)
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    img_path_tissue = join(args.data_path_tissue, "images")
    img_path_cell = join(args.data_path_cell, "images")

    gt_path_tissue = join(args.data_path_tissue, "labels")
    gt_path_cell = join(args.data_path_cell, "labels")
    mask_meta_path = join(args.data_path_tissue, "mask_meta")
    # gt_path = join(args.data_path, "labels")

    
    img_names = sorted(os.listdir(img_path_cell))
    # 取出img_names的文件名就是对应的图片id，tissue和cell一样
    ids = [img_name.split(".")[0] for img_name in img_names]
    gt_names = [img_name.split(".")[0] + ".png" for img_name in img_names]
    img_num = len(img_names)
    val_frac = 0.0
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    val_split = int(img_num * val_frac)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

   
    weight_path = args.weight_path # 距离变换权重
    direct_names = [gt_name.split(".")[0] + ".npy" for gt_name in gt_names]
        ## 重新初始化一下
    np.random.seed(args.seed+local_rank)
    train_files = [
        {
        "id": ids[i],
        "img_tissue": join(img_path_tissue, img_names[i]), 
        "label": join(gt_path_tissue, gt_names[i]), 
        "img_cell": join(img_path_cell, img_names[i]),
        "weight": join(weight_path,direct_names[i]),
        "label_cell": join(gt_path_cell,gt_names[i]),
        "mask_meta" : join(mask_meta_path,gt_names[i])
        }
        for i in train_indices
    ]

    val_files = [
        {
        "id": ids[i],
        "img_tissue": join(img_path_tissue, img_names[i]), 
        "label": join(gt_path_tissue, gt_names[i]), 
        "img_cell": join(img_path_cell, img_names[i]),
        "weight": join(weight_path,direct_names[i]),
        "label_cell": join(gt_path_cell,gt_names[i]),
        "mask_meta" : join(mask_meta_path,gt_names[i])
       }
        for i in val_indices
    ]
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    #%% define transforms for image and segmentation
    train_transforms_tissue = Compose(
        [
            LoadImaged(
                keys=["img_tissue", "img_cell", "label", "label_cell","mask_meta"], reader=PILReader, dtype=np.float32
            ),  # image three channels (H, W, 3); label: (H, W)
            LoadImaged(
                keys=["weight"],reader=NumpyReader, dtype=np.float32
            ),
            AddChanneld(keys=["label", "label_cell", "weight","mask_meta"], allow_missing_keys=True),  # label: (1, H, W)
            AsChannelFirstd(
                keys=["img_tissue","img_cell"], channel_dim=-1, allow_missing_keys=True,
            ),  # image: (3, H, W)
            ScaleIntensityd(
                keys=["img_tissue","img_cell"], allow_missing_keys=True,
            ),  # Do not scale label

            # 这里对图片有缩放，所以在tissue中裁剪对应区域的时候应该按照比例来
            RandZoomd(
                keys=["img_tissue", "img_cell", "label", "weight", "label_cell","mask_meta"],
                prob=1,
                min_zoom=0.5,
                max_zoom=0.5,
                mode=["area", "area", "nearest", "bilinear","nearest","nearest"],
                keep_size=False,
            ),
            RandAxisFlipd(keys=["img_tissue", "img_cell", "label","weight","label_cell","mask_meta"], prob=0.5),
            RandRotate90d(keys=["img_tissue", "img_cell", "label","weight","label_cell","mask_meta"], prob=0.5, spatial_axes=[0, 1]),
            RandGaussianNoised(keys=["img_tissue","img_cell"], prob=0.25, mean=0, std=0.1),
            # RandAdjustContrastd(keys=["img_tissue","img_cell"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img_tissue","img_cell"], prob=0.25, sigma_x=(1, 2)),
            EnsureTyped(keys=["img_tissue", "img_cell", "label","weight","label_cell","mask_meta"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img_tissue", "img_cell", "label","label_cell","mask_meta"], reader=PILReader, dtype=np.float32),
            LoadImaged(
                keys=["weight"], reader=NumpyReader, dtype=np.float32
            ),
            AddChanneld(keys=["label", "weight","label_cell","mask_meta"], allow_missing_keys=True),
            AsChannelFirstd(keys=["img_tissue", "img_cell"], channel_dim=-1, allow_missing_keys=True),
            # ScaleIntensityd(keys=["img_tissue", "img_cell"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img_tissue", "img_cell", "label","weight","label_cell","mask_meta"]),
        ]
    )
    
    if dist.get_rank() == 0:
        check_ds = monai.data.Dataset(data=train_files, transform=train_transforms_tissue)
        check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
        check_data = monai.utils.misc.first(check_loader)
        print(
            "sanity check:",
            check_data["img_tissue"].shape, 
            check_data["img_cell"].shape, 
            torch.max(check_data["img_tissue"]),
            torch.max(check_data["img_cell"]),
            check_data["label"].shape, 
            check_data["label_cell"].shape, 
            torch.max(check_data["label"]),
            torch.max(check_data["label_cell"]),
            check_data["weight"].shape, 
            torch.min(check_data["weight"]),
        )
    args.batch_size = args.batch_size // torch.cuda.device_count()
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms_tissue)
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler= train_sampler,
        # shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
 
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name.lower() == "samh_unet_final":
        model = SAM_Model(model_path='/ssd/lizhaoyang/code/neurips22-cellseg_saltfish/OCELOT2023/code/checkpoint/sam_vit_h.pth',img_size=(args.input_size, args.input_size), in_channels=3, out_channels=args.num_class)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device) 
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=False)
# loss_function = monai.losses.DiceFocalLoss(softmax = False)
    loss_function = nn.CrossEntropyLoss(ignore_index=2)
    loss_function_cell = monai.losses.DiceCELoss(softmax = True)
    # loss_function_tissue = monai.losses.DiceFocalLoss(softmax = False)
    loss_function_tissue = Dice_Loss.DiceLoss() # 这里的Dice_loss里面自带了softmax
    loss_function2 = wL1Loss()

    base_lr = args.initial_lr
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    # start a typical PyTorch training
    max_epochs = args.max_epochs

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    val_interval = args.val_interval
    epoch_tolerance = args.epoch_tolerance
    if dist.get_rank() == 0:
        writer = SummaryWriter(model_path)
    # writer = SummaryWriter(model_path)
    iter_num = 0
    max_iterations = args.max_epochs * len(train_loader)
    for epoch in range(1, max_epochs+1):
        # train_sampler.set_epoch(epoch)
        train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        loss_reg = 0
        loss_cls = 0
        for step, batch_data in enumerate(train_loader, 1):
            ids, inputs_tissue, inputs_cell, labels, labels_cell, mask_meta = batch_data['id'], batch_data["img_tissue"].to(device), batch_data["img_cell"].to(device), batch_data["label"].to(device, dtype=torch.long), batch_data["label_cell"].to(device, dtype=torch.long), batch_data["mask_meta"].to(device),
            optimizer.zero_grad()
            labels[labels==255] = 2
            labels_onehot_tissue = monai.networks.one_hot(
                labels, 3
            )  
            labels_onehot = monai.networks.one_hot(
                labels_cell, 3
            )  
            outputs, dis = model(ids, inputs_tissue, inputs_cell, mask_meta) 
            loss1 = loss_function_tissue(outputs, labels_onehot_tissue[:,0:2,:,:]) + loss_function(outputs,labels.squeeze(1))
            loss2 = loss_function_cell(dis, labels_onehot) 
            loss =loss1+loss2     
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_  
            iter_num = iter_num + 1 

            epoch_loss += loss.item()
            # print("epoch_loss",epoch_loss.device)
            loss_reg += loss2.item()
            loss_cls += loss1.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            if dist.get_rank() == 0:
                writer.add_scalar("cls_loss", loss1.item(), epoch_len * epoch + step)
                writer.add_scalar("dice_loss", loss2.item(), epoch_len * epoch + step) 
        epoch_loss /= step
        loss_reg /= step
        loss_cls /= step
        epoch_loss_values.append(epoch_loss)
        if dist.get_rank() == 0:
            print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
            print(f"epoch {epoch} average loss cls: {loss_cls:.4f}")
            print(f"epoch {epoch} average loss dice: {loss_reg:.4f}")
            print("lr:",optimizer.param_groups[0]['lr'])  
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                # "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss_values,
            }
            if epoch == args.max_epochs:
                torch.save(checkpoint, join(model_path, "seg_celltissue512_lrsame_enc2_f1_final_model.pth"))
    if dist.get_rank() == 0:    
        writer.close()

if __name__ == "__main__":
    main()
