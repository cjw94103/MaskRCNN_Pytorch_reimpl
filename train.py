import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import numpy as np
import argparse

from torch import optim
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from transform_util import Compose, RandomHorizontalFlip, PILToTensor, ToDtype, RandomPhotometricDistort
from coco_dataset import COCODataset
from torch.utils.data import DataLoader

from utils import *
from make_args import Args
from train_func import train

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default='./config/apple2orange_single_gpu.json', help="config path")
opt = parser.parse_args()

# load config.json
args = Args(opt.config_path)

# DataLoader
def collator(batch):
    return tuple(zip(*batch))

train_transform = Compose(
    [
        PILToTensor(),
        RandomHorizontalFlip(),
        RandomPhotometricDistort(),
        ToDtype(scale=True, dtype=torch.float)
    ]
)
val_transform = Compose(
    [
        PILToTensor(),
        ToDtype(scale=True, dtype=torch.float)
    ]
)

train_dataset = COCODataset(args.data_path, train=True, transform=train_transform)
val_dataset = COCODataset(args.data_path, train=False, transform=val_transform)

train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collator, num_workers=args.num_workers
)
val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collator, num_workers=args.num_workers
)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(train_dataset.new_categories)

if args.backbone == 'resnet50fpn':
    model = maskrcnn_resnet50_fpn(pretrained_backbone=True) # imagenet pretrained
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=num_classes
    )
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=model.roi_heads.mask_predictor.conv5_mask.in_channels,
        dim_reduced=args.hidden_layer,
        num_classes=num_classes
    )
    model.to(device)   

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Train
train(args, model, train_dataloader, val_dataloader, optimizer)