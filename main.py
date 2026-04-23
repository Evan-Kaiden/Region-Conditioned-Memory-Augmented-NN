import torch
import torch.nn as nn

import argparse
import os
import logging

from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 #MemoryResNet18, MemoryResNet34, MemoryResNet50, MemoryResNet101, MemoryResNet152, 
from cca_memory_model import CCAMemoryModel
from data import get_dataloader
from utils import set_seed, load_pretrained_imagenet, get_encoder_out_channels, get_pytorch_device
from train import run


parser = argparse.ArgumentParser()

parser.add_argument('--backbone', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--weight_decay', type=float, default=0.025)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_scheduler', type=str, default='cosine',
                    choices=['cosine', 'linear', 'step', 'none'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'stl10', 'oxfordpets'])
parser.add_argument('--use_imagenet_weights', action='store_true', default=False)
parser.add_argument('--use_correlation', action='store_true', default=False,
                    help='Use correlation attention + spatial contrastive loss. '
                         'If False, uses BaselineMemory with cross-entropy only.')
parser.add_argument('--generate_viz', action='store_true', default=False,
                    help='Save attention visualisations during evaluation.')
parser.add_argument('--seed', type=int, default=1666)

args = parser.parse_args()

run_dir = os.path.join(
    'Experiments',
    str(args.seed),
    args.dataset,
    args.backbone,
    'correlation' if args.use_correlation else 'baseline',
)
os.makedirs(run_dir, exist_ok=True)
args.run_dir = run_dir

log_path = os.path.join(run_dir, 'train.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),          # also print to console
    ]
)
logger = logging.getLogger(__name__)
logger.info(f'Run dir : {run_dir}')
logger.info(f'Args    : {vars(args)}')

set_seed(args.seed)

device = get_pytorch_device()

logger.info(f'Device  : {device}')

# _backbone_map = {
#     'resnet18':  MemoryResNet18,
#     'resnet34':  MemoryResNet34,
#     'resnet50':  MemoryResNet50,
#     'resnet101': MemoryResNet101,
#     'resnet152': MemoryResNet152,
# }

dset = get_dataloader(args.dataset, args.batch_size)


_backbone_map = {
    'resnet18':  ResNet18,
    'resnet34':  ResNet34,
    'resnet50':  ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
}
backbone = _backbone_map[args.backbone]().to(device) #(use_correlation=args.use_correlation, dataset=args.dataset).to(device)

if args.use_imagenet_weights:
    backbone = load_pretrained_imagenet(backbone, args.backbone)



out_channels = get_encoder_out_channels(backbone, dset.train_loader, device)
model = CCAMemoryModel(encoder=backbone, encoder_out_channels=out_channels, num_classes=len(dset.classes)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.lr_scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.lr_scheduler == 'linear':
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=args.epochs)
elif args.lr_scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
else:
    scheduler = None

criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    run(args, model, optimizer, criterion, scheduler, dset, device, logger)