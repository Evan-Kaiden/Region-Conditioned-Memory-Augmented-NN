import torch
import torch.nn as nn

import argparse
import os
import logging

from resnet import MemoryResNet18, MemoryResNet34, MemoryResNet50, MemoryResNet101, MemoryResNet152

from data import get_dataloader
from utils import set_seed, load_pretrained_imagenet
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device  : {device}')

_backbone_map = {
    'resnet18':  MemoryResNet18,
    'resnet34':  MemoryResNet34,
    'resnet50':  MemoryResNet50,
    'resnet101': MemoryResNet101,
    'resnet152': MemoryResNet152,
}
model = _backbone_map[args.backbone](use_correlation=args.use_correlation, dataset=args.dataset).to(device)

if args.use_imagenet_weights:
    model = load_pretrained_imagenet(model, args.backbone)

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
dset      = get_dataloader(args.dataset, args.batch_size)

if __name__ == '__main__':
    run(args, model, optimizer, criterion, scheduler, dset, device, logger)