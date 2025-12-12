#!/usr/bin/env python
"""
Unified training script for PCT family (NaivePCT, SPCT, PCT) on ModelNet40.

Features:
 - Configurable model variant
 - Adam or SGD optimizer
 - CosineAnnealingLR or constant LR
 - Automatic checkpoint directory management
 - Resume from checkpoint
 - Mixed Precision (AMP) optional
 - Distributed Data Parallel (DDP) if launched via torchrun / WORLD_SIZE>1 or --ddp
 - DataParallel fallback for multi-GPU without DDP
 - Periodic checkpoint saving & best model tracking
 - Evaluation-only mode

Quick Examples:
 Single GPU (Adam):
   python train_pct.py --exp_name pct_adam --model pct --epochs 5

 Single GPU (SGD):
   python train_pct.py --exp_name pct_sgd --model pct --use_sgd --lr 0.001 --epochs 5

 Mixed precision:
   python train_pct.py --exp_name pct_amp --amp --epochs 5

 DDP (2 GPUs):
   torchrun --nproc_per_node=2 train_pct.py --exp_name pct_ddp --model pct --epochs 5 --ddp

 Resume:
   python train_pct.py --exp_name pct_resume --resume checkpoints/pct_adam/best.pt --epochs 5

 Eval only:
   python train_pct.py --exp_name pct_eval --eval_only --resume checkpoints/pct_adam/best.pt

Environment variable for dataset (optional):
   export MODELNET40_HDF5_DIR=/path/to/modelnet40_ply_hdf5_2048
"""

import argparse
import os
import time
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics

from dataset import ModelNet40
from model import NaivePCTCls, SPCTCls, PCTCls, PCTKDTCls
from util import cal_loss, Logger

MODELS = {
    'navie_pct': NaivePCTCls,
    'spct': SPCTCls,
    'pct': PCTCls,
    'pct_kdtree': PCTKDTCls,
}


@dataclass
class TrainState:
    epoch: int
    best_acc: float
    optimizer: Dict[str, Any]
    scaler: Dict[str, Any]


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def init_distributed(args):
    if args.ddp:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
        else:
            # Single-node launch fallback
            args.rank = 0
            args.world_size = 1
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
        if is_main_process():
            print(f"[DDP] Initialized. Rank {args.rank}/{args.world_size}")
        torch.cuda.set_device(args.local_rank)


def setup_output_dir(args):
    out_dir = os.path.join('checkpoints', args.exp_name)
    model_dir = os.path.join(out_dir, 'models')
    if is_main_process():
        os.makedirs(model_dir, exist_ok=True)
    return out_dir, model_dir


def build_model(args):
    model_cls = MODELS[args.model]
    if args.model == 'pct_kdtree':
        model = model_cls(leaf_size=args.leaf_size, strategy=args.leaf_strategy).to(args.device)
    else:
        model = model_cls().to(args.device)
    if args.ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    else:
        # Optional DataParallel if multiple GPUs detected and not using DDP
        if args.device.type == 'cuda' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    return model


def build_optimizer(args, model):
    if args.use_sgd:
        base_lr = args.lr * (100.0 if args.lr_scale_sgd else 1.0)
        opt = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr if args.min_lr is not None else args.lr)
    else:
        scheduler = None
    return opt, scheduler


def save_checkpoint(path, model, optimizer, scaler, epoch, best_acc):
    save_obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'best_acc': best_acc,
    }
    torch.save(save_obj, path)


def load_checkpoint(args, model, optimizer=None, scaler=None):
    ckpt = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt and ckpt['optimizer']:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler and 'scaler' in ckpt and ckpt['scaler'] and scaler is not None:
        scaler.load_state_dict(ckpt['scaler'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_acc = ckpt.get('best_acc', 0.0)
    return start_epoch, best_acc


def gather_arrays(tensor: torch.Tensor):
    """Gather a 1D tensor from all processes and return concatenated numpy array."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor.detach().cpu().numpy()
    tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, tensor)
    cat = torch.cat(tensors, dim=0)
    return cat.detach().cpu().numpy()


def evaluate(args, model, loader, criterion):
    model.eval()
    test_loss = 0.0
    count = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for data, label in loader:
            data = data.to(args.device).permute(0, 2, 1)
            label = label.to(args.device).squeeze()
            batch_size = data.size(0)
            logits = model(data)
            loss = criterion(logits, label)
            pred = logits.max(dim=1)[1]
            test_loss += loss.item() * batch_size
            count += batch_size
            preds_all.append(pred)
            labels_all.append(label)
    preds_cat = torch.cat(preds_all)
    labels_cat = torch.cat(labels_all)
    # Gather if DDP
    preds_np = gather_arrays(preds_cat)
    labels_np = gather_arrays(labels_cat)
    acc = metrics.accuracy_score(labels_np, preds_np)
    bal_acc = metrics.balanced_accuracy_score(labels_np, preds_np)
    return test_loss / max(count, 1), acc, bal_acc


def train_one_epoch(args, model, loader, optimizer, scaler, criterion, epoch):
    model.train()
    train_loss = 0.0
    count = 0
    preds_all = []
    labels_all = []
    start_epoch_time = time.time()
    for data, label in loader:
        data = data.to(args.device).permute(0, 2, 1)
        label = label.to(args.device).squeeze()
        batch_size = data.size(0)
        optimizer.zero_grad(set_to_none=True)
        if args.amp:
            with torch.cuda.amp.autocast():
                logits = model(data)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
        pred = logits.max(dim=1)[1]
        train_loss += loss.item() * batch_size
        count += batch_size
        preds_all.append(pred)
        labels_all.append(label)
    epoch_time = time.time() - start_epoch_time
    preds_cat = torch.cat(preds_all)
    labels_cat = torch.cat(labels_all)
    preds_np = gather_arrays(preds_cat)
    labels_np = gather_arrays(labels_cat)
    acc = metrics.accuracy_score(labels_np, preds_np)
    bal_acc = metrics.balanced_accuracy_score(labels_np, preds_np)
    return train_loss / max(count, 1), acc, bal_acc, epoch_time


def main():
    parser = argparse.ArgumentParser(description='Unified PCT Training')
    parser.add_argument('--exp_name', type=str, default='exp_pct')
    parser.add_argument('--model', type=str, default='pct', choices=list(MODELS.keys()))
    parser.add_argument('--leaf_size', type=int, default=64, help='Leaf size for kd-tree simple sampler when using pct_kdtree')
    parser.add_argument('--leaf_strategy', type=str, default='random', choices=['random', 'fps'], help='Per-leaf sampling strategy for pct_kdtree')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'])
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=None, help='Min LR for cosine scheduler')
    parser.add_argument('--use_sgd', action='store_true', help='Use SGD instead of Adam')
    parser.add_argument('--lr_scale_sgd', action='store_true', help='Scale SGD lr by 100 like original script')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--save_freq', type=int, default=50, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume/eval')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only (requires --resume)')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--ddp', action='store_true', help='Use DistributedDataParallel')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, default='env://')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank passed by torchrun')
    args = parser.parse_args()

    # Device & seed
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # DDP init if required
    init_distributed(args)

    # Output directories / logger
    if not args.output_dir:
        out_dir, model_dir = setup_output_dir(args)
    else:
        out_dir = args.output_dir
        model_dir = os.path.join(out_dir, 'models')
        if is_main_process():
            os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'run.log')
    io = Logger(log_path) if is_main_process() else None
    if is_main_process():
        _log_args = dict(vars(args))
        _log_args['device'] = str(args.device)
        io.cprint('Args: ' + json.dumps(_log_args, indent=2))

    # Data
    train_dataset = ModelNet40(num_points=args.num_points, partition='train')
    test_dataset = ModelNet40(num_points=args.num_points, partition='test')
    if args.ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=(test_sampler is None),
                             sampler=test_sampler, num_workers=args.workers, pin_memory=True, drop_last=False)

    # Model / Optimizer
    model = build_model(args)
    optimizer, scheduler = build_optimizer(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = cal_loss

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch, best_acc = load_checkpoint(args, model, optimizer if not args.eval_only else None, scaler if not args.eval_only else None)
            if is_main_process():
                io.cprint(f"Resumed from {args.resume} at epoch {start_epoch} best_acc={best_acc:.4f}")
        else:
            if is_main_process():
                io.cprint(f"Resume path {args.resume} not found.")

    if args.eval_only:
        if not args.resume:
            raise ValueError('--eval_only requires --resume checkpoint')
        test_loss, test_acc, test_bal = evaluate(args, model, test_loader, criterion)
        if is_main_process():
            io.cprint(f"Eval :: loss={test_loss:.6f} acc={test_acc:.6f} bal_acc={test_bal:.6f}")
        return

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if args.ddp:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc, train_bal_acc, train_time = train_one_epoch(
            args, model, train_loader, optimizer, scaler, criterion, epoch
        )
        test_loss, test_acc, test_bal_acc = evaluate(args, model, test_loader, criterion)
        if scheduler:
            scheduler.step()
        if is_main_process():
            io.cprint(f"Epoch {epoch} | Train: loss={train_loss:.6f} acc={train_acc:.6f} bal={train_bal_acc:.6f} time={train_time:.2f}s | Test: loss={test_loss:.6f} acc={test_acc:.6f} bal={test_bal_acc:.6f}")
        improved = test_acc > best_acc
        if improved:
            best_acc = test_acc
        if is_main_process() and (improved or (epoch % args.save_freq == 0) or (epoch == args.epochs - 1)):
            ckpt_name = 'best.pt' if improved else f'epoch_{epoch}.pt'
            save_checkpoint(os.path.join(model_dir, ckpt_name), model, optimizer, scaler, epoch, best_acc)
            if improved:
                io.cprint(f"Saved improved checkpoint: {ckpt_name} (acc={best_acc:.6f})")

    if is_main_process():
        io.cprint(f"Training complete. Best test acc: {best_acc:.6f}")

    if args.ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
