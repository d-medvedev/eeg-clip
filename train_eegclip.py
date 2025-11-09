#!/usr/bin/env python3
"""
Обучение EEG-CLIP модели
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import math

from eegclip.models import EEGCLIPModel
from eegclip.data import ThingsEEGDataset, create_subject_splits, collate_fn
from eegclip.losses import InfoNCELoss, AuxiliaryClassificationLoss
from eegclip.metrics import compute_retrieval_metrics
from eegclip.utils import set_seed, save_checkpoint, load_checkpoint, save_config, get_device, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description='Train EEG-CLIP model')
    
    # Data
    parser.add_argument('--data_root', type=str, default='data', help='Root directory with eeg/ and images/')
    parser.add_argument('--n_classes', type=int, default=None, help='Number of classes to use (None = all)')
    parser.add_argument('--eeg_len', type=int, default=100, help='Fixed EEG length in samples')
    parser.add_argument('--fs', type=float, default=500.0, help='Sampling frequency')
    parser.add_argument('--preprocess_eeg', action='store_true', help='Enable EEG preprocessing (filtering, normalization). Data is already preprocessed by default.')
    parser.add_argument('--bandpass', type=float, nargs=2, default=[0.5, 45.0], help='Bandpass filter [low, high] (only if --preprocess_eeg)')
    parser.add_argument('--notch', type=float, default=None, help='Notch filter frequency (50 or 60) (only if --preprocess_eeg)')
    
    # Augmentation
    parser.add_argument('--augment_eeg', action='store_true', help='Enable EEG augmentation')
    parser.add_argument('--noise_std', type=float, default=0.01, help='Gaussian noise std')
    parser.add_argument('--jitter_ms', type=float, default=20.0, help='Time jitter in ms')
    parser.add_argument('--time_mask_prob', type=float, default=0.2, help='Time mask probability')
    parser.add_argument('--channel_drop_prob', type=float, default=0.1, help='Channel dropout probability')
    
    # Model
    parser.add_argument('--vision_encoder', type=str, default='openclip_vit_b32', 
                       choices=['openclip_vit_b32', 'torchvision_vit_b32'],
                       help='Vision encoder type')
    parser.add_argument('--freeze_vision', action='store_true', default=True, help='Freeze vision encoder')
    parser.add_argument('--eeg_d_model', type=int, default=256, help='EEG encoder d_model')
    parser.add_argument('--eeg_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--eeg_hidden', type=int, default=512, help='EEG encoder hidden dim')
    parser.add_argument('--proj_dim', type=int, default=512, help='Projection dimension')
    parser.add_argument('--proj_hidden', type=int, default=1024, help='Projection head hidden dim')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--temporal_pool', type=str, default='cls', choices=['cls', 'mean', 'max'],
                       help='Temporal pooling method')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Validation split
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation subject ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test subject ratio')
    
    # System
    parser.add_argument('--devices', type=str, default='1', help='Device(s) to use')
    parser.add_argument('--precision', type=str, default='amp', choices=['amp', 'fp32'], help='Precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Checkpoints
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard log directory')
    parser.add_argument('--log_every', type=int, default=100, help='Log every N steps')
    
    return parser.parse_args()


def train_epoch(
    model: EEGCLIPModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    args
) -> int:
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        eeg = batch['eeg'].to(device)
        image = batch['image'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if args.precision == 'amp':
            with autocast():
                eeg_emb, img_emb = model(eeg, image)
                temperature = model.get_logit_scale()  # Это уже exp(logit_scale)
                loss_fn = InfoNCELoss()
                loss = loss_fn(eeg_emb, img_emb, temperature)
        else:
            eeg_emb, img_emb = model(eeg, image)
            temperature = model.get_logit_scale()  # Это уже exp(logit_scale)
            loss_fn = InfoNCELoss()
            loss = loss_fn(eeg_emb, img_emb, temperature)
        
        # Backward
        if args.precision == 'amp':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            # Ограничиваем параметр logit_scale после шага оптимизатора (для AMP)
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100.0))
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # Ограничиваем параметр logit_scale после шага оптимизатора
        # Это предотвращает неконтролируемый рост температуры
        with torch.no_grad():
            model.logit_scale.clamp_(max=math.log(100.0))
        
        total_loss += loss.item()
        n_batches += 1
        
        # Логирование
        logit_scale_param = model.get_logit_scale_param()  # Сам параметр для логирования
        if global_step % args.log_every == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/logit_scale_param', logit_scale_param.item(), global_step)
            writer.add_scalar('train/temperature', temperature.item(), global_step)
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'temp': f"{temperature.item():.2f}", 'log_scale': f"{logit_scale_param.item():.2f}"})
        global_step += 1
    
    avg_loss = total_loss / n_batches
    return global_step, avg_loss


@torch.no_grad()
def validate(
    model: EEGCLIPModel,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int
) -> dict:
    """Валидация"""
    model.eval()
    all_eeg_emb = []
    all_img_emb = []
    
    for batch in tqdm(dataloader, desc='Validation'):
        eeg = batch['eeg'].to(device)
        image = batch['image'].to(device)
        
        eeg_emb, img_emb = model(eeg, image)
        
        all_eeg_emb.append(eeg_emb.cpu())
        all_img_emb.append(img_emb.cpu())
    
    # Объединяем все эмбеддинги
    eeg_emb = torch.cat(all_eeg_emb, dim=0)
    img_emb = torch.cat(all_img_emb, dim=0)
    
    # Вычисляем метрики
    metrics = compute_retrieval_metrics(eeg_emb, img_emb, k_list=[1, 5, 10])
    
    # Логирование
    for key, value in metrics.items():
        writer.add_scalar(f'val/{key}', value, global_step)
    
    return metrics


def main():
    args = parse_args()
    
    # Фиксация сида
    set_seed(args.seed)
    
    # Устройство
    device = get_device(args.devices)
    print(f"🔧 Устройство: {device}")
    
    # Создаем директории
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем конфигурацию
    config_dict = vars(args)
    save_config(config_dict, save_dir)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Subject splits
    all_subjects = list(range(1, 11))
    subject_splits = create_subject_splits(
        all_subjects,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    print(f"📊 Subject splits:")
    for split, subjects in subject_splits.items():
        print(f"   {split}: {subjects}")
    
    # Датасеты
    train_dataset = ThingsEEGDataset(
        data_root=args.data_root,
        n_classes=args.n_classes,
        split='train',
        subject_splits=subject_splits,
        eeg_len=args.eeg_len,
        fs=args.fs,
        preprocess_eeg=args.preprocess_eeg,  # По умолчанию False - данные уже предобработаны
        bandpass=tuple(args.bandpass),
        notch=args.notch,
        augment=args.augment_eeg,
        noise_std=args.noise_std,
        jitter_ms=args.jitter_ms,
        time_mask_prob=args.time_mask_prob,
        channel_drop_prob=args.channel_drop_prob
    )
    
    val_dataset = ThingsEEGDataset(
        data_root=args.data_root,
        n_classes=args.n_classes,
        split='val',
        subject_splits=subject_splits,
        eeg_len=args.eeg_len,
        fs=args.fs,
        preprocess_eeg=args.preprocess_eeg,  # По умолчанию False - данные уже предобработаны
        bandpass=tuple(args.bandpass),
        notch=args.notch,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Модель
    model = EEGCLIPModel(
        n_channels=17,  # Things-EEG
        n_timepoints=args.eeg_len,
        eeg_d_model=args.eeg_d_model,
        eeg_layers=args.eeg_layers,
        eeg_hidden=args.eeg_hidden,
        vision_encoder=args.vision_encoder,
        freeze_vision=args.freeze_vision,
        proj_dim=args.proj_dim,
        proj_hidden=args.proj_hidden,
        dropout=args.dropout,
        temporal_pool=args.temporal_pool
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"📊 Модель: {n_params:,} обучаемых параметров")
    
    # Оптимизатор и scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.lr * 0.01
    )
    
    # Mixed precision
    scaler = GradScaler() if args.precision == 'amp' else None
    
    # Resume
    start_epoch = 0
    global_step = 0
    best_metric = 0.0
    
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), model, optimizer, scheduler, scaler)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['step']
        best_metric = checkpoint['metrics'].get('val_eeg2img_recall@1', 0.0)
        print(f"✅ Возобновлено с эпохи {start_epoch}, шаг {global_step}")
    
    # Обучение
    print(f"\n🚀 Начало обучения на {len(train_dataset)} образцах")
    
    for epoch in range(start_epoch, args.epochs):
        # Обучение
        global_step, train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, writer, global_step, args
        )
        
        # Scheduler step
        if global_step > warmup_steps:
            scheduler.step()
        else:
            # Warmup
            lr = args.lr * (global_step / warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Валидация
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Каждые 5 эпох
            val_metrics = validate(model, val_loader, device, epoch, writer, global_step)
            
            # Сохраняем лучший чекпоинт
            current_metric = val_metrics.get('eeg2img_recall@1', 0.0)
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
            
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, global_step, val_metrics,
                save_dir, is_best=is_best
            )
            
            print(f"\n📊 Epoch {epoch}: Train Loss={train_loss:.4f}, "
                  f"Val Recall@1={current_metric:.4f}, Best={best_metric:.4f}")
        
        # Периодическое сохранение
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, global_step, {},
                save_dir, is_best=False
            )
    
    print(f"\n✅ Обучение завершено! Лучший Recall@1: {best_metric:.4f}")
    writer.close()


if __name__ == '__main__':
    main()

