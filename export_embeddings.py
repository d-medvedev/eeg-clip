#!/usr/bin/env python3
"""
Экспорт эмбеддингов для оценки
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

from eegclip.models import EEGCLIPModel
from eegclip.data import ThingsEEGDataset, create_subject_splits, collate_fn
from eegclip.utils import load_checkpoint, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Export embeddings')
    
    parser.add_argument('--data_root', type=str, default='data', help='Root directory')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--devices', type=str, default='1', help='Device')
    parser.add_argument('--out', type=str, default='embeddings.npz', help='Output file')
    
    # Model args (должны совпадать с обучением)
    parser.add_argument('--eeg_len', type=int, default=100)
    parser.add_argument('--vision_encoder', type=str, default='openclip_vit_b32')
    parser.add_argument('--freeze_vision', action='store_true', default=True)
    parser.add_argument('--eeg_d_model', type=int, default=256)
    parser.add_argument('--eeg_layers', type=int, default=4)
    parser.add_argument('--eeg_hidden', type=int, default=512)
    parser.add_argument('--proj_dim', type=int, default=512)
    parser.add_argument('--proj_hidden', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--temporal_pool', type=str, default='cls')
    
    return parser.parse_args()


@torch.no_grad()
def export_embeddings(model, dataloader, device, split_name):
    """Экспорт эмбеддингов"""
    model.eval()
    
    all_eeg_emb = []
    all_img_emb = []
    record_ids = []
    image_ids = []
    subject_ids = []
    class_indices = []
    
    for batch in tqdm(dataloader, desc='Exporting'):
        eeg = batch['eeg'].to(device)
        image = batch['image'].to(device)
        
        eeg_emb, img_emb = model(eeg, image)
        
        all_eeg_emb.append(eeg_emb.cpu().numpy())
        all_img_emb.append(img_emb.cpu().numpy())
        record_ids.extend(batch['record_id'])
        image_ids.extend(batch['image_id'])
        subject_ids.append(batch['subject_id'].numpy())
        class_indices.append(batch['class_idx'].numpy())
    
    # Объединяем
    eeg_emb = np.concatenate(all_eeg_emb, axis=0)
    img_emb = np.concatenate(all_img_emb, axis=0)
    subject_ids = np.concatenate(subject_ids, axis=0)
    class_indices = np.concatenate(class_indices, axis=0)
    
    return {
        'eeg': eeg_emb,
        'img': img_emb,
        'record_ids': record_ids,
        'image_ids': image_ids,
        'subject_ids': subject_ids,
        'class_indices': class_indices,
        'split': [split_name] * len(record_ids)
    }


def main():
    args = parse_args()
    
    device = get_device(args.devices)
    print(f"🔧 Устройство: {device}")
    
    # Subject splits (должны совпадать с обучением)
    all_subjects = list(range(1, 11))
    subject_splits = create_subject_splits(
        all_subjects,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    
    # Датасет
    dataset = ThingsEEGDataset(
        data_root=args.data_root,
        n_classes=args.n_classes,
        split=args.split,
        subject_splits=subject_splits,
        eeg_len=args.eeg_len,
        fs=500.0,
        preprocess_eeg=False,  # Данные уже предобработаны
        augment=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Модель
    model = EEGCLIPModel(
        n_channels=17,
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
    
    # Загрузка чекпоинта
    load_checkpoint(Path(args.ckpt), model)
    print(f"✅ Загружен чекпоинт: {args.ckpt}")
    
    # Экспорт
    embeddings = export_embeddings(model, dataloader, device, args.split)
    
    # Сохранение
    np.savez(args.out, **embeddings)
    print(f"✅ Эмбеддинги сохранены: {args.out}")
    print(f"   EEG: {embeddings['eeg'].shape}")
    print(f"   Image: {embeddings['img'].shape}")


if __name__ == '__main__':
    main()

