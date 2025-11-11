#!/usr/bin/env python3
"""
Проверка эмбеддингов модели
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import argparse
import json

from eegclip.data import ThingsEEGDataset, create_subject_splits, collate_fn
from eegclip.models import EEGCLIPModel
from eegclip.utils import load_checkpoint, get_device, load_config
from eegclip.metrics import compute_retrieval_metrics


def check_embeddings(checkpoint_path=None, config_path=None, data_root="data", n_classes=10, device_str="cuda:0"):
    """Проверка эмбеддингов"""
    
    print("=" * 70)
    print("🔍 ПРОВЕРКА ЭМБЕДДИНГОВ МОДЕЛИ")
    print("=" * 70)
    
    device = get_device(device_str)
    
    # Загружаем конфигурацию из чекпоинта, если есть
    config = None
    if checkpoint_path and Path(checkpoint_path).exists():
        config_dir = Path(checkpoint_path).parent
        config_path = config_dir / "config.json"
        if config_path.exists():
            config = load_config(config_path)
            print(f"✅ Загружена конфигурация: {config_path}")
    
    # Параметры модели из конфига или дефолтные
    if config:
        use_features = config.get('use_features', False)
        n_features = config.get('n_features', 155)  # 19 средних + 136 корреляций для 17 каналов
        eeg_d_model = config.get('eeg_d_model', 256)
        eeg_layers = config.get('eeg_layers', 4)
        eeg_hidden = config.get('eeg_hidden', 512)
        vision_encoder = config.get('vision_encoder', 'openclip_vit_b32')
        freeze_vision = config.get('freeze_vision', True)
        proj_dim = config.get('proj_dim', 512)
        proj_hidden = config.get('proj_hidden', 1024)
        dropout = config.get('dropout', 0.1)
        temporal_pool = config.get('temporal_pool', 'cls')
    else:
        use_features = False
        n_features = 155
        eeg_d_model = 256
        eeg_layers = 4
        eeg_hidden = 512
        vision_encoder = 'openclip_vit_b32'
        freeze_vision = True
        proj_dim = 512
        proj_hidden = 1024
        dropout = 0.1
        temporal_pool = 'cls'
    
    # Subject splits
    all_subjects = list(range(1, 11))
    subject_splits = create_subject_splits(
        all_subjects,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    
    # Датасет
    val_dataset = ThingsEEGDataset(
        data_root=data_root,
        n_classes=n_classes,
        split='val',
        subject_splits=subject_splits,
        eeg_len=100,
        fs=500.0,
        preprocess_eeg=False,
        augment=False,
        use_features=use_features
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Модель
    model = EEGCLIPModel(
        n_channels=17,
        n_timepoints=100,
        use_features=use_features,
        n_features=n_features,
        eeg_d_model=eeg_d_model,
        eeg_layers=eeg_layers,
        eeg_hidden=eeg_hidden,
        vision_encoder=vision_encoder,
        freeze_vision=freeze_vision,
        proj_dim=proj_dim,
        proj_hidden=proj_hidden,
        dropout=dropout,
        temporal_pool=temporal_pool
    ).to(device)
    
    # Загружаем чекпоинт, если есть
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"📝 Загрузка чекпоинта: {checkpoint_path}")
        load_checkpoint(Path(checkpoint_path), model)
        print(f"✅ Загружен чекпоинт: {checkpoint_path}")
    else:
        print("📝 Используется модель с начальными весами")
    
    model.eval()
    
    # Получаем все данные из валидационного датасета
    all_eeg_emb = []
    all_img_emb = []
    all_class_indices = []
    
    print(f"\n📊 Загрузка всех данных из валидационного датасета...")
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            image = batch['image'].to(device)
            eeg_emb, img_emb = model(eeg, image)
            all_eeg_emb.append(eeg_emb.cpu())
            all_img_emb.append(img_emb.cpu())
            all_class_indices.extend(batch['class_idx'].tolist())
    
    # Объединяем все эмбеддинги
    eeg_emb = torch.cat(all_eeg_emb, dim=0)
    img_emb = torch.cat(all_img_emb, dim=0)
    
    print(f"   Всего образцов: {eeg_emb.shape[0]}")
    print(f"   Уникальных классов: {len(set(all_class_indices))}")
    
    # Для анализа берем один батч
    batch = next(iter(val_loader))
    eeg_sample = batch['eeg'].to(device)
    image_sample = batch['image'].to(device)
    
    print(f"\n📊 Пример батча:")
    print(f"   EEG shape: {eeg_sample.shape}")
    print(f"   Image shape: {image_sample.shape}")
    print(f"   Class indices: {batch['class_idx'].tolist()}")
    
    print(f"\n📊 Эмбеддинги:")
    print(f"   EEG embeddings shape: {eeg_emb.shape}")
    print(f"   Image embeddings shape: {img_emb.shape}")
    
    # Проверяем нормализацию
    eeg_norms = torch.norm(eeg_emb, dim=1)
    img_norms = torch.norm(img_emb, dim=1)
    
    print(f"\n📏 L2 нормы:")
    print(f"   EEG norms: min={eeg_norms.min():.4f}, max={eeg_norms.max():.4f}, mean={eeg_norms.mean():.4f}")
    print(f"   Image norms: min={img_norms.min():.4f}, max={img_norms.max():.4f}, mean={img_norms.mean():.4f}")
    
    # Матрица сходства (перемещаем на устройство для вычислений)
    eeg_emb_device = eeg_emb.to(device)
    img_emb_device = img_emb.to(device)
    similarity = eeg_emb_device @ img_emb_device.T
    print(f"\n📊 Матрица сходства:")
    print(f"   Shape: {similarity.shape}")
    print(f"   Диагональ (правильные пары): {torch.diag(similarity).tolist()}")
    print(f"   Диагональ mean: {torch.diag(similarity).mean():.4f}")
    print(f"   Диагональ std: {torch.diag(similarity).std():.4f}")
    print(f"   Вне диагонали mean: {similarity[~torch.eye(similarity.shape[0], dtype=bool)].mean():.4f}")
    print(f"   Вне диагонали std: {similarity[~torch.eye(similarity.shape[0], dtype=bool)].std():.4f}")
    
    # Проверяем, правильно ли модель различает пары
    diag_similarity = torch.diag(similarity)
    off_diag_mean = similarity[~torch.eye(similarity.shape[0], dtype=bool)].mean()
    
    print(f"\n🔍 Анализ различимости:")
    print(f"   Среднее сходство правильных пар: {diag_similarity.mean():.4f}")
    print(f"   Среднее сходство неправильных пар: {off_diag_mean:.4f}")
    print(f"   Разница: {diag_similarity.mean() - off_diag_mean:.4f}")
    
    if diag_similarity.mean() > off_diag_mean:
        print("   ✅ Правильные пары имеют большее сходство")
    else:
        print("   ❌ ПРОБЛЕМА: Правильные пары НЕ имеют большего сходства!")
    
    # Вычисляем метрики на всех данных
    print(f"\n📊 Анализ на всех {eeg_emb.shape[0]} образцах:")
    metrics = compute_retrieval_metrics(eeg_emb.to(device), img_emb.to(device), k_list=[1, 5, 10])
    
    print(f"\n📈 Метрики retrieval:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # Проверяем logit_scale
    logit_scale_param = model.get_logit_scale_param()  # Сам параметр
    temperature = model.get_logit_scale()  # Температура (exp с клиппингом)
    print(f"\n🌡️  Logit scale (температура):")
    print(f"   logit_scale (параметр): {logit_scale_param.item():.4f}")
    print(f"   temperature: {temperature.item():.4f}")
    
    # Проверяем, как температура влияет на сходство
    scaled_similarity = temperature * similarity
    print(f"\n📊 Масштабированная матрица сходства (с температурой):")
    print(f"   Диагональ mean: {torch.diag(scaled_similarity).mean():.4f}")
    print(f"   Вне диагонали mean: {scaled_similarity[~torch.eye(scaled_similarity.shape[0], dtype=bool)].mean():.4f}")
    
    # Проверяем предсказания
    pred_eeg2img = similarity.argmax(dim=1)
    pred_img2eeg = similarity.argmax(dim=0)
    labels = torch.arange(similarity.shape[0], device=device)
    
    eeg2img_acc = (pred_eeg2img == labels).float().mean().item()
    img2eeg_acc = (pred_img2eeg == labels).float().mean().item()
    
    print(f"\n🎯 Точность предсказаний:")
    print(f"   EEG→Image: {eeg2img_acc:.4f} ({eeg2img_acc*100:.2f}%)")
    print(f"   Image→EEG: {img2eeg_acc:.4f} ({img2eeg_acc*100:.2f}%)")
    print(f"   Baseline (случайное): {1.0/similarity.shape[0]:.4f} ({100.0/similarity.shape[0]:.2f}%)")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check embeddings of EEG-CLIP model')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to config.json (optional, auto-detected from checkpoint dir)')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory with data')
    parser.add_argument('--n_classes', type=int, default=10,
                       help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0, cpu, etc.)')
    
    args = parser.parse_args()
    
    # Если указан чекпоинт, проверяем только его
    if args.checkpoint_path:
        metrics = check_embeddings(
            checkpoint_path=args.checkpoint_path,
            config_path=args.config_path,
            data_root=args.data_root,
            n_classes=args.n_classes,
            device_str=args.device
        )
    else:
        # Проверяем модель с начальными весами
        print("\n" + "="*70)
        print("1. ПРОВЕРКА МОДЕЛИ С НАЧАЛЬНЫМИ ВЕСАМИ")
        print("="*70)
        metrics_init = check_embeddings(
            data_root=args.data_root,
            n_classes=args.n_classes,
            device_str=args.device
        )
        
        # Проверяем обученную модель (если есть дефолтный чекпоинт)
        checkpoint_path = "checkpoints_test/best.pt"
        if Path(checkpoint_path).exists():
            print("\n" + "="*70)
            print("2. ПРОВЕРКА ОБУЧЕННОЙ МОДЕЛИ")
            print("="*70)
            metrics_trained = check_embeddings(
                checkpoint_path=checkpoint_path,
                data_root=args.data_root,
                n_classes=args.n_classes,
                device_str=args.device
            )
            
            print("\n" + "="*70)
            print("📊 СРАВНЕНИЕ")
            print("="*70)
            print(f"Начальная Recall@1: {metrics_init['eeg2img_recall@1']:.4f}")
            print(f"Обученная Recall@1: {metrics_trained['eeg2img_recall@1']:.4f}")
            print(f"Улучшение: {metrics_trained['eeg2img_recall@1'] - metrics_init['eeg2img_recall@1']:.4f}")
        else:
            print(f"\n⚠️  Чекпоинт не найден: {checkpoint_path}")

