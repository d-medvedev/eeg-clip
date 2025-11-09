#!/usr/bin/env python3
"""
Проверка соответствия данных: EEG ↔ Image
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from eegclip.data import ThingsEEGDataset, create_subject_splits, collate_fn


def check_data_alignment(data_root="data", n_classes=10, n_samples_to_check=10):
    """Проверка соответствия данных"""
    
    print("=" * 70)
    print("🔍 ПРОВЕРКА СООТВЕТСТВИЯ ДАННЫХ: EEG ↔ Image")
    print("=" * 70)
    
    # Subject splits
    all_subjects = list(range(1, 11))
    subject_splits = create_subject_splits(
        all_subjects,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    
    # Датасет
    dataset = ThingsEEGDataset(
        data_root=data_root,
        n_classes=n_classes,
        split='train',
        subject_splits=subject_splits,
        eeg_len=100,
        fs=500.0,
        preprocess_eeg=False,
        augment=False
    )
    
    print(f"\n📊 Загружено образцов: {len(dataset)}")
    
    # Проверяем несколько образцов
    print(f"\n{'='*70}")
    print(f"📋 ПРОВЕРКА {n_samples_to_check} ОБРАЗЦОВ")
    print(f"{'='*70}")
    
    issues = []
    
    for idx in range(min(n_samples_to_check, len(dataset))):
        sample = dataset.samples[idx]
        
        # Проверяем соответствие
        eeg_shape = sample['eeg_data'].shape
        image_path = sample['image_path']
        class_idx = sample['class_idx']
        class_name = sample['class_name']
        record_id = sample['record_id']
        subject_id = sample['subject_id']
        
        # Проверяем, что изображение существует
        if not image_path.exists():
            issues.append(f"❌ Образец {idx}: изображение не найдено: {image_path}")
            continue
        
        # Проверяем, что имя класса в пути соответствует class_idx
        image_class_name = image_path.parent.name
        expected_class_name = f"{class_idx+1:05d}_{class_name}" if class_name else f"{class_idx+1:05d}_unknown"
        
        # Извлекаем номер класса из пути изображения
        path_parts = image_path.parent.name.split('_')
        if len(path_parts) > 0:
            try:
                image_class_num = int(path_parts[0]) - 1  # 0-based
            except ValueError:
                image_class_num = -1
        else:
            image_class_num = -1
        
        # Проверка соответствия
        is_correct = (image_class_num == class_idx)
        
        status = "✅" if is_correct else "❌"
        print(f"\n{status} Образец {idx}:")
        print(f"   Record ID: {record_id}")
        print(f"   Subject: {subject_id}")
        print(f"   Class Index: {class_idx}")
        print(f"   Class Name: {class_name}")
        print(f"   Image Path: {image_path.name}")
        print(f"   Image Class from path: {image_class_num} (expected: {class_idx})")
        print(f"   EEG Shape: {eeg_shape}")
        
        if not is_correct:
            issues.append(f"Образец {idx}: class_idx={class_idx}, но изображение из класса {image_class_num}")
    
    # Проверяем батч
    print(f"\n{'='*70}")
    print("📦 ПРОВЕРКА БАТЧА")
    print(f"{'='*70}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,  # Не перемешиваем для проверки
        num_workers=0,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    
    print(f"\nРазмер батча: {batch['eeg'].shape[0]}")
    print(f"EEG shape: {batch['eeg'].shape}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Subject IDs: {batch['subject_id'].tolist()}")
    print(f"Class Indices: {batch['class_idx'].tolist()}")
    print(f"\nRecord IDs:")
    for i, rid in enumerate(batch['record_id']):
        print(f"  [{i}] {rid}")
    print(f"\nImage IDs:")
    for i, img_id in enumerate(batch['image_id']):
        print(f"  [{i}] {img_id}")
    
    # Проверяем, что в батче правильное соответствие
    # В InfoNCE loss предполагается, что i-й EEG соответствует i-му изображению
    print(f"\n{'='*70}")
    print("🔬 ПРОВЕРКА СООТВЕТСТВИЯ В БАТЧЕ")
    print(f"{'='*70}")
    
    # Для каждого элемента в батче проверяем, что class_idx совпадает
    batch_issues = []
    for i in range(batch['eeg'].shape[0]):
        # Получаем оригинальный образец
        sample_idx = i  # Если shuffle=False, то индекс в батче = индекс в датасете
        if sample_idx < len(dataset):
            original_sample = dataset.samples[sample_idx]
            batch_class_idx = batch['class_idx'][i].item()
            original_class_idx = original_sample['class_idx']
            
            if batch_class_idx != original_class_idx:
                batch_issues.append(
                    f"Элемент {i} в батче: class_idx={batch_class_idx}, "
                    f"но в датасете class_idx={original_class_idx}"
                )
    
    # Проверяем уникальность пар
    print(f"\n{'='*70}")
    print("🔍 ПРОВЕРКА УНИКАЛЬНОСТИ ПАР")
    print(f"{'='*70}")
    
    # Проверяем, что в батче нет дубликатов пар
    pairs = [(batch['record_id'][i], batch['image_id'][i]) 
             for i in range(batch['eeg'].shape[0])]
    unique_pairs = set(pairs)
    
    print(f"Всего пар в батче: {len(pairs)}")
    print(f"Уникальных пар: {len(unique_pairs)}")
    
    if len(pairs) != len(unique_pairs):
        print("⚠️  ВНИМАНИЕ: Есть дубликаты пар в батче!")
        from collections import Counter
        pair_counts = Counter(pairs)
        duplicates = {pair: count for pair, count in pair_counts.items() if count > 1}
        print(f"Дубликаты: {duplicates}")
    else:
        print("✅ Все пары уникальны")
    
    # Итоговый отчет
    print(f"\n{'='*70}")
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*70}")
    
    if issues:
        print(f"\n❌ Найдено проблем: {len(issues)}")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ Проблем с соответствием не найдено")
    
    if batch_issues:
        print(f"\n❌ Проблемы в батче: {len(batch_issues)}")
        for issue in batch_issues:
            print(f"   - {issue}")
    else:
        print("\n✅ Батч корректен")
    
    # Проверяем статистику по классам
    print(f"\n{'='*70}")
    print("📈 СТАТИСТИКА ПО КЛАССАМ")
    print(f"{'='*70}")
    
    class_counts = {}
    for sample in dataset.samples:
        class_idx = sample['class_idx']
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
    
    print(f"\nРаспределение по классам:")
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        sample = next(s for s in dataset.samples if s['class_idx'] == class_idx)
        class_name = sample.get('class_name', 'unknown')
        print(f"  Class {class_idx:2d} ({class_name:30s}): {count:4d} образцов")
    
    return issues, batch_issues


if __name__ == '__main__':
    issues, batch_issues = check_data_alignment(n_classes=10, n_samples_to_check=20)
    
    if issues or batch_issues:
        print(f"\n⚠️  ВНИМАНИЕ: Обнаружены проблемы с соответствием данных!")
        exit(1)
    else:
        print(f"\n✅ Все проверки пройдены успешно!")
        exit(0)

