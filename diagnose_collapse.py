#!/usr/bin/env python3
"""
Диагностика коллапса эмбеддингов и проблем с обучением
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


def diagnose_collapse(checkpoint_path, data_root="data", n_classes=10, device_str="cuda:0", n_samples=100):
    """Диагностика коллапса эмбеддингов"""
    
    print("=" * 70)
    print("🔬 ДИАГНОСТИКА КОЛЛАПСА ЭМБЕДДИНГОВ")
    print("=" * 70)
    
    device = get_device(device_str)
    
    # Загружаем конфигурацию
    config_dir = Path(checkpoint_path).parent
    config_path = config_dir / "config.json"
    config = load_config(config_path)
    
    # Параметры модели
    eeg_d_model = config.get('eeg_d_model', 256)
    eeg_layers = config.get('eeg_layers', 4)
    eeg_hidden = config.get('eeg_hidden', 512)
    vision_encoder = config.get('vision_encoder', 'openclip_vit_b32')
    freeze_vision = config.get('freeze_vision', True)
    proj_dim = config.get('proj_dim', 512)
    proj_hidden = config.get('proj_hidden', 1024)
    dropout = config.get('dropout', 0.1)
    temporal_pool = config.get('temporal_pool', 'cls')
    
    # Модель
    model = EEGCLIPModel(
        n_channels=17,
        n_timepoints=100,
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
    
    # Загружаем чекпоинт
    checkpoint = load_checkpoint(checkpoint_path, model, None, None, device)
    model.eval()
    
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
        split='val',
        subject_splits=subject_splits,
        eeg_len=100,
        fs=500.0,
        preprocess_eeg=False,
        augment=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"\n📊 Загружено образцов: {len(dataset)}")
    
    # Собираем эмбеддинги и метки
    all_eeg_emb = []
    all_img_emb = []
    all_class_idx = []
    all_subject_id = []
    
    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            image = batch['image'].to(device)
            class_idx = batch['class_idx']
            subject_id = batch['subject_id']
            
            eeg_emb, img_emb = model(eeg, image)
            
            all_eeg_emb.append(eeg_emb.cpu())
            all_img_emb.append(img_emb.cpu())
            all_class_idx.append(class_idx)
            all_subject_id.append(subject_id)
            
            if len(all_eeg_emb) * 32 >= n_samples:
                break
    
    # Объединяем
    eeg_emb = torch.cat(all_eeg_emb, dim=0)
    img_emb = torch.cat(all_img_emb, dim=0)
    class_idx = torch.cat(all_class_idx, dim=0)
    subject_id = torch.cat(all_subject_id, dim=0)
    
    n_samples = min(n_samples, len(eeg_emb))
    eeg_emb = eeg_emb[:n_samples]
    img_emb = img_emb[:n_samples]
    class_idx = class_idx[:n_samples]
    subject_id = subject_id[:n_samples]
    
    print(f"\n📊 Анализируем {n_samples} образцов")
    
    # 1. Проверка коллапса эмбеддингов
    print(f"\n{'='*70}")
    print("1️⃣  ПРОВЕРКА КОЛЛАПСА ЭМБЕДДИНГОВ")
    print(f"{'='*70}")
    
    # Средние эмбеддинги по классам
    unique_classes = torch.unique(class_idx)
    print(f"\n📊 Уникальные классы: {unique_classes.tolist()}")
    
    eeg_emb_by_class = {}
    img_emb_by_class = {}
    
    for cls in unique_classes:
        mask = (class_idx == cls)
        eeg_emb_by_class[cls.item()] = eeg_emb[mask]
        img_emb_by_class[cls.item()] = img_emb[mask]
    
    # Средние эмбеддинги
    print(f"\n📊 Средние эмбеддинги по классам:")
    for cls in sorted(unique_classes):
        cls_eeg = eeg_emb_by_class[cls.item()]
        cls_img = img_emb_by_class[cls.item()]
        
        mean_eeg = cls_eeg.mean(dim=0)
        mean_img = cls_img.mean(dim=0)
        
        std_eeg = cls_eeg.std(dim=0).mean().item()
        std_img = cls_img.std(dim=0).mean().item()
        
        print(f"   Class {cls.item():2d}: EEG std={std_eeg:.6f}, Image std={std_img:.6f}, "
              f"samples={len(cls_eeg)}")
    
    # 2. Сходство между классами
    print(f"\n{'='*70}")
    print("2️⃣  СХОДСТВО МЕЖДУ КЛАССАМИ")
    print(f"{'='*70}")
    
    # Средние эмбеддинги для каждого класса
    mean_eeg_by_class = {}
    mean_img_by_class = {}
    
    for cls in unique_classes:
        mean_eeg_by_class[cls.item()] = eeg_emb_by_class[cls.item()].mean(dim=0, keepdim=True)
        mean_img_by_class[cls.item()] = img_emb_by_class[cls.item()].mean(dim=0, keepdim=True)
    
    # Матрица сходства между классами (EEG)
    print(f"\n📊 Матрица сходства между классами (EEG эмбеддинги):")
    n_classes_found = len(unique_classes)
    similarity_matrix_eeg = torch.zeros(n_classes_found, n_classes_found)
    
    for i, cls_i in enumerate(sorted(unique_classes)):
        for j, cls_j in enumerate(sorted(unique_classes)):
            sim = (mean_eeg_by_class[cls_i.item()] @ mean_eeg_by_class[cls_j.item()].T).item()
            similarity_matrix_eeg[i, j] = sim
            if i == j:
                print(f"   Class {cls_i.item():2d} <-> Class {cls_j.item():2d}: {sim:.4f} (self)")
    
    # Диагональ vs вне диагонали
    diagonal_eeg = torch.diag(similarity_matrix_eeg)
    off_diagonal_eeg = similarity_matrix_eeg[~torch.eye(n_classes_found, dtype=bool)]
    
    print(f"\n   Диагональ (self-similarity): mean={diagonal_eeg.mean():.4f}, std={diagonal_eeg.std():.4f}")
    print(f"   Вне диагонали (cross-class): mean={off_diagonal_eeg.mean():.4f}, std={off_diagonal_eeg.std():.4f}")
    print(f"   Разница: {diagonal_eeg.mean() - off_diagonal_eeg.mean():.4f}")
    
    if diagonal_eeg.mean() - off_diagonal_eeg.mean() < 0.1:
        print(f"   ❌ ПРОБЛЕМА: Классы не различаются!")
    else:
        print(f"   ✅ Классы различаются")
    
    # 3. Сходство между правильными и неправильными парами
    print(f"\n{'='*70}")
    print("3️⃣  СХОДСТВО ПРАВИЛЬНЫХ VS НЕПРАВИЛЬНЫХ ПАР")
    print(f"{'='*70}")
    
    # Для каждого образца находим правильную и неправильную пару
    correct_similarities = []
    incorrect_similarities = []
    
    for i in range(n_samples):
        cls_i = class_idx[i].item()
        eeg_i = eeg_emb[i:i+1]
        
        # Правильная пара (из того же класса)
        correct_mask = (class_idx == cls_i)
        correct_img = img_emb[correct_mask]
        if len(correct_img) > 0:
            correct_sim = (eeg_i @ correct_img.T).mean().item()
            correct_similarities.append(correct_sim)
        
        # Неправильные пары (из других классов)
        incorrect_mask = (class_idx != cls_i)
        if incorrect_mask.any():
            incorrect_img = img_emb[incorrect_mask]
            incorrect_sim = (eeg_i @ incorrect_img.T).mean().item()
            incorrect_similarities.append(incorrect_sim)
    
    if correct_similarities and incorrect_similarities:
        print(f"\n   Правильные пары: mean={np.mean(correct_similarities):.4f}, std={np.std(correct_similarities):.4f}")
        print(f"   Неправильные пары: mean={np.mean(incorrect_similarities):.4f}, std={np.std(incorrect_similarities):.4f}")
        diff = np.mean(correct_similarities) - np.mean(incorrect_similarities)
        print(f"   Разница: {diff:.4f}")
        
        if diff < 0.05:
            print(f"   ❌ ПРОБЛЕМА: Правильные и неправильные пары не различаются!")
        else:
            print(f"   ✅ Правильные пары имеют большее сходство")
    
    # 4. Разброс эмбеддингов
    print(f"\n{'='*70}")
    print("4️⃣  РАЗБРОС ЭМБЕДДИНГОВ")
    print(f"{'='*70}")
    
    # Общий разброс
    eeg_std = eeg_emb.std(dim=0).mean().item()
    img_std = img_emb.std(dim=0).mean().item()
    
    print(f"\n   Общий разброс (std по всем измерениям):")
    print(f"   EEG: {eeg_std:.6f}")
    print(f"   Image: {img_std:.6f}")
    
    if eeg_std < 0.01 or img_std < 0.01:
        print(f"   ❌ ПРОБЛЕМА: Эмбеддинги коллапсировали (слишком маленький разброс)!")
    else:
        print(f"   ✅ Эмбеддинги имеют достаточный разброс")
    
    # 5. Различия между субъектами
    print(f"\n{'='*70}")
    print("5️⃣  РАЗЛИЧИЯ МЕЖДУ СУБЪЕКТАМИ")
    print(f"{'='*70}")
    
    unique_subjects = torch.unique(subject_id)
    print(f"\n📊 Уникальные субъекты: {unique_subjects.tolist()}")
    
    # Средние эмбеддинги по субъектам
    eeg_emb_by_subject = {}
    for subj in unique_subjects:
        mask = (subject_id == subj)
        eeg_emb_by_subject[subj.item()] = eeg_emb[mask]
    
    # Сходство между субъектами
    mean_eeg_by_subject = {}
    for subj in unique_subjects:
        mean_eeg_by_subject[subj.item()] = eeg_emb_by_subject[subj.item()].mean(dim=0, keepdim=True)
    
    if len(unique_subjects) > 1:
        subj_list = sorted(unique_subjects)
        subj_similarities = []
        for i, subj_i in enumerate(subj_list):
            for j, subj_j in enumerate(subj_list):
                if i != j:
                    sim = (mean_eeg_by_subject[subj_i.item()] @ mean_eeg_by_subject[subj_j.item()].T).item()
                    subj_similarities.append(sim)
        
        print(f"\n   Сходство между субъектами: mean={np.mean(subj_similarities):.4f}, std={np.std(subj_similarities):.4f}")
        
        # Сравниваем с сходством между классами
        if np.mean(subj_similarities) > off_diagonal_eeg.mean():
            print(f"   ⚠️  ВНИМАНИЕ: Субъекты различаются больше, чем классы!")
            print(f"      Это может означать, что модель учится различать субъектов, а не классы.")
    
    # Итоговый вывод
    print(f"\n{'='*70}")
    print("📊 ИТОГОВЫЙ ДИАГНОЗ")
    print(f"{'='*70}")
    
    issues = []
    
    if diagonal_eeg.mean() - off_diagonal_eeg.mean() < 0.1:
        issues.append("❌ Классы не различаются в эмбеддингах")
    
    if correct_similarities and incorrect_similarities:
        if np.mean(correct_similarities) - np.mean(incorrect_similarities) < 0.05:
            issues.append("❌ Правильные и неправильные пары не различаются")
    
    if eeg_std < 0.01 or img_std < 0.01:
        issues.append("❌ Эмбеддинги коллапсировали (слишком маленький разброс)")
    
    if issues:
        print(f"\n⚠️  Обнаружены проблемы:")
        for issue in issues:
            print(f"   {issue}")
        print(f"\n💡 Рекомендации:")
        print(f"   1. Проверьте, что данные содержат различимую информацию между классами")
        print(f"   2. Увеличьте learning rate или измените архитектуру")
        print(f"   3. Проверьте, что loss функция работает правильно")
        print(f"   4. Попробуйте инициализацию с большим разбросом")
    else:
        print(f"\n✅ Проблем с коллапсом не обнаружено")
        print(f"   Возможно, проблема в другом (например, в данных или архитектуре)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_samples', type=int, default=100)
    
    args = parser.parse_args()
    
    diagnose_collapse(
        checkpoint_path=args.checkpoint_path,
        data_root=args.data_root,
        n_classes=args.n_classes,
        device_str=args.device,
        n_samples=args.n_samples
    )

