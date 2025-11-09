#!/usr/bin/env python3
"""
Тестовый скрипт для проверки EEG-CLIP на небольшом подмножестве данных
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Проверка зависимостей
try:
    import torch
    import torchvision
    print(f"✅ PyTorch {torch.__version__} установлен")
except ImportError as e:
    print(f"❌ PyTorch не установлен: {e}")
    print("   Установите: pip install torch torchvision")
    sys.exit(1)

try:
    import open_clip
    print("✅ open_clip установлен")
except ImportError:
    print("⚠️  open_clip не установлен. Будет использован torchvision ViT")
    print("   Для установки: pip install open-clip-torch")

from eegclip.models import EEGCLIPModel
from eegclip.data import ThingsEEGDataset, create_subject_splits, collate_fn
from eegclip.utils import set_seed, get_device


def test_data_loading(n_classes=10):
    """Тест загрузки данных"""
    print("="*70)
    print("🧪 ТЕСТ 1: Загрузка данных")
    print("="*70)
    
    try:
        dataset = ThingsEEGDataset(
            data_root="data",
            n_classes=n_classes,
            split='train',
            subject_splits=None,  # Используем все субъекты для теста
            eeg_len=100,
            fs=500.0,
            augment=False
        )
        
        print(f"✅ Датасет создан: {len(dataset)} образцов")
        
        # Проверяем один образец
        sample = dataset[0]
        print(f"✅ Образец загружен:")
        print(f"   EEG shape: {sample['eeg'].shape}")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Record ID: {sample['record_id']}")
        print(f"   Subject ID: {sample['subject_id']}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Тест создания модели"""
    print("\n" + "="*70)
    print("🧪 ТЕСТ 2: Создание модели")
    print("="*70)
    
    try:
        device = get_device('cpu')  # Используем CPU для теста
        print(f"🔧 Устройство: {device}")
        
        # Пробуем сначала open_clip
        try:
            model = EEGCLIPModel(
                n_channels=17,
                n_timepoints=100,
                vision_encoder='openclip_vit_b32',
                freeze_vision=True
            ).to(device)
            print("✅ Модель создана (OpenCLIP ViT-B/32)")
        except Exception as e:
            print(f"⚠️  OpenCLIP не доступен: {e}")
            print("   Пробуем torchvision ViT...")
            model = EEGCLIPModel(
                n_channels=17,
                n_timepoints=100,
                vision_encoder='torchvision_vit_b32',
                freeze_vision=True
            ).to(device)
            print("✅ Модель создана (torchvision ViT-B/32)")
        
        # Тест forward pass
        batch_size = 2
        dummy_eeg = torch.randn(batch_size, 17, 100).to(device)
        dummy_img = torch.randn(batch_size, 3, 224, 224).to(device)
        
        with torch.no_grad():
            eeg_emb, img_emb = model(dummy_eeg, dummy_img)
        
        print(f"✅ Forward pass успешен:")
        print(f"   EEG embedding: {eeg_emb.shape}")
        print(f"   Image embedding: {img_emb.shape}")
        print(f"   Logit scale: {model.get_logit_scale().item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка создания модели: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(n_classes=10, batch_size=4):
    """Тест одного шага обучения"""
    print("\n" + "="*70)
    print("🧪 ТЕСТ 3: Один шаг обучения")
    print("="*70)
    
    try:
        device = get_device('cpu')
        set_seed(42)
        
        # Датасет
        dataset = ThingsEEGDataset(
            data_root="data",
            n_classes=n_classes,
            split='train',
            subject_splits=None,
            eeg_len=100,
            fs=500.0,
            augment=False
        )
        
        if len(dataset) < batch_size:
            print(f"⚠️  Недостаточно данных: {len(dataset)} < {batch_size}")
            return False
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # 0 для отладки
            collate_fn=collate_fn
        )
        
        # Модель
        try:
            model = EEGCLIPModel(
                n_channels=17,
                n_timepoints=100,
                vision_encoder='openclip_vit_b32',
                freeze_vision=True
            ).to(device)
        except:
            model = EEGCLIPModel(
                n_channels=17,
                n_timepoints=100,
                vision_encoder='torchvision_vit_b32',
                freeze_vision=True
            ).to(device)
        
        # Loss
        from eegclip.losses import InfoNCELoss
        loss_fn = InfoNCELoss()
        
        # Оптимизатор
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Один батч
        batch = next(iter(dataloader))
        eeg = batch['eeg'].to(device)
        image = batch['image'].to(device)
        
        # Forward
        eeg_emb, img_emb = model(eeg, image)
        logit_scale = model.get_logit_scale()
        loss = loss_fn(eeg_emb, img_emb, logit_scale)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✅ Шаг обучения успешен:")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Logit scale: {logit_scale.item():.4f}")
        print(f"   EEG emb norm: {eeg_emb.norm(dim=1).mean().item():.4f}")
        print(f"   Image emb norm: {img_emb.norm(dim=1).mean().item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Тест EEG-CLIP на небольшом подмножестве')
    parser.add_argument('--n_classes', type=int, default=10, help='Количество классов для теста')
    parser.add_argument('--batch_size', type=int, default=4, help='Размер батча')
    args = parser.parse_args()
    
    print("🚀 ТЕСТИРОВАНИЕ EEG-CLIP")
    print("="*70)
    
    results = []
    
    # Тест 1: Загрузка данных
    results.append(("Загрузка данных", test_data_loading(args.n_classes)))
    
    # Тест 2: Создание модели
    results.append(("Создание модели", test_model_creation()))
    
    # Тест 3: Обучение
    results.append(("Шаг обучения", test_training_step(args.n_classes, args.batch_size)))
    
    # Итоги
    print("\n" + "="*70)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*70)
    
    for test_name, success in results:
        status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
        print(f"   {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✅ Все тесты пройдены! Можно запускать полное обучение.")
    else:
        print("\n❌ Некоторые тесты провалены. Проверьте ошибки выше.")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

