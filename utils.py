"""
Утилиты: фиксация сидов, сохранение/загрузка чекпоинтов, логирование
"""

import torch
import numpy as np
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional
import os


def set_seed(seed: int = 42):
    """Фиксация всех сидов для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    save_path: Path,
    is_best: bool = False
):
    """Сохранение чекпоинта"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Сохраняем последний чекпоинт
    torch.save(checkpoint, save_path / 'last.pt')
    
    # Сохраняем лучший чекпоинт
    if is_best:
        torch.save(checkpoint, save_path / 'best.pt')


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None
) -> Dict[str, Any]:
    """Загрузка чекпоинта"""
    # PyTorch 2.6+ требует weights_only=False для чекпоинтов с метриками
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'metrics': checkpoint.get('metrics', {})
    }


def save_config(config: Dict[str, Any], save_path: Path):
    """Сохранение конфигурации в JSON"""
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)


def get_device(devices: str = '1') -> torch.device:
    """Определение устройства для обучения"""
    if devices.startswith('cuda') or (devices.isdigit() and torch.cuda.is_available()):
        if devices.isdigit():
            device_id = int(devices)
            if device_id < torch.cuda.device_count():
                return torch.device(f'cuda:{device_id}')
            else:
                return torch.device('cuda:0')
        else:
            return torch.device(devices)
    elif devices == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """Подсчет обучаемых параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

