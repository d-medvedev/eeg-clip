#!/usr/bin/env python3
"""
Улучшения для обучения: увеличение learning rate, лучшая инициализация
"""

import torch
import torch.nn as nn
import math


def init_projection_head(module: nn.Module):
    """Улучшенная инициализация проекционной головы"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # Kaiming инициализация для ReLU/GELU активаций
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def get_improved_config():
    """Конфигурация с улучшенными гиперпараметрами"""
    return {
        'lr': 1e-3,  # Увеличен с 3e-4 до 1e-3
        'wd': 0.05,
        'warmup_ratio': 0.1,  # Увеличен warmup
        'grad_clip': 1.0,
        'batch_size': 32,  # Увеличен batch size
        'epochs': 100,
        # Архитектура
        'proj_hidden': 2048,  # Увеличен размер скрытого слоя
        'dropout': 0.1,
    }


def create_improved_projection_head(in_dim: int, hidden_dim: int = 2048, 
                                     out_dim: int = 512, dropout: float = 0.1):
    """Создание улучшенной проекционной головы с лучшей инициализацией"""
    head = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),  # Добавлен LayerNorm
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),  # Дополнительный слой
        nn.LayerNorm(hidden_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, out_dim)
    )
    
    # Инициализация
    init_projection_head(head)
    
    return head

