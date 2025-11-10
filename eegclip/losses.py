"""
Loss функции для контрастивного обучения
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (симметричная кросс-энтропия)
    
    Для батча из N пар (eeg_i, img_i):
    - Создаем матрицу сходства S = τ * (E_eeg @ E_img^T)
    - Цели: диагональ (i, i) для правильных пар
    - Loss = (CE(row-wise) + CE(column-wise)) / 2
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, eeg_emb: torch.Tensor, img_emb: torch.Tensor, 
                logit_scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_emb: [B, D] - L2-нормированные эмбеддинги EEG
            img_emb: [B, D] - L2-нормированные эмбеддинги изображений
            logit_scale: скаляр - температура (exp(logit_scale))
        Returns:
            loss: скаляр
        """
        # Матрица сходства: [B, B]
        logits = logit_scale * (eeg_emb @ img_emb.T)
        
        # Цели: диагональ (правильные пары)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # Симметричная кросс-энтропия
        loss_eeg2img = F.cross_entropy(logits, labels)
        loss_img2eeg = F.cross_entropy(logits.T, labels)
        
        loss = (loss_eeg2img + loss_img2eeg) / 2.0
        
        return loss


class AuxiliaryClassificationLoss(nn.Module):
    """
    Вспомогательная классификационная потеря (опционально)
    """
    
    def __init__(self, num_classes: int, hidden_dim: int = 512, weight: float = 0.2):
        super().__init__()
        self.weight = weight
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, eeg_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_emb: [B, hidden_dim] - эмбеддинги EEG
            labels: [B] - метки классов
        Returns:
            loss: скаляр
        """
        logits = self.classifier(eeg_emb)
        loss = self.criterion(logits, labels)
        return self.weight * loss

