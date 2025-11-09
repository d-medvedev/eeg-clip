"""
Модели для EEG-CLIP: EEG Encoder, Vision Encoder, Projection Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding1D(nn.Module):
    """Синусоидальные позиционные эмбеддинги для временных последовательностей"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNN1DBlock(nn.Module):
    """1D CNN блок с residual connection"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=(kernel_size - 1) * dilation // 2, 
                             dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (in_channels == out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            [B, C, T]
        """
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        if self.residual:
            out = out + residual
        return out


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder блок (MHSA + FFN)"""
    
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(attn_out)
        
        # FFN
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        
        return x


class EEGEncoder(nn.Module):
    """
    EEG Encoder: CNN + Transformer
    
    Архитектура:
    1. 1D CNN блоки для извлечения временных признаков
    2. Проекция к d_model
    3. Transformer Encoder для временной модели
    4. Temporal pooling (CLS или mean)
    5. Проекционная голова к совместному пространству
    """
    
    def __init__(
        self,
        n_channels: int = 17,
        n_timepoints: int = 100,
        d_model: int = 256,
        n_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        d_eeg: int = 512,
        proj_dim: int = 512,
        dropout: float = 0.1,
        temporal_pool: str = 'cls'  # 'cls', 'mean', 'max'
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_pool = temporal_pool
        
        # 1D CNN блоки
        self.cnn_blocks = nn.Sequential(
            CNN1DBlock(n_channels, 64, kernel_size=7, dilation=1, dropout=dropout),
            CNN1DBlock(64, 128, kernel_size=5, dilation=2, dropout=dropout),
            CNN1DBlock(128, 128, kernel_size=3, dilation=4, dropout=dropout),
        )
        
        # Проекция к d_model
        self.proj_to_dmodel = nn.Linear(128, d_model)
        
        # CLS токен (если используется)
        if temporal_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            # Позиционные эмбеддинги должны учитывать CLS токен
            pos_max_len = n_timepoints + 1
        else:
            pos_max_len = n_timepoints
        
        # Позиционные эмбеддинги
        self.pos_encoder = PositionalEncoding1D(d_model, max_len=pos_max_len, dropout=dropout)
        
        # Transformer Encoder блоки
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        
        # Финальная проекция
        self.final_proj = nn.Sequential(
            nn.Linear(d_model, d_eeg),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_eeg, proj_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] - EEG сигнал
        Returns:
            [B, proj_dim] - L2-нормированный эмбеддинг
        """
        B, C, T = x.shape
        
        # CNN блоки
        x = self.cnn_blocks(x)  # [B, 128, T]
        
        # Перестановка для линейного слоя: [B, T, 128]
        x = x.transpose(1, 2)
        x = self.proj_to_dmodel(x)  # [B, T, d_model]
        
        # Добавляем CLS токен если нужно
        if self.temporal_pool == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, d_model]
        
        # Позиционные эмбеддинги
        x = self.pos_encoder(x)
        
        # Transformer блоки
        for block in self.transformer_blocks:
            x = block(x)
        
        # Temporal pooling
        if self.temporal_pool == 'cls':
            x = x[:, 0, :]  # [B, d_model] - CLS токен
        elif self.temporal_pool == 'mean':
            x = x.mean(dim=1)  # [B, d_model]
        elif self.temporal_pool == 'max':
            x = x.max(dim=1)[0]  # [B, d_model]
        else:
            raise ValueError(f"Unknown temporal_pool: {temporal_pool}")
        
        # Финальная проекция
        x = self.final_proj(x)  # [B, proj_dim]
        
        # L2 нормализация
        x = F.normalize(x, p=2, dim=1)
        
        return x


class VisionEncoderWrapper(nn.Module):
    """
    Обертка для визуального энкодера (OpenCLIP ViT или torchvision ViT)
    """
    
    def __init__(self, model_name: str = 'openclip_vit_b32', freeze: bool = True):
        super().__init__()
        self.freeze = freeze
        
        if model_name.startswith('openclip'):
            try:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='openai'
                )
                self.model = model.visual
            except ImportError:
                raise ImportError("open_clip_torch не установлен. Установите: pip install open-clip-torch")
        elif model_name.startswith('torchvision'):
            from torchvision.models import vit_b_32, ViT_B_32_Weights
            weights = ViT_B_32_Weights.IMAGENET1K_V1
            model = vit_b_32(weights=weights)
            self.model = model
        else:
            raise ValueError(f"Unknown vision encoder: {model_name}")
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] - изображение (уже нормализованное)
        Returns:
            [B, d_vision] - эмбеддинг изображения
        """
        if self.freeze:
            self.model.eval()
            with torch.no_grad():
                x = self._extract_features(x)
        else:
            x = self._extract_features(x)
        
        return x
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Извлечение признаков из визуального энкодера"""
        # Проверяем тип модели
        if hasattr(self.model, 'forward_features'):
            # OpenCLIP
            x = self.model.forward_features(x)
            if isinstance(x, tuple):
                x = x[0]
            # Берем CLS токен или mean pool
            if len(x.shape) == 3:  # [B, N, D]
                x = x[:, 0, :] if x.shape[1] > 1 else x.mean(dim=1)
        elif hasattr(self.model, '_process_input'):
            # torchvision ViT
            x = self.model._process_input(x)
            n = x.shape[0]
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.model.encoder(x)
            x = x[:, 0]  # CLS токен
        else:
            # Fallback: просто forward
            x = self.model(x)
            if len(x.shape) > 2:
                x = x.mean(dim=1) if x.shape[1] > 1 else x[:, 0, :]
        
        return x


class ProjectionHead(nn.Module):
    """Проекционная голова MLP"""
    
    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim]
        Returns:
            [B, out_dim] - L2-нормированный эмбеддинг
        """
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class EEGCLIPModel(nn.Module):
    """
    Полная модель EEG-CLIP
    """
    
    def __init__(
        self,
        # EEG encoder params
        n_channels: int = 17,
        n_timepoints: int = 100,
        eeg_d_model: int = 256,
        eeg_layers: int = 4,
        eeg_hidden: int = 512,
        # Vision encoder params
        vision_encoder: str = 'openclip_vit_b32',
        freeze_vision: bool = True,
        # Projection params
        proj_dim: int = 512,
        proj_hidden: int = 1024,
        # Other
        dropout: float = 0.1,
        temporal_pool: str = 'cls',
        learnable_temp: bool = True,
        init_temp: float = 0.07
    ):
        super().__init__()
        
        # EEG Encoder
        self.eeg_encoder = EEGEncoder(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            d_model=eeg_d_model,
            n_layers=eeg_layers,
            d_eeg=eeg_hidden,
            proj_dim=proj_dim,
            dropout=dropout,
            temporal_pool=temporal_pool
        )
        
        # Vision Encoder
        self.vision_encoder = VisionEncoderWrapper(vision_encoder, freeze=freeze_vision)
        
        # Получаем размерность визуального энкодера
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224)
            dummy_emb = self.vision_encoder(dummy_img)
            d_vision = dummy_emb.shape[1]
        
        # Projection heads
        self.eeg_proj = ProjectionHead(proj_dim, proj_hidden, proj_dim, dropout)
        self.vision_proj = ProjectionHead(d_vision, proj_hidden, proj_dim, dropout)
        
        # Температура (learnable logit scale)
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / init_temp))
        else:
            self.register_buffer('logit_scale', torch.tensor(math.log(1 / init_temp)))
    
    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """Кодирование EEG"""
        eeg_emb = self.eeg_encoder(eeg)
        eeg_emb = self.eeg_proj(eeg_emb)
        return eeg_emb
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Кодирование изображения"""
        img_emb = self.vision_encoder(image)
        img_emb = self.vision_proj(img_emb)
        return img_emb
    
    def forward(self, eeg: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            eeg: [B, C, T]
            image: [B, 3, H, W]
        Returns:
            eeg_emb: [B, proj_dim]
            img_emb: [B, proj_dim]
        """
        eeg_emb = self.encode_eeg(eeg)
        img_emb = self.encode_image(image)
        return eeg_emb, img_emb
    
    def get_logit_scale(self) -> torch.Tensor:
        """Получить температуру с клиппингом"""
        # Клиппим параметр перед exp, чтобы температура не превышала 100
        # Это эквивалентно температуре = min(exp(logit_scale), 100)
        return self.logit_scale.clamp(max=math.log(100.0)).exp()
    
    def get_logit_scale_param(self) -> torch.Tensor:
        """Получить сам параметр logit_scale (для логирования)"""
        return self.logit_scale

