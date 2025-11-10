"""
Даталоадеры и предобработка для EEG-CLIP
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import Optional, Tuple, List, Dict
from scipy import signal
import random
from torchvision import transforms


class EEGPreprocessor:
    """Предобработка EEG сигналов"""
    
    def __init__(
        self,
        fs: float = 500.0,
        bandpass: Optional[Tuple[float, float]] = (0.5, 45.0),
        notch: Optional[float] = None,
        zscore: bool = True
    ):
        self.fs = fs
        self.bandpass = bandpass
        self.notch = notch
        self.zscore = zscore
        
        # Создаем фильтры
        self.bandpass_filter = None
        self.notch_filter = None
        
        if bandpass:
            nyquist = fs / 2.0
            low, high = bandpass
            if high >= nyquist:
                high = nyquist * 0.95
            self.bandpass_filter = signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
        
        if notch:
            # Notch filter для 50/60 Hz
            f0 = notch
            Q = 30.0
            self.notch_filter = signal.iirnotch(f0, Q, fs=fs)
    
    def __call__(self, eeg: np.ndarray) -> np.ndarray:
        """
        Применить предобработку
        
        Args:
            eeg: [C, T] - EEG сигнал
        Returns:
            [C, T] - обработанный сигнал
        """
        eeg = eeg.copy()
        
        # Применяем фильтры по каналам
        for ch in range(eeg.shape[0]):
            signal_ch = eeg[ch, :]
            
            # Notch filter
            if self.notch_filter is not None:
                signal_ch = signal.sosfiltfilt(self.notch_filter[0], signal_ch)
            
            # Bandpass filter
            if self.bandpass_filter is not None:
                signal_ch = signal.sosfiltfilt(self.bandpass_filter, signal_ch)
            
            eeg[ch, :] = signal_ch
        
        # Z-score нормализация по каналам
        if self.zscore:
            mean = eeg.mean(axis=1, keepdims=True)
            std = eeg.std(axis=1, keepdims=True) + 1e-8
            eeg = (eeg - mean) / std
        
        return eeg


class EEGAugmentation:
    """Аугментации для EEG"""
    
    def __init__(
        self,
        noise_std: float = 0.01,
        jitter_ms: float = 20.0,
        fs: float = 500.0,
        time_mask_prob: float = 0.2,
        time_mask_len: int = 10,
        channel_drop_prob: float = 0.1
    ):
        self.noise_std = noise_std
        self.jitter_samples = int(jitter_ms * fs / 1000.0)
        self.fs = fs
        self.time_mask_prob = time_mask_prob
        self.time_mask_len = time_mask_len
        self.channel_drop_prob = channel_drop_prob
    
    def __call__(self, eeg: np.ndarray) -> np.ndarray:
        """
        Применить аугментации
        
        Args:
            eeg: [C, T] - EEG сигнал
        Returns:
            [C, T] - аугментированный сигнал
        """
        eeg = eeg.copy()
        C, T = eeg.shape
        
        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, eeg.shape)
            eeg = eeg + noise
        
        # Time jitter (циклический сдвиг)
        if self.jitter_samples > 0:
            shift = random.randint(-self.jitter_samples, self.jitter_samples)
            if shift != 0:
                eeg = np.roll(eeg, shift, axis=1)
        
        # Time mask (SpecAugment-like)
        if random.random() < self.time_mask_prob:
            mask_start = random.randint(0, max(1, T - self.time_mask_len))
            mask_end = min(mask_start + self.time_mask_len, T)
            eeg[:, mask_start:mask_end] = 0
        
        # Channel dropout
        if random.random() < self.channel_drop_prob:
            ch_to_drop = random.randint(0, C - 1)
            eeg[ch_to_drop, :] = 0
        
        return eeg


class ThingsEEGDataset(Dataset):
    """
    Датасет для Things-EEG формата
    
    Структура:
    - EEG: data/eeg/sub-XX/preprocessed_eeg_training.npy
      Форма: (n_trials, n_repetitions, n_channels, n_timepoints)
    - Images: data/images/training_images/XXXXX_classname/*.jpg
    """
    
    def __init__(
        self,
        data_root: str = "data",
        subjects: Optional[List[int]] = None,
        n_classes: Optional[int] = None,
        split: str = 'train',
        subject_splits: Optional[Dict[str, List[int]]] = None,
        eeg_len: Optional[int] = None,
        fs: float = 500.0,
        preprocess_eeg: bool = False,  # По умолчанию отключено, т.к. данные уже предобработаны
        bandpass: Optional[Tuple[float, float]] = (0.5, 45.0),
        notch: Optional[float] = None,
        augment: bool = False,
        noise_std: float = 0.01,
        jitter_ms: float = 20.0,
        time_mask_prob: float = 0.2,
        channel_drop_prob: float = 0.1,
        use_features: bool = False  # Использовать извлеченные метрики вместо сырых данных
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.eeg_len = eeg_len
        self.augment = augment
        self.preprocess_eeg = preprocess_eeg
        self.use_features = use_features
        self.fs = fs
        
        # Предобработка (только если включена)
        self.eeg_preprocessor = EEGPreprocessor(fs=fs, bandpass=bandpass, notch=notch) if preprocess_eeg else None
        self.eeg_augmenter = EEGAugmentation(
            noise_std=noise_std,
            jitter_ms=jitter_ms,
            fs=fs,
            time_mask_prob=time_mask_prob,
            channel_drop_prob=channel_drop_prob
        ) if augment else None
        
        # Трансформации изображений
        # Используем нормализацию OpenCLIP (подходит и для torchvision)
        # OpenCLIP: mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        if split == 'train':
            self.image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                   std=[0.26862954, 0.26130258, 0.27577711])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                   std=[0.26862954, 0.26130258, 0.27577711])
            ])
        
        # Загружаем метаданные
        self.samples = self._load_samples(subjects, n_classes, subject_splits)
        
        print(f"✅ Загружено {len(self.samples)} образцов для split='{split}'")
    
    def _load_samples(
        self,
        subjects: Optional[List[int]],
        n_classes: Optional[int],
        subject_splits: Optional[Dict[str, List[int]]]
    ) -> List[Dict]:
        """Загрузка списка образцов"""
        samples = []
        
        eeg_dir = self.data_root / "eeg"
        images_dir = self.data_root / "images" / "training_images"
        
        # Получаем список классов из директорий изображений
        class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        if n_classes:
            class_dirs = class_dirs[:n_classes]
        
        # Определяем какие субъекты использовать
        if subjects is None:
            subjects = list(range(1, 11))
        
        # Фильтруем субъектов по split
        if subject_splits:
            subjects = [s for s in subjects if s in subject_splits.get(self.split, [])]
        
        # Загружаем данные от каждого субъекта
        for subject_id in subjects:
            subject_str = f"sub-{subject_id:02d}"
            eeg_file = eeg_dir / subject_str / "preprocessed_eeg_training.npy"
            
            if not eeg_file.exists():
                continue
            
            # Загружаем EEG данные
            data_dict = np.load(eeg_file, allow_pickle=True).item()
            eeg_data = data_dict['preprocessed_eeg_data']  # (n_trials, n_repetitions, n_channels, n_timepoints)
            
            n_trials = min(eeg_data.shape[0], len(class_dirs))
            
            # Для каждого класса
            for class_idx in range(n_trials):
                class_dir = class_dirs[class_idx]
                class_name = class_dir.name
                
                # Получаем изображения для этого класса
                image_files = sorted(list(class_dir.glob("*.jpg")))
                if not image_files:
                    continue
                
                # Для каждого повторения EEG создаем пары со ВСЕМИ изображениями класса
                # Это увеличивает данные в len(image_files) раз!
                n_repetitions = eeg_data.shape[1]
                for rep_idx in range(n_repetitions):
                    for img_idx, image_file in enumerate(image_files):
                        # Создаем образец для каждой комбинации (повторение, изображение)
                        samples.append({
                            'eeg_data': eeg_data[class_idx, rep_idx, :, :],  # [C, T]
                            'image_path': image_file,
                            'record_id': f"{subject_str}_class{class_idx}_rep{rep_idx}_img{img_idx}",
                            'image_id': image_file.stem,
                            'subject_id': subject_id,
                            'class_idx': class_idx,
                            'class_name': class_name
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # EEG
        eeg_raw = sample['eeg_data'].astype(np.float32)  # [C, T]
        
        if self.use_features:
            # Извлекаем метрики из ЭЭГ
            # Импорт здесь, чтобы избежать циклических зависимостей
            try:
                from .features import extract_eeg_features
            except ImportError:
                from eegclip.features import extract_eeg_features
            eeg = extract_eeg_features(eeg_raw, fs=self.fs)  # [n_features]
            eeg = torch.from_numpy(eeg).float()
        else:
            # Используем сырые данные
            eeg = eeg_raw
            
            # Предобработка (только если включена, данные уже предобработаны)
            if self.preprocess_eeg and self.eeg_preprocessor is not None:
                eeg = self.eeg_preprocessor(eeg)
            
            # Обрезка до фиксированной длины (если нужно)
            # НЕ используем паддинг, чтобы не терять информацию и не добавлять нули
            if self.eeg_len:
                C, T = eeg.shape
                if T > self.eeg_len:
                    # Центрированная обрезка (если данные длиннее требуемого)
                    start = (T - self.eeg_len) // 2
                    eeg = eeg[:, start:start + self.eeg_len]
                # Если T < eeg_len, оставляем данные как есть (не паддим нулями)
            
            # Аугментация (только на train)
            if self.augment and self.eeg_augmenter:
                eeg = self.eeg_augmenter(eeg)
            
            eeg = torch.from_numpy(eeg).float()
        
        # Изображение
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.image_transform(image)
        
        return {
            'eeg': eeg,
            'image': image,
            'record_id': sample['record_id'],
            'image_id': sample['image_id'],
            'subject_id': sample['subject_id'],
            'class_idx': sample['class_idx']
        }


def create_subject_splits(
    subjects: List[int],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    Создание subject-wise разбиений
    
    Args:
        subjects: список ID субъектов
        val_ratio: доля для validation
        test_ratio: доля для test
        seed: random seed
    Returns:
        словарь {'train': [...], 'val': [...], 'test': [...]}
    """
    random.seed(seed)
    subjects_shuffled = subjects.copy()
    random.shuffle(subjects_shuffled)
    
    n_total = len(subjects_shuffled)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_test - n_val
    
    splits = {
        'train': subjects_shuffled[:n_train],
        'val': subjects_shuffled[n_train:n_train + n_val],
        'test': subjects_shuffled[n_train + n_val:]
    }
    
    return splits


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Коллятор для батча"""
    eegs = torch.stack([item['eeg'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    record_ids = [item['record_id'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    subject_ids = torch.tensor([item['subject_id'] for item in batch])
    class_indices = torch.tensor([item['class_idx'] for item in batch])
    
    return {
        'eeg': eegs,
        'image': images,
        'record_id': record_ids,
        'image_id': image_ids,
        'subject_id': subject_ids,
        'class_idx': class_indices
    }

