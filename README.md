# EEG-CLIP: Contrastive Learning for EEG-Image Alignment

Repository for storing source code for netology diploma project for EEG-Image classification model.

Контрастивное обучение для сопоставления эмбеддингов EEG с эмбеддингами изображений по методу CLIP/InfoNCE.

## Установка

```bash
pip install -r requirements.txt
```

## Структура данных

Проект адаптирован для формата Things-EEG:

```
data/
  eeg/
    sub-01/
      preprocessed_eeg_training.npy  # (n_trials, n_repetitions, n_channels, n_timepoints)
    sub-02/
      ...
  images/
    training_images/
      00001_aardvark/
        *.jpg
      00002_abacus/
        *.jpg
      ...
```

## Обучение

```bash
python train_eegclip.py \
  --data_root data \
  --epochs 50 \
  --batch_size 128 \
  --num_workers 4 \
  --eeg_len 100 \
  --fs 500.0 \
  --bandpass 0.5 45.0 \
  --notch 50 \
  --augment_eeg \
  --augment_eeg \
  --vision_encoder openclip_vit_b32 \
  --freeze_vision \
  --eeg_d_model 256 \
  --eeg_layers 4 \
  --proj_dim 512 \
  --lr 3e-4 \
  --wd 0.05 \
  --devices 1 \
  --precision amp \
  --save_dir ./checkpoints \
  --log_dir ./logs \
  --seed 42
```

### Основные параметры

**Данные:**
- `--data_root`: корневая директория с `eeg/` и `images/`
- `--n_classes`: количество классов (None = все)
- `--eeg_len`: фиксированная длина EEG в сэмплах (100)
- `--fs`: частота дискретизации (500 Гц)
- `--bandpass`: полосовой фильтр [low, high] (0.5, 45.0)
- `--notch`: частота notch-фильтра (50 или 60)

**Аугментации:**
- `--augment_eeg`: включить аугментации EEG
- `--noise_std`: стандартное отклонение шума (0.01)
- `--jitter_ms`: временной сдвиг в мс (20.0)
- `--time_mask_prob`: вероятность временной маски (0.2)
- `--channel_drop_prob`: вероятность dropout канала (0.1)

**Модель:**
- `--vision_encoder`: тип визуального энкодера (`openclip_vit_b32` или `torchvision_vit_b32`)
- `--freeze_vision`: заморозить визуальный энкодер (по умолчанию True)
- `--eeg_d_model`: размерность модели EEG encoder (256)
- `--eeg_layers`: количество transformer слоев (4)
- `--eeg_hidden`: скрытая размерность EEG encoder (512)
- `--proj_dim`: размерность совместного пространства (512)
- `--temporal_pool`: метод временного пулинга (`cls`, `mean`, `max`)

**Обучение:**
- `--epochs`: количество эпох (50)
- `--batch_size`: размер батча (128)
- `--lr`: learning rate (3e-4)
- `--wd`: weight decay (0.05)
- `--warmup_ratio`: доля шагов для warmup (0.05)
- `--precision`: `amp` (mixed precision) или `fp32`

**Система:**
- `--devices`: устройство (`1`, `cuda:0`, `mps`, `cpu`)
- `--save_dir`: директория для чекпоинтов
- `--log_dir`: директория для TensorBoard логов
- `--resume`: путь к чекпоинту для возобновления

## Экспорт эмбеддингов

```bash
python export_embeddings.py \
  --data_root data \
  --ckpt ./checkpoints/best.pt \
  --split test \
  --out ./embeddings_test.npz
```

## Оценка retrieval

```bash
python eval_retrieval.py \
  --embeds ./embeddings_test.npz \
  --metric recall@1,5,10 \
  --also mrr,ndcg \
  --out ./eval_report.json
```

## Архитектура

### EEG Encoder
1. **1D CNN блоки** (3 слоя с kernel sizes 7, 5, 3 и dilation)
2. **Проекция к d_model** (256)
3. **Transformer Encoder** (4 слоя, 8 heads)
4. **Temporal pooling** (CLS токен или mean/max)
5. **Проекционная голова** MLP (512 → 1024 → 512)

### Vision Encoder
- **OpenCLIP ViT-B/32** (замороженный по умолчанию)
- Проекционная голова MLP для выравнивания размерности

### Loss
- **InfoNCE Loss** (симметричная кросс-энтропия)
- Обучаемая температура (logit_scale)

## Метрики

- **Recall@K** (K=1,5,10) для обоих направлений: EEG→Image и Image→EEG
- **Mean Rank**
- **MRR** (Mean Reciprocal Rank)
- **nDCG@K**

## Subject-wise Split

Автоматическое разбиение по субъектам:
- Train: 80%
- Val: 10%
- Test: 10%

Это предотвращает data leakage между субъектами.

## TensorBoard

```bash
tensorboard --logdir logs
```

## Примеры результатов

После обучения модель должна достичь:
- **Recall@1**: > 0.1 (для 1654 классов)
- **Recall@5**: > 0.3
- **Recall@10**: > 0.5

## Примечания

- Модель обучается только на EEG encoder и проекционных головах
- Визуальный энкодер заморожен по умолчанию
- Используется mixed precision (AMP) для ускорения
- Subject-wise split предотвращает data leakage
- Поддержка возобновления обучения с чекпоинта
