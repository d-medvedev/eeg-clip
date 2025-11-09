#!/usr/bin/env python3
"""
Оценка retrieval метрик
"""

import argparse
import numpy as np
import json
from pathlib import Path
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate retrieval metrics')
    parser.add_argument('--embeds', type=str, required=True, help='Embeddings .npz file')
    parser.add_argument('--metric', type=str, default='recall@1,5,10', help='Metrics to compute')
    parser.add_argument('--also', type=str, default='mrr,ndcg', help='Additional metrics')
    parser.add_argument('--out', type=str, default='eval_report.json', help='Output JSON file')
    return parser.parse_args()


def compute_retrieval_metrics_numpy(eeg_emb, img_emb, k_list=[1, 5, 10]):
    """Вычисление метрик на numpy массивах"""
    eeg_emb = torch.from_numpy(eeg_emb).float()
    img_emb = torch.from_numpy(img_emb).float()
    
    # Нормализация
    eeg_emb = torch.nn.functional.normalize(eeg_emb, p=2, dim=1)
    img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=1)
    
    # Матрица сходства
    similarity = eeg_emb @ img_emb.T
    N = similarity.shape[0]
    labels = torch.arange(N)
    
    metrics = {}
    
    # EEG → Image
    eeg2img_scores, eeg2img_indices = similarity.topk(N, dim=1)
    eeg2img_ranks = torch.zeros(N)
    for i in range(N):
        rank = (eeg2img_indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        eeg2img_ranks[i] = rank.item() + 1 if len(rank) > 0 else N + 1
    
    for k in k_list:
        recall = (eeg2img_ranks <= k).float().mean().item()
        metrics[f'eeg2img_recall@{k}'] = recall
    
    metrics['eeg2img_mean_rank'] = eeg2img_ranks.float().mean().item()
    metrics['eeg2img_mrr'] = (1.0 / eeg2img_ranks).float().mean().item()
    
    # Image → EEG
    img2eeg_scores, img2eeg_indices = similarity.T.topk(N, dim=1)
    img2eeg_ranks = torch.zeros(N)
    for i in range(N):
        rank = (img2eeg_indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        img2eeg_ranks[i] = rank.item() + 1 if len(rank) > 0 else N + 1
    
    for k in k_list:
        recall = (img2eeg_ranks <= k).float().mean().item()
        metrics[f'img2eeg_recall@{k}'] = recall
    
    metrics['img2eeg_mean_rank'] = img2eeg_ranks.float().mean().item()
    metrics['img2eeg_mrr'] = (1.0 / img2eeg_ranks).float().mean().item()
    
    return metrics


def main():
    args = parse_args()
    
    # Загрузка эмбеддингов
    data = np.load(args.embeds, allow_pickle=True)
    eeg_emb = data['eeg']
    img_emb = data['img']
    
    print(f"📊 Загружено эмбеддингов:")
    print(f"   EEG: {eeg_emb.shape}")
    print(f"   Image: {img_emb.shape}")
    
    # Парсинг метрик
    k_list = []
    if 'recall' in args.metric:
        for k_str in args.metric.split(','):
            if '@' in k_str:
                k = int(k_str.split('@')[1])
                k_list.append(k)
    
    if not k_list:
        k_list = [1, 5, 10]
    
    # Вычисление метрик
    metrics = compute_retrieval_metrics_numpy(eeg_emb, img_emb, k_list=k_list)
    
    # Вывод
    print(f"\n📊 Результаты:")
    print(f"   EEG → Image:")
    for k in k_list:
        print(f"      Recall@{k}: {metrics[f'eeg2img_recall@{k}']:.4f}")
    print(f"      Mean Rank: {metrics['eeg2img_mean_rank']:.2f}")
    print(f"      MRR: {metrics['eeg2img_mrr']:.4f}")
    
    print(f"\n   Image → EEG:")
    for k in k_list:
        print(f"      Recall@{k}: {metrics[f'img2eeg_recall@{k}']:.4f}")
    print(f"      Mean Rank: {metrics['img2eeg_mean_rank']:.2f}")
    print(f"      MRR: {metrics['img2eeg_mrr']:.4f}")
    
    # Сохранение
    with open(args.out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Отчет сохранен: {args.out}")


if __name__ == '__main__':
    main()

