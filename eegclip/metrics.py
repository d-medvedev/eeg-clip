"""
Метрики для оценки retrieval: Recall@K, MRR, nDCG, Mean Rank
"""

import torch
import numpy as np
from typing import Dict, Tuple


def compute_retrieval_metrics(
    eeg_emb: torch.Tensor,
    img_emb: torch.Tensor,
    k_list: list = [1, 5, 10]
) -> Dict[str, float]:
    """
    Вычисление метрик retrieval для обоих направлений: EEG→Image и Image→EEG
    
    Args:
        eeg_emb: [N, D] - эмбеддинги EEG
        img_emb: [N, D] - эмбеддинги изображений
        k_list: список K для Recall@K
    Returns:
        словарь с метриками
    """
    device = eeg_emb.device
    N = eeg_emb.shape[0]
    
    # Матрица сходства: [N, N]
    similarity = eeg_emb @ img_emb.T  # косинусное сходство (уже L2-нормированные)
    
    # Правильные пары - диагональ
    labels = torch.arange(N, device=device)
    
    metrics = {}
    
    # ===== EEG → Image =====
    eeg2img_scores, eeg2img_indices = similarity.topk(N, dim=1)
    eeg2img_ranks = torch.zeros(N, device=device)
    for i in range(N):
        rank = (eeg2img_indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        eeg2img_ranks[i] = rank.item() + 1 if len(rank) > 0 else N + 1
    
    # Recall@K
    for k in k_list:
        recall = (eeg2img_ranks <= k).float().mean().item()
        metrics[f'eeg2img_recall@{k}'] = recall
    
    # Mean Rank
    metrics['eeg2img_mean_rank'] = eeg2img_ranks.float().mean().item()
    
    # MRR (Mean Reciprocal Rank)
    metrics['eeg2img_mrr'] = (1.0 / eeg2img_ranks).float().mean().item()
    
    # nDCG@K
    for k in k_list:
        ndcg = compute_ndcg_at_k(eeg2img_scores, labels, k)
        metrics[f'eeg2img_ndcg@{k}'] = ndcg
    
    # ===== Image → EEG =====
    img2eeg_scores, img2eeg_indices = similarity.T.topk(N, dim=1)
    img2eeg_ranks = torch.zeros(N, device=device)
    for i in range(N):
        rank = (img2eeg_indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        img2eeg_ranks[i] = rank.item() + 1 if len(rank) > 0 else N + 1
    
    # Recall@K
    for k in k_list:
        recall = (img2eeg_ranks <= k).float().mean().item()
        metrics[f'img2eeg_recall@{k}'] = recall
    
    # Mean Rank
    metrics['img2eeg_mean_rank'] = img2eeg_ranks.float().mean().item()
    
    # MRR
    metrics['img2eeg_mrr'] = (1.0 / img2eeg_ranks).float().mean().item()
    
    # nDCG@K
    for k in k_list:
        ndcg = compute_ndcg_at_k(img2eeg_scores, labels, k)
        metrics[f'img2eeg_ndcg@{k}'] = ndcg
    
    return metrics


def compute_ndcg_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Вычисление nDCG@K
    
    Args:
        scores: [N, N] - матрица сходства (уже отсортированная по убыванию)
        labels: [N] - правильные индексы
        k: K для nDCG@K
    Returns:
        nDCG@K: float
    """
    N = scores.shape[0]
    device = scores.device
    
    # DCG@K
    dcg = 0.0
    for i in range(N):
        relevant_idx = labels[i].item()
        # Находим позицию правильного ответа в топ-K
        topk_scores = scores[i, :k]
        # Проверяем, есть ли правильный ответ в топ-K
        if relevant_idx < k:
            # Находим позицию в топ-K (она равна relevant_idx, т.к. scores уже отсортированы)
            rank_in_topk = relevant_idx
            dcg += 1.0 / np.log2(rank_in_topk + 2)  # +2 потому что rank начинается с 1
    
    # IDCG@K (идеальный случай - правильный ответ всегда на первом месте)
    idcg = sum(1.0 / np.log2(j + 2) for j in range(min(k, N)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_top1_accuracy(eeg_emb: torch.Tensor, img_emb: torch.Tensor) -> float:
    """
    Top-1 точность (эквивалентна Recall@1)
    """
    similarity = eeg_emb @ img_emb.T
    labels = torch.arange(eeg_emb.shape[0], device=eeg_emb.device)
    pred = similarity.argmax(dim=1)
    accuracy = (pred == labels).float().mean().item()
    return accuracy

