import numpy as np
from typing import List, Dict, Set


def precision_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """Calculates Precision@K for a single user."""
    if not actual or k == 0:
        return 0.0
    pred_k = predicted[:k]
    actual_set = set(actual)
    hits = sum(1 for item in pred_k if item in actual_set)
    return hits / float(k)


def recall_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """Calculates Recall@K for a single user."""
    if not actual or k == 0:
        return 0.0
    pred_k = predicted[:k]
    actual_set = set(actual)
    hits = sum(1 for item in pred_k if item in actual_set)
    return hits / float(len(actual))


def ndcg_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """Calculates Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
    if not actual or k == 0:
        return 0.0

    actual_set = set(actual)
    pred_k = predicted[:k]

    # Calculate DCG@K
    dcg = 0.0
    for rank, item in enumerate(pred_k, start=1):
        if item in actual_set:
            dcg += 1.0 / np.log2(rank + 1)

    # Calculate IDCG@K (Ideal DCG)
    idcg_len = min(len(actual), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, idcg_len + 1))

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def map_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """Calculates Mean Average Precision (MAP@K) for a single user."""
    if not actual or k == 0:
        return 0.0

    actual_set = set(actual)
    pred_k = predicted[:k]

    hits = 0
    score = 0.0
    for rank, item in enumerate(pred_k, start=1):
        if item in actual_set:
            hits += 1
            score += hits / float(rank)

    denom = min(len(actual), k)
    if denom == 0:
        return 0.0
    return float(score / denom)


def evaluate_model_ranking(
    model,
    train_matrix,
    test_ground_truth: Dict[int, List[int]],
    k: int = 10,
    model_type: str = 'cf'
) -> Dict[str, float]:
    """
    Evaluates a model across a test dictionary (user_idx -> actual items).
    Returns mean Precision@K, Recall@K, NDCG@K, MAP@K, and Catalog Coverage.
    """
    precisions = []
    recalls = []
    ndcgs = []
    maps = []
    recommended_unique_items: Set[int] = set()

    for user_idx, actual_items in test_ground_truth.items():
        if not actual_items:
            continue

        # Generate recommendations based on model_type signature
        if model_type == 'pop':
            recs = model.recommend(user_idx=user_idx, n=k, filter_items=set(train_matrix[user_idx].indices))
        elif model_type == 'cb':
            recs = model.recommend(user_idx=user_idx, train_matrix=train_matrix, n=k)
        elif model_type == 'cf':
            recs = model.recommend(user_idx=user_idx, n=k)
        elif model_type == 'hybrid':
            recs = model.recommend(user_idx=user_idx, train_matrix=train_matrix, n=k)
        else:
            recs = []

        pred_items = [item_idx for item_idx, _ in recs]
        recommended_unique_items.update(pred_items)

        precisions.append(precision_at_k(actual_items, pred_items, k=k))
        recalls.append(recall_at_k(actual_items, pred_items, k=k))
        ndcgs.append(ndcg_at_k(actual_items, pred_items, k=k))
        maps.append(map_at_k(actual_items, pred_items, k=k))

    total_catalog_size = train_matrix.shape[1]
    coverage = len(recommended_unique_items) / float(total_catalog_size) if total_catalog_size > 0 else 0.0

    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"MAP@{k}": float(np.mean(maps)) if maps else 0.0,
        f"Catalog_Coverage@{k}": float(coverage),
        "Users_Evaluated": len(precisions)
    }
