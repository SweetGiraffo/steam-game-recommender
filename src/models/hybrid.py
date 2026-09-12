import logging
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional, Set
from src.config import HYBRID_CF_WEIGHT, COLD_START_THRESHOLD
from src.models.collaborative import ImplicitALSRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.baseline import PopularityRecommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def min_max_scale(arr: np.ndarray) -> np.ndarray:
    """Min-Max scales an array into [0, 1]."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    denom = max_val - min_val
    if denom > 1e-8:
        return (arr - min_val) / denom
    return np.zeros_like(arr)


class HybridRecommender:
    """
    Adaptive Hybrid Recommender combining Implicit ALS and Content-Based representations.
    Solves the cold-start problem via dynamic weighting and semantic fallback.
    """
    def __init__(
        self,
        cf_model: ImplicitALSRecommender,
        cb_model: ContentBasedRecommender,
        pop_model: Optional[PopularityRecommender] = None,
        cf_weight: float = HYBRID_CF_WEIGHT,
        cold_threshold: int = COLD_START_THRESHOLD
    ):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.pop_model = pop_model
        self.cf_weight = cf_weight
        self.cold_threshold = cold_threshold

    def recommend(
        self,
        user_idx: Optional[int],
        train_matrix: csr_matrix,
        n: int = 10,
        filter_items: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Recommends top-N items using adaptive hybrid fusion.
        Automatically detects cold-start state and routes accordingly.
        """
        num_items = train_matrix.shape[1]
        filter_set = set(filter_items) if filter_items else set()

        # Case 1: Unseen or Cold-Start User
        if user_idx is None or user_idx >= train_matrix.shape[0]:
            # Absolute cold user: fallback to popularity or content
            if self.pop_model is not None:
                return self.pop_model.recommend(user_idx=None, n=n, filter_items=filter_set)
            return []

        user_row = train_matrix[user_idx]
        user_items = user_row.indices.tolist()
        user_weights = user_row.data.tolist()
        user_inter_count = len(user_items)

        # Include user's existing items in filter_set
        for item in user_items:
            filter_set.add(item)

        # Case 2: Near Cold-Start User (< cold_threshold interactions)
        if user_inter_count < self.cold_threshold:
            # Shift weight almost entirely to content-based
            cb_scores = self.cb_model.score_items_for_user(user_items, user_weights)
            # Mask filtered items
            for f_item in filter_set:
                if f_item < len(cb_scores):
                    cb_scores[f_item] = -1e9
            top_indices = np.argpartition(-cb_scores, n)[:n]
            top_sorted = top_indices[np.argsort(-cb_scores[top_indices])]
            return [(int(idx), float(cb_scores[idx])) for idx in top_sorted]

        # Case 3: Warm User - Weighted Late Fusion
        cf_raw_scores = self.cf_model.score_all_items_for_user(user_idx)
        cb_raw_scores = self.cb_model.score_items_for_user(user_items, user_weights)

        # Normalize scores to [0, 1] range
        cf_norm = min_max_scale(cf_raw_scores)
        cb_norm = min_max_scale(cb_raw_scores)

        # Ensure matching lengths
        min_len = min(len(cf_norm), len(cb_norm), num_items)
        hybrid_scores = (
            self.cf_weight * cf_norm[:min_len] +
            (1.0 - self.cf_weight) * cb_norm[:min_len]
        )

        # Mask filtered items
        for f_item in filter_set:
            if f_item < min_len:
                hybrid_scores[f_item] = -1e9

        top_indices = np.argpartition(-hybrid_scores, n)[:n]
        top_sorted = top_indices[np.argsort(-hybrid_scores[top_indices])]
        return [(int(idx), float(hybrid_scores[idx])) for idx in top_sorted]

    def recommend_custom(
        self,
        liked_game_indices: List[int],
        n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Interactive cold-start recommendation for a new player who selected
        a list of favorite games.
        """
        if not liked_game_indices:
            if self.pop_model:
                return self.pop_model.recommend(None, n=n)
            return []

        cb_scores = self.cb_model.score_items_for_user(liked_game_indices)
        filter_set = set(liked_game_indices)
        for f_item in filter_set:
            if f_item < len(cb_scores):
                cb_scores[f_item] = -1e9

        top_indices = np.argpartition(-cb_scores, n)[:n]
        top_sorted = top_indices[np.argsort(-cb_scores[top_indices])]
        return [(int(idx), float(cb_scores[idx])) for idx in top_sorted]
