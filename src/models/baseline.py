import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional, Set


class PopularityRecommender:
    """
    Non-personalized baseline recommender that suggests items with
    the highest aggregated interaction confidence/playtime.
    """
    def __init__(self):
        self.popular_indices: np.ndarray = np.array([])
        self.popular_scores: np.ndarray = np.array([])

    def fit(self, train_matrix: csr_matrix):
        """Compute item popularity from the sparse training matrix."""
        # Sum column-wise (total confidence received per game)
        item_scores = np.array(train_matrix.sum(axis=0)).flatten()
        # Sort descending
        sorted_indices = np.argsort(-item_scores)
        self.popular_indices = sorted_indices
        self.popular_scores = item_scores[sorted_indices]
        return self

    def recommend(
        self,
        user_idx: Optional[int],
        n: int = 10,
        filter_items: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Returns top-N popular items, optionally filtering out items
        already owned/played by the user.
        """
        filter_set = set(filter_items) if filter_items else set()
        recommendations = []

        for idx, score in zip(self.popular_indices, self.popular_scores):
            if idx in filter_set:
                continue
            recommendations.append((int(idx), float(score)))
            if len(recommendations) >= n:
                break

        return recommendations
