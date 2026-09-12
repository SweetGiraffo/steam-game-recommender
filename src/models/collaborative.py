import os
import logging
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional, Set
import implicit
from threadpoolctl import threadpool_limits
from src.config import (
    ALS_FACTORS,
    ALS_REGULARIZATION,
    ALS_ITERATIONS,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImplicitALSRecommender:
    """
    Collaborative Filtering using Implicit Alternating Least Squares (Hu, Koren, Volinsky).
    Optimizes latent user and item factors against confidence-weighted implicit feedback.
    """
    def __init__(
        self,
        factors: int = ALS_FACTORS,
        regularization: float = ALS_REGULARIZATION,
        iterations: int = ALS_ITERATIONS,
        random_seed: int = RANDOM_SEED
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_seed = random_seed
        self.model: Optional[implicit.als.AlternatingLeastSquares] = None
        self.train_matrix: Optional[csr_matrix] = None
        self.num_users: int = 0
        self.num_items: int = 0

    def fit(self, train_matrix: csr_matrix):
        """Fits ALS model on sparse user-item interaction matrix."""
        self.train_matrix = train_matrix
        self.num_users, self.num_items = train_matrix.shape

        with threadpool_limits(limits=1, user_api='blas'):
            self.model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=self.random_seed
            )
            logger.info(
                f"Fitting Implicit ALS (factors={self.factors}, reg={self.regularization}, iters={self.iterations})..."
            )
            # implicit accepts user_items matrix
            self.model.fit(self.train_matrix, show_progress=False)
        logger.info("Implicit ALS training completed.")
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        filter_items: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Recommends top-N items for user_idx using learned latent representations.
        """
        if self.model is None or self.train_matrix is None:
            return []

        if user_idx >= self.num_users:
            # Cold-start user not seen during training
            return []

        user_row = self.train_matrix[user_idx]
        ids, scores = self.model.recommend(
            user_idx,
            user_row,
            N=n,
            filter_already_liked_items=True
        )

        filter_set = set(filter_items) if filter_items else set()
        results = []
        for item_id, score in zip(ids, scores):
            if int(item_id) not in filter_set and score > -1e30:
                results.append((int(item_id), float(score)))

        return results[:n]

    def score_all_items_for_user(self, user_idx: int) -> np.ndarray:
        """
        Computes dot-product predicted scores for all items for a given user.
        Returns 0 array if user is unseen/cold-start.
        """
        if self.model is None or user_idx >= self.num_users:
            return np.zeros(self.num_items, dtype=np.float32)

        user_vec = self.model.user_factors[user_idx]
        item_vecs = self.model.item_factors
        # Dot product
        scores = np.dot(item_vecs, user_vec)
        return scores.astype(np.float32)

    def similar_items(self, item_idx: int, n: int = 10) -> List[Tuple[int, float]]:
        """Finds top-N similar items using item factor cosine similarity."""
        if self.model is None or item_idx >= self.num_items:
            return []

        ids, scores = self.model.similar_items(item_idx, N=n + 1)
        results = []
        for idx, score in zip(ids, scores):
            if int(idx) != item_idx:
                results.append((int(idx), float(score)))
        return results[:n]
