import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Set, Dict
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from src.config import PROCESSED_CATALOG_FILE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Content-Based filtering using game metadata (genres, categories, developer, descriptions).
    Computes TF-IDF representations and builds user profile vectors via weighted aggregation.
    """
    def __init__(self, max_features: int = 8000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.item_vectors: Optional[np.ndarray] = None
        self.num_games: int = 0
        self.idx2game: Dict[int, str] = {}
        self.game2idx: Dict[str, int] = {}

    def fit(self, idx2game: Dict[int, str], game2idx: Dict[str, int]):
        """Builds TF-IDF content representations for each game index."""
        self.idx2game = {int(k): v for k, v in idx2game.items()}
        self.game2idx = {k: int(v) for k, v in game2idx.items()}
        self.num_games = len(self.idx2game)

        # Load catalog
        catalog_df = pd.read_parquet(PROCESSED_CATALOG_FILE)
        # Create map from title to metadata text
        cat_map = {}
        for _, row in catalog_df.iterrows():
            title = row['game_title']
            text = f"{title} {row.get('genres', '')} {row.get('categories', '')} {row.get('developer', '')} {row.get('description', '')}"
            cat_map[title] = text

        # Assemble text in index order
        corpus = []
        for i in range(self.num_games):
            title = self.idx2game.get(i, "")
            text = cat_map.get(title, title)
            corpus.append(text)

        logger.info(f"Fitting TF-IDF on {len(corpus)} game metadata documents...")
        tfidf_sparse = self.vectorizer.fit_transform(corpus)
        # L2-normalize sparse vectors for fast cosine similarity and compact serialization
        self.item_vectors = normalize(tfidf_sparse, norm='l2', axis=1).astype(np.float32).tocsr()
        logger.info(f"Content matrix shape: {self.item_vectors.shape} (sparse format, {self.item_vectors.nnz} non-zeros)")
        return self

    def similar_items(self, game_idx: int, n: int = 10) -> List[Tuple[int, float]]:
        """Returns top-N most semantically similar games to game_idx."""
        if self.item_vectors is None or game_idx >= self.num_games:
            return []
        
        target_vec = self.item_vectors[game_idx]
        sim_scores = np.asarray(self.item_vectors.dot(target_vec.T).toarray()).flatten()
        # Exclude self
        sim_scores[game_idx] = -1.0
        top_indices = np.argpartition(-sim_scores, n)[:n]
        top_sorted = top_indices[np.argsort(-sim_scores[top_indices])]

        return [(int(idx), float(sim_scores[idx])) for idx in top_sorted]

    def score_items_for_user(
        self,
        user_interacted_items: List[int],
        user_interacted_weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Computes content match scores across all items for a given user profile."""
        if not user_interacted_items or self.item_vectors is None:
            return np.zeros(self.num_games, dtype=np.float32)

        valid_items = [idx for idx in user_interacted_items if idx < self.num_games]
        if not valid_items:
            return np.zeros(self.num_games, dtype=np.float32)

        if user_interacted_weights is not None:
            weights = np.array(user_interacted_weights, dtype=np.float32)
            # Log compress weights to prevent high-playtime game from swamping profile
            weights = np.log1p(weights)
            weights = weights / (np.sum(weights) + 1e-8)
            sub_mat = self.item_vectors[valid_items]
            profile = np.asarray(sub_mat.T.dot(weights)).flatten()
        else:
            sub_mat = self.item_vectors[valid_items]
            profile = np.asarray(sub_mat.mean(axis=0)).flatten()

        # Normalize profile
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm

        scores = np.asarray(self.item_vectors.dot(profile)).flatten()
        return scores.astype(np.float32)

    def recommend(
        self,
        user_idx: Optional[int],
        train_matrix: Optional[csr_matrix] = None,
        n: int = 10,
        filter_items: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Recommends top-N games for user based on their interaction history."""
        if user_idx is None or train_matrix is None or user_idx >= train_matrix.shape[0]:
            return []

        user_row = train_matrix[user_idx]
        items = user_row.indices.tolist()
        weights = user_row.data.tolist()

        scores = self.score_items_for_user(items, weights)

        filter_set = set(filter_items) if filter_items else set(items)
        # Mask out filter items
        for f_idx in filter_set:
            if f_idx < len(scores):
                scores[f_idx] = -1e9

        top_indices = np.argpartition(-scores, n)[:n]
        top_sorted = top_indices[np.argsort(-scores[top_indices])]
        return [(int(idx), float(scores[idx])) for idx in top_sorted]
