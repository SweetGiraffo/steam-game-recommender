import json
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from typing import Tuple, Dict, List
from src.config import (
    PROCESSED_INTERACTIONS_FILE,
    USER_ITEM_MATRIX_FILE,
    MAPPINGS_FILE,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset() -> Tuple[pd.DataFrame, csr_matrix, dict]:
    """Load processed interactions, sparse matrix, and ID mappings."""
    df = pd.read_parquet(PROCESSED_INTERACTIONS_FILE)
    sparse_mat = load_npz(USER_ITEM_MATRIX_FILE)
    with open(MAPPINGS_FILE, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    return df, sparse_mat, mappings


def train_test_split_warm(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = RANDOM_SEED
) -> Tuple[csr_matrix, Dict[int, List[int]], Dict[int, List[int]]]:
    """
    User-stratified train/test split for warm evaluation.
    Holds out test_ratio of interactions for each user.
    Returns:
        - train_sparse: csr_matrix for model training
        - test_ground_truth: dict mapping user_idx -> list of held-out game_idx
        - train_ground_truth: dict mapping user_idx -> list of trained game_idx
    """
    np.random.seed(seed)
    train_rows, train_cols, train_data = [], [], []
    test_ground_truth = {}
    train_ground_truth = {}

    grouped = df.groupby('user_idx')

    for user_idx, group in grouped:
        indices = group.index.values
        n_items = len(indices)

        if n_items < 3:
            # Not enough interactions to evaluate reliably, keep in train
            train_rows.extend(group['user_idx'].values)
            train_cols.extend(group['game_idx'].values)
            train_data.extend(group['confidence'].values)
            train_ground_truth[user_idx] = group['game_idx'].tolist()
            continue

        n_test = max(1, int(np.round(n_items * test_ratio)))
        shuffled = np.random.permutation(indices)
        test_idx = shuffled[:n_test]
        train_idx = shuffled[n_test:]

        train_part = group.loc[train_idx]
        test_part = group.loc[test_idx]

        train_rows.extend(train_part['user_idx'].values)
        train_cols.extend(train_part['game_idx'].values)
        train_data.extend(train_part['confidence'].values)

        train_ground_truth[user_idx] = train_part['game_idx'].tolist()
        test_ground_truth[user_idx] = test_part['game_idx'].tolist()

    num_users = df['user_idx'].max() + 1
    num_games = df['game_idx'].max() + 1

    train_sparse = csr_matrix(
        (train_data, (train_rows, train_cols)),
        shape=(num_users, num_games),
        dtype=np.float32
    )

    logger.info(
        f"Created warm split: Train matrix has {train_sparse.nnz} interactions, {len(test_ground_truth)} users in test set."
    )
    return train_sparse, test_ground_truth, train_ground_truth


def train_test_split_cold_items(
    df: pd.DataFrame,
    cold_item_ratio: float = 0.15,
    seed: int = RANDOM_SEED
) -> Tuple[csr_matrix, Dict[int, List[int]], List[int]]:
    """
    Splits items into warm items and completely held-out cold-start items.
    Allows stress-testing hybrid vs pure collaborative filtering on cold-start items.
    """
    np.random.seed(seed)
    all_games = df['game_idx'].unique()
    n_cold = int(len(all_games) * cold_item_ratio)
    cold_games = set(np.random.choice(all_games, size=n_cold, replace=False))
    warm_games = set(all_games) - cold_games

    train_df = df[df['game_idx'].isin(warm_games)].copy()
    test_df = df[df['game_idx'].isin(cold_games)].copy()

    test_ground_truth = {}
    for user_idx, group in test_df.groupby('user_idx'):
        # only evaluate users who also have warm training items so we know their preferences
        if user_idx in train_df['user_idx'].values:
            test_ground_truth[user_idx] = group['game_idx'].tolist()

    num_users = df['user_idx'].max() + 1
    num_games = df['game_idx'].max() + 1

    train_sparse = csr_matrix(
        (train_df['confidence'].values, (train_df['user_idx'].values, train_df['game_idx'].values)),
        shape=(num_users, num_games),
        dtype=np.float32
    )

    logger.info(
        f"Created cold-start item split: {len(cold_games)} cold items held out, {len(test_ground_truth)} evaluation users."
    )
    return train_sparse, test_ground_truth, list(cold_games)
