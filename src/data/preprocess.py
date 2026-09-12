import json
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from src.config import (
    RAW_INTERACTIONS_FILE,
    PROCESSED_INTERACTIONS_FILE,
    USER_ITEM_MATRIX_FILE,
    MAPPINGS_FILE,
    ALS_ALPHA,
    PROCESSED_DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_interactions(min_user_interactions: int = 3, min_game_interactions: int = 2):
    """
    Transforms raw Steam-200k interactions into confidence-weighted implicit feedback.
    Applies k-core filtering to prune inactive users/items.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading raw interactions from {RAW_INTERACTIONS_FILE}...")

    df = pd.read_csv(
        RAW_INTERACTIONS_FILE,
        header=None,
        names=['user_id', 'game_title', 'behavior', 'value', 'zero']
    )
    df = df.dropna(subset=['user_id', 'game_title', 'behavior', 'value'])

    # Pivot / Aggregate purchases and playtimes
    # Behavior: 'purchase' (value=1.0) and 'play' (value=hours played)
    play_df = df[df['behavior'] == 'play'][['user_id', 'game_title', 'value']].rename(columns={'value': 'hours'})
    purch_df = df[df['behavior'] == 'purchase'][['user_id', 'game_title']].drop_duplicates()
    purch_df['purchased'] = 1

    merged = pd.merge(purch_df, play_df, on=['user_id', 'game_title'], how='outer')
    merged['purchased'] = merged['purchased'].fillna(1).astype(int)
    merged['hours'] = merged['hours'].fillna(0.0).astype(float)

    # Calculate implicit interaction strength:
    # Playtime is strong engagement signal; ownership without play has base weight 1.0
    merged['interaction_strength'] = np.where(merged['hours'] > 0, merged['hours'] + 1.0, 1.0)

    # K-core filtering
    user_counts = merged['user_id'].value_counts()
    game_counts = merged['game_title'].value_counts()

    valid_users = user_counts[user_counts >= min_user_interactions].index
    valid_games = game_counts[game_counts >= min_game_interactions].index

    filtered = merged[merged['user_id'].isin(valid_users) & merged['game_title'].isin(valid_games)].copy()
    logger.info(
        f"Filtered interactions: {len(filtered)} rows across {filtered['user_id'].nunique()} users and {filtered['game_title'].nunique()} games."
    )

    # Confidence calculation: C_ui = 1 + alpha * log(1 + r_ui / epsilon)
    # Log transformation handles heavy-tailed Pareto distribution of playtime
    filtered['confidence'] = 1.0 + ALS_ALPHA * np.log1p(filtered['interaction_strength'])

    # Build Index Mappings
    unique_users = sorted(filtered['user_id'].unique())
    unique_games = sorted(filtered['game_title'].unique())

    user2idx = {str(uid): idx for idx, uid in enumerate(unique_users)}
    idx2user = {idx: str(uid) for idx, uid in enumerate(unique_users)}
    game2idx = {str(g): idx for idx, g in enumerate(unique_games)}
    idx2game = {idx: str(g) for idx, g in enumerate(unique_games)}

    filtered['user_idx'] = filtered['user_id'].astype(str).map(user2idx)
    filtered['game_idx'] = filtered['game_title'].astype(str).map(game2idx)

    # Save cleaned interactions
    filtered.to_parquet(PROCESSED_INTERACTIONS_FILE, index=False)
    logger.info(f"Saved cleaned interactions to {PROCESSED_INTERACTIONS_FILE}")

    # Build Sparse CSR Matrix (num_users x num_games)
    rows = filtered['user_idx'].values
    cols = filtered['game_idx'].values
    data = filtered['confidence'].values

    num_users = len(unique_users)
    num_games = len(unique_games)

    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_games), dtype=np.float32)
    save_npz(USER_ITEM_MATRIX_FILE, sparse_matrix)
    logger.info(f"Saved sparse user-item matrix of shape {sparse_matrix.shape} (sparsity: {100.0 * (1 - sparse_matrix.nnz / (num_users * num_games)):.2f}%) to {USER_ITEM_MATRIX_FILE}")

    # Save Mappings
    mappings = {
        'user2idx': user2idx,
        'idx2user': idx2user,
        'game2idx': game2idx,
        'idx2game': idx2game,
        'num_users': num_users,
        'num_games': num_games
    }
    with open(MAPPINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(mappings, f)
    logger.info(f"Saved ID mappings to {MAPPINGS_FILE}")

    return filtered, sparse_matrix, mappings


if __name__ == '__main__':
    process_interactions()
