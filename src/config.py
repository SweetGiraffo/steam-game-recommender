from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Remote Dataset URLs
STEAM_200K_URL = "https://raw.githubusercontent.com/guilherminog/raw-steam200k/main/steam-200k.csv"
METADATA_PARQUET_URL = "https://huggingface.co/api/datasets/FronkonGames/steam-games-dataset/parquet/default/train/0.parquet"

# Cached Files
RAW_INTERACTIONS_FILE = RAW_DATA_DIR / "steam-200k.csv"
RAW_METADATA_FILE = RAW_DATA_DIR / "steam_games_metadata.parquet"

PROCESSED_INTERACTIONS_FILE = PROCESSED_DATA_DIR / "interactions.parquet"
PROCESSED_CATALOG_FILE = PROCESSED_DATA_DIR / "games_catalog.parquet"
USER_ITEM_MATRIX_FILE = PROCESSED_DATA_DIR / "user_item_sparse.npz"
MAPPINGS_FILE = PROCESSED_DATA_DIR / "mappings.json"

MODEL_CACHE_FILE = ARTIFACTS_DIR / "models.pkl"
TFIDF_CACHE_FILE = ARTIFACTS_DIR / "tfidf_features.npz"
METRICS_FILE = ARTIFACTS_DIR / "metrics.json"

# Recommender Hyperparameters
ALS_FACTORS = 64
ALS_REGULARIZATION = 0.05
ALS_ITERATIONS = 25
ALS_ALPHA = 40.0  # Confidence scaling factor C_ui = 1 + alpha * r_ui
HYBRID_CF_WEIGHT = 0.65  # Collaborative filtering weight vs Content-based
COLD_START_THRESHOLD = 3  # Minimum interactions before relying purely on CF
DEFAULT_TOP_K = 10
RANDOM_SEED = 42
