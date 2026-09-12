import json
import pickle
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from src.config import (
    MODEL_CACHE_FILE,
    METRICS_FILE,
    PROCESSED_CATALOG_FILE,
    DEFAULT_TOP_K
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title='Steam Game Recommendation API',
    description='Production-ready hybrid recommender combining Implicit ALS, Content-Based semantic matching, and Cold-Start resolution.',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Global State Loaded on Startup
models = {}
catalog_df = None
game_lookup = {}
id_mappings = {}


class GameInfo(BaseModel):
    game_idx: int
    game_title: str
    genres: str
    categories: str
    price: float
    metacritic_score: int
    header_image: str
    score: Optional[float] = None


class RecommendationResponse(BaseModel):
    user_id: Optional[str] = None
    model_used: str
    recommendations: List[GameInfo]
    user_history: Optional[List[GameInfo]] = None


class ColdStartRequest(BaseModel):
    liked_games: List[str]
    k: int = DEFAULT_TOP_K


@app.on_event('startup')
def load_artifacts():
    global models, catalog_df, game_lookup, id_mappings
    logger.info('Loading models and catalog into memory...')

    if not MODEL_CACHE_FILE.exists():
        raise RuntimeError(f'Model artifact not found at {MODEL_CACHE_FILE}. Run benchmark first.')

    with open(MODEL_CACHE_FILE, 'rb') as f:
        models = pickle.load(f)

    catalog_df = pd.read_parquet(PROCESSED_CATALOG_FILE)
    id_mappings = models['mappings']
    game2idx = id_mappings['game2idx']

    # Pre-index catalog by game_idx
    game_lookup = {}
    for _, row in catalog_df.iterrows():
        title = row['game_title']
        if title in game2idx:
            g_idx = game2idx[title]
            game_lookup[g_idx] = {
                'game_idx': g_idx,
                'game_title': title,
                'genres': str(row.get('genres', '')),
                'categories': str(row.get('categories', '')),
                'price': float(row.get('price', 0.0)),
                'metacritic_score': int(row.get('metacritic_score', 0)),
                'header_image': str(row.get('header_image', ''))
            }

    logger.info(f'Loaded {len(game_lookup)} games and models successfully.')


@app.get('/health')
def health_check():
    return {
        'status': 'healthy',
        'num_users': id_mappings.get('num_users', 0),
        'num_games': id_mappings.get('num_games', 0),
        'models_available': ['popularity', 'content_based', 'implicit_als', 'hybrid']
    }


@app.get('/games/search', response_model=List[GameInfo])
def search_games(query: str = Query(..., min_length=2), limit: int = 10):
    """Autocomplete search for game titles in the catalog."""
    q = query.lower()
    matches = []
    for g_idx, info in game_lookup.items():
        if q in info['game_title'].lower():
            matches.append(GameInfo(**info))
            if len(matches) >= limit:
                break
    return matches


@app.get('/recommend/user/{user_id}', response_model=RecommendationResponse)
def recommend_for_user(
    user_id: str,
    k: int = Query(DEFAULT_TOP_K, ge=1, le=50),
    model: str = Query('hybrid', pattern='^(hybrid|cf|cb|popularity)$')
):
    """Generates personalized game recommendations for a specific Steam user ID."""
    user2idx = id_mappings.get('user2idx', {})
    sparse_mat = models['sparse_matrix']

    if user_id not in user2idx:
        raise HTTPException(
            status_code=404,
            detail=f'User ID {user_id} not found in historical interaction database.'
        )

    user_idx = user2idx[user_id]
    user_row = sparse_mat[user_idx]
    user_items = user_row.indices.tolist()

    # User history items
    history_games = []
    for idx in user_items[:8]:
        if idx in game_lookup:
            history_games.append(GameInfo(**game_lookup[idx]))

    # Model inference
    if model == 'cf':
        recs = models['cf_model'].recommend(user_idx, n=k)
    elif model == 'cb':
        recs = models['cb_model'].recommend(user_idx, train_matrix=sparse_mat, n=k)
    elif model == 'popularity':
        recs = models['pop_model'].recommend(user_idx, n=k, filter_items=set(user_items))
    else:  # hybrid
        recs = models['hybrid_model'].recommend(user_idx, train_matrix=sparse_mat, n=k)

    rec_games = []
    for g_idx, score in recs:
        if g_idx in game_lookup:
            info = dict(game_lookup[g_idx])
            info['score'] = float(score)
            rec_games.append(GameInfo(**info))

    return RecommendationResponse(
        user_id=user_id,
        model_used=model,
        recommendations=rec_games,
        user_history=history_games
    )


@app.get('/recommend/game/{game_title}', response_model=List[GameInfo])
def recommend_similar_games(game_title: str, k: int = Query(DEFAULT_TOP_K, ge=1, le=50)):
    """Finds games similar to a given title using item factor embeddings & content similarity."""
    game2idx = id_mappings.get('game2idx', {})
    if game_title not in game2idx:
        # Try case-insensitive matching
        match = None
        for title, idx in game2idx.items():
            if title.lower() == game_title.lower():
                match = idx
                break
        if match is None:
            raise HTTPException(status_code=404, detail=f'Game \"{game_title}\" not found in catalog.')
        game_idx = match
    else:
        game_idx = game2idx[game_title]

    # Combine CF item factor similarity with Content similarity
    cf_similar = dict(models['cf_model'].similar_items(game_idx, n=k * 2))
    cb_similar = dict(models['cb_model'].similar_items(game_idx, n=k * 2))

    all_candidates = set(cf_similar.keys()).union(set(cb_similar.keys()))
    fused_scores = []

    for c_idx in all_candidates:
        s_cf = cf_similar.get(c_idx, 0.0)
        s_cb = cb_similar.get(c_idx, 0.0)
        # Combine normalized similarity
        score = 0.5 * s_cf + 0.5 * s_cb
        fused_scores.append((c_idx, score))

    fused_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for g_idx, score in fused_scores[:k]:
        if g_idx in game_lookup:
            info = dict(game_lookup[g_idx])
            info['score'] = float(score)
            results.append(GameInfo(**info))

    return results


@app.post('/recommend/cold-start', response_model=RecommendationResponse)
def recommend_cold_start(req: ColdStartRequest):
    """Cold-start onboarding recommendations based on picked favorite titles."""
    game2idx = id_mappings.get('game2idx', {})
    liked_indices = []

    for title in req.liked_games:
        if title in game2idx:
            liked_indices.append(game2idx[title])
        else:
            for g_name, idx in game2idx.items():
                if g_name.lower() == title.lower():
                    liked_indices.append(idx)
                    break

    recs = models['hybrid_model'].recommend_custom(liked_indices, n=req.k)
    rec_games = []
    for g_idx, score in recs:
        if g_idx in game_lookup:
            info = dict(game_lookup[g_idx])
            info['score'] = float(score)
            rec_games.append(GameInfo(**info))

    return RecommendationResponse(
        user_id='cold_start_player',
        model_used='hybrid_cold_start',
        recommendations=rec_games
    )


@app.get('/evaluation/metrics')
def get_benchmark_metrics():
    """Returns offline benchmark ranking metrics comparing all models."""
    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail='Metrics file not found.')
    with open(METRICS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)
