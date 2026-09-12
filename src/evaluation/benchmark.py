import json
import pickle
import logging
import pandas as pd
from tabulate import tabulate
from src.config import (
    MODEL_CACHE_FILE,
    METRICS_FILE,
    ARTIFACTS_DIR,
    RANDOM_SEED
)
from src.data.dataset import (
    load_dataset,
    train_test_split_warm,
    train_test_split_cold_items
)
from src.models.baseline import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import ImplicitALSRecommender
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import evaluate_model_ranking

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_benchmark():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Offline Recommendation Benchmark...")

    # 1. Load Data
    df_interactions, sparse_mat, mappings = load_dataset()
    idx2game = mappings['idx2game']
    game2idx = mappings['game2idx']

    # 2. Warm Split
    train_sparse, test_warm_gt, train_warm_gt = train_test_split_warm(
        df_interactions,
        test_ratio=0.2,
        seed=RANDOM_SEED
    )

    # 3. Train Models
    logger.info("--- Training Models on Warm Split ---")

    # Popularity
    logger.info("Fitting Popularity Baseline...")
    pop_model = PopularityRecommender().fit(train_sparse)

    # Content-Based
    logger.info("Fitting Content-Based Model...")
    cb_model = ContentBasedRecommender().fit(idx2game=idx2game, game2idx=game2idx)

    # Collaborative Filtering (Implicit ALS)
    logger.info("Fitting Implicit ALS Model...")
    cf_model = ImplicitALSRecommender().fit(train_sparse)

    # Hybrid
    logger.info("Constructing Adaptive Hybrid Model...")
    hybrid_model = HybridRecommender(
        cf_model=cf_model,
        cb_model=cb_model,
        pop_model=pop_model
    )

    # 4. Evaluate Warm Split
    logger.info("--- Evaluating Warm Split (K=10) ---")
    results_warm = {}
    models_to_eval = [
        ('Popularity Baseline', pop_model, 'pop'),
        ('Content-Based (TF-IDF)', cb_model, 'cb'),
        ('Implicit ALS (Pure CF)', cf_model, 'cf'),
        ('Hybrid (ALS + Content)', hybrid_model, 'hybrid')
    ]

    for name, model, m_type in models_to_eval:
        logger.info(f"Evaluating {name} on warm test set...")
        res = evaluate_model_ranking(
            model=model,
            train_matrix=train_sparse,
            test_ground_truth=test_warm_gt,
            k=10,
            model_type=m_type
        )
        results_warm[name] = res

    # 5. Evaluate Cold-Start Experiment
    logger.info("--- Running Cold-Start Item Stress Test ---")
    cold_train_sparse, cold_test_gt, cold_items = train_test_split_cold_items(
        df_interactions,
        cold_item_ratio=0.15,
        seed=RANDOM_SEED
    )

    # Train CF and Hybrid specifically for cold test
    cold_cf_model = ImplicitALSRecommender().fit(cold_train_sparse)
    cold_hybrid = HybridRecommender(
        cf_model=cold_cf_model,
        cb_model=cb_model,
        pop_model=pop_model
    )

    results_cold = {}
    results_cold['Implicit ALS (Pure CF - Cold)'] = evaluate_model_ranking(
        model=cold_cf_model,
        train_matrix=cold_train_sparse,
        test_ground_truth=cold_test_gt,
        k=10,
        model_type='cf'
    )
    results_cold['Hybrid Engine (Cold)'] = evaluate_model_ranking(
        model=cold_hybrid,
        train_matrix=cold_train_sparse,
        test_ground_truth=cold_test_gt,
        k=10,
        model_type='hybrid'
    )

    # 6. Format and Display Results
    warm_rows = []
    for model_name, m in results_warm.items():
        warm_rows.append([
            model_name,
            f"{m['Precision@10']:.4f}",
            f"{m['Recall@10']:.4f}",
            f"{m['NDCG@10']:.4f}",
            f"{m['MAP@10']:.4f}",
            f"{m['Catalog_Coverage@10'] * 100:.2f}%"
        ])

    print("\n" + "=" * 75)
    print("WARM RECOMMENDATION BENCHMARK (K=10)")
    print("=" * 75)
    print(tabulate(
        warm_rows,
        headers=["Model", "Precision@10", "Recall@10", "NDCG@10", "MAP@10", "Coverage"],
        tablefmt="github"
    ))

    cold_rows = []
    for model_name, m in results_cold.items():
        cold_rows.append([
            model_name,
            f"{m['Precision@10']:.4f}",
            f"{m['Recall@10']:.4f}",
            f"{m['NDCG@10']:.4f}",
            f"{m['MAP@10']:.4f}",
            f"{m['Catalog_Coverage@10'] * 100:.2f}%"
        ])

    print("\n" + "=" * 75)
    print("COLD-START ITEM BENCHMARK (K=10)")
    print("=" * 75)
    print(tabulate(
        cold_rows,
        headers=["Model", "Precision@10", "Recall@10", "NDCG@10", "MAP@10", "Coverage"],
        tablefmt="github"
    ))
    print("=" * 75 + "\n")

    # 7. Save Artifacts for API and UI
    # Now retrain final production models on full interaction matrix
    logger.info("--- Retraining Final Production Models on 100% Interaction Matrix ---")
    final_cf = ImplicitALSRecommender().fit(sparse_mat)
    final_hybrid = HybridRecommender(
        cf_model=final_cf,
        cb_model=cb_model,
        pop_model=pop_model
    )

    artifacts = {
        'pop_model': pop_model,
        'cb_model': cb_model,
        'cf_model': final_cf,
        'hybrid_model': final_hybrid,
        'sparse_matrix': sparse_mat,
        'mappings': mappings
    }

    with open(MODEL_CACHE_FILE, 'wb') as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved serialized models to {MODEL_CACHE_FILE}")

    metrics_payload = {
        'warm_benchmark': results_warm,
        'cold_benchmark': results_cold
    }
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved benchmark metrics to {METRICS_FILE}")

    return metrics_payload


if __name__ == '__main__':
    run_benchmark()
