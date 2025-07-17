"""
Nested Evaluation Script for Recommender Systems

This Django management command performs a repeated holdout evaluation of multiple
recommender algorithms using nested grid search for hyperparameter tuning.

The script:
- Splits interaction data into train/val/test sets multiple times
- Runs grid search on train/val to select best params
- Evaluates each model on the test set
- Computes and logs metrics: precision@N, recall@N, hits@N, nDCG, ILD, PopLift
- Outputs results as JSON and plots metric comparisons
"""

import time
from collections import Counter
import numpy as np
import pandas as pd
from typing import List
from django.core.management.base import BaseCommand

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid

from main.recommender.utils.data_loader import load_perfume_interaction_data
from main.recommender.utils.vectorizer_io import load_or_fit_vectors
from .utils.eval_config import GRID_PARAMS, MODEL_CLASSES, RUNS, TOP_N
from .utils.plot import plot_metrics

# Set reproducible seed
SEED = 10
np.random.seed(SEED)

# Globals for computing diversity/popularity
GLOBAL_POPULARITY = None
GLOBAL_VECTORS = None


def train_test_holdout(interaction_matrix, test_ratio):
    """
    Split an interaction matrix into train and test via user-wise holdout.

    Each user has a random subset of their interactions held out for testing.

    Args:
        interaction_matrix (csr_matrix): User–item interaction matrix
        test_ratio (float): Fraction of interactions to hold out per user

    Returns:
        train (csr_matrix): Matrix with held-out interactions zeroed
        test (csr_matrix): Matrix with only held-out interactions
    """
    train = interaction_matrix.copy().tolil()
    test = np.zeros_like(interaction_matrix.toarray())

    for user in range(interaction_matrix.shape[0]):
        item_indices = interaction_matrix[user].nonzero()[1]
        if len(item_indices) < 2:
            continue
        test_size = max(1, int(len(item_indices) * test_ratio))
        held_out = np.random.choice(
            item_indices, size=test_size, replace=False)
        train[user, held_out] = 0
        test[user, held_out] = 1

    return train.tocsr(), csr_matrix(test)


def compute_poplift(recommended_items: List[int]) -> float:
    """
    Compute popularity lift for a list of recommended items.

    PopLift = (avg popularity of recommended items - global avg popularity) / global avg popularity

    Args:
        recommended_items (List[int]): List of recommended item indices

    Returns:
        float: PopLift score
    """
    if GLOBAL_POPULARITY is None:
        raise ValueError("GLOBAL_POPULARITY is not initialized.")
    rec_pop = np.mean([GLOBAL_POPULARITY[i] for i in recommended_items])
    avg_pop = np.mean(GLOBAL_POPULARITY)
    return (rec_pop - avg_pop) / avg_pop if avg_pop > 0 else 0.0


def compute_intralist_diversity(recommended_items: List[int]) -> float:
    """
    Compute intra-list diversity (ILD) based on cosine dissimilarity.

    Args:
        recommended_items (List[int]): List of recommended item indices

    Returns:
        float: ILD score
    """
    if GLOBAL_VECTORS is None:
        raise ValueError("GLOBAL_VECTORS is not initialized.")
    if len(recommended_items) < 2:
        return 0.0
    vecs = GLOBAL_VECTORS[recommended_items]
    sim_matrix = cosine_similarity(vecs)
    k = len(recommended_items)
    return (1 - sim_matrix[np.triu_indices(k, k=1)]).mean()


def evaluate_on_holdout_split(recommender, train_matrix, test_matrix, top_n):
    """
    Fit and evaluate a recommender model on a train/test split.

    Args:
        recommender: Model implementing .fit() and .recommend()
        train_matrix (csr_matrix): Training interaction matrix
        test_matrix (csr_matrix): Test interaction matrix
        top_n (int): Number of top recommendations to evaluate

    Returns:
        pd.DataFrame: Per-user metric dataframe with fit/rec time in .attrs
    """
    user_metrics = []
    start_fit = time.time()
    recommender.fit(train_matrix)
    fit_duration = time.time() - start_fit

    start_rec = time.time()
    for user in range(test_matrix.shape[0]):
        test_items = test_matrix[user].nonzero()[1]
        if len(test_items) == 0:
            continue
        user_vector = train_matrix[user].toarray().ravel()
        recommended = recommender.recommend(
            user_vector=user_vector, user_index=user, top_n=top_n)
        recommended_items = [i for i, _ in recommended]
        hit_items = set(test_items) & set(recommended_items)
        hits = len(hit_items)
        dcg = sum(1 / np.log2(i + 2)
                  for i, item in enumerate(recommended_items) if item in test_items)
        idcg = sum(1 / np.log2(i + 2)
                   for i in range(min(len(test_items), top_n)))
        ild = compute_intralist_diversity(recommended_items)
        poplift = compute_poplift(recommended_items)
        user_metrics.append({
            "precision": hits / top_n,
            "recall": hits / len(test_items),
            "hits": 1 if hits > 0 else 0,
            "ndcg": dcg / idcg if idcg > 0 else 0,
            "ild": ild,
            "poplift": poplift
        })

    rec_duration = time.time() - start_rec
    df = pd.DataFrame(user_metrics)
    df.attrs["fit_time"] = fit_duration
    df.attrs["rec_time"] = rec_duration
    return df


def gridsearch_on_train_val(train, val, model_class, param_grid, top_n):
    """
    Find best hyperparameters on train/val using recall.

    Returns:
        best_params (dict): Best config from grid
    """
    best_score = -np.inf
    best_params = None
    for params in ParameterGrid(param_grid):
        print(f"{model_class.__name__} + {params}")
        model = model_class(config=params)
        df = evaluate_on_holdout_split(
            model, train_matrix=train, test_matrix=val, top_n=top_n)
        recall = df["recall"].mean()
        if recall > best_score:
            best_score = recall
            best_params = params
    return best_params


def gridsearch_evaluate_repeated(model_class, param_grid, splits, top_n):
    """
    Run repeated holdout evaluation with internal validation-based grid search.

    Returns:
        all_metrics (List[dict])
        mean_metrics (dict)
    """
    all_metrics = []
    fit_times = []
    rec_times = []

    for run, (train, val, test) in enumerate(splits):
        print(
            f"[Nested Run {run + 1}/{len(splits)}] with {train.shape[0]} users")
        best_params = gridsearch_on_train_val(
            train, val, model_class, param_grid, top_n)
        print(f"  Best params: {best_params}")
        model = model_class(config=best_params)
        train_full = train + val

        df = evaluate_on_holdout_split(
            model, train_matrix=train_full, test_matrix=test, top_n=top_n)
        metric_dict = {
            f"precision@{top_n}": df["precision"].mean(),
            f"recall@{top_n}": df["recall"].mean(),
            f"hits@{top_n}": df["hits"].mean(),
            f"ndcg@{top_n}": df["ndcg"].mean(),
            "poplift": df["poplift"].mean(),
            "ild": df["ild"].mean(),
            "fit_time": df.attrs["fit_time"],
            "rec_time": df.attrs["rec_time"],
            "params": str(best_params)
        }
        all_metrics.append(metric_dict)
        fit_times.append(df.attrs["fit_time"])
        rec_times.append(df.attrs["rec_time"])

    df_runs = pd.DataFrame(all_metrics)
    mean_metrics = df_runs.drop(columns=["params"]).mean().to_dict()
    std_metrics = df_runs.drop(columns=["params"]).std().to_dict()
    mean_metrics.update({f"std_{k}": v for k, v in std_metrics.items()})
    param_counts = Counter(str(m["params"]) for m in all_metrics)
    mean_metrics["params"] = param_counts.most_common(1)[0][0]
    mean_metrics["all_params"] = [str(m["params"]) for m in all_metrics]
    return all_metrics, mean_metrics


class Command(BaseCommand):
    """
    Django command to run the full nested evaluation pipeline.

    Saves full and summary results to JSON and plots metrics.
    """
    help = "Run nested evaluation."

    def handle(self, *args, **options):
        data = load_perfume_interaction_data()
        matrix = data["interaction_matrix"]
        top_n = TOP_N
        runs = RUNS

        # Initialize global popularity and feature vectors
        global GLOBAL_POPULARITY
        perfumes = list(data["matrix_id_to_orm_perfume"].values())
        GLOBAL_POPULARITY = np.array([p.rating_count for p in perfumes])

        global GLOBAL_VECTORS
        _, GLOBAL_VECTORS = load_or_fit_vectors(
            perfumes,
            vectorizer_path="serialized/perfume_vectorizer.pkl",
            vectors_path="serialized/perfume_vectors.npy"
        )

        # Generate repeated splits
        splits = []
        for _ in range(runs):
            test_ratio = 0.2
            validation_ratio = 0.2 / (1 - test_ratio)
            train_full, test = train_test_holdout(
                matrix, test_ratio=test_ratio)
            train, val = train_test_holdout(
                train_full, test_ratio=validation_ratio)
            print(
                f"Generated split with train interactions: {train.nnz}, val: {val.nnz}, test: {test.nnz}")
            splits.append((train, val, test))

        all_results = []
        summary_rows = []

        for name, model_class in MODEL_CLASSES.items():
            print(f"\nEvaluating {name.upper()}...")
            param_grid = GRID_PARAMS.get(name, {})
            param_grid = {k: v if isinstance(v, list) else [
                v] for k, v in param_grid.items()}

            # Wrap content-based models with vector input
            if name == "content":
                vectorizer, vectors = load_or_fit_vectors(
                    perfumes,
                    vectorizer_path="serialized/perfume_vectorizer.pkl",
                    vectors_path="serialized/perfume_vectors.npy"
                )
                index_to_perfume_id = {i: p.id for i, p in enumerate(perfumes)}

                def content_wrapper(config):
                    return model_class(
                        perfume_vectors=vectors,
                        index_to_perfume_id=index_to_perfume_id,
                        config=config
                    )
                eval_class = content_wrapper

            # Wrap popularity model (no config)
            elif name == "popularity":
                popularity_model_instance = model_class(perfumes=perfumes)

                def popularity_wrapper(config):
                    return popularity_model_instance

                eval_class = popularity_wrapper

            else:
                eval_class = model_class

            # Evaluate using nested holdout
            model_metrics, summary = gridsearch_evaluate_repeated(
                model_class=eval_class,
                param_grid=param_grid,
                splits=splits,
                top_n=top_n
            )

            for m in model_metrics:
                m["algorithm"] = name.upper()
                all_results.append(m)
            summary["algorithm"] = name.upper()
            summary_rows.append(summary)

        # Save and plot results
        df_all = pd.DataFrame(all_results)
        df_summary = pd.DataFrame(summary_rows)
        df_all.to_json("results/nested_eval_results.json",
                       orient="records", indent=2)
        df_summary.to_json("results/nested_eval_summary.json",
                           orient="records", indent=2)
        plot_metrics(df_summary, top_k=top_n)
