from main.recommender.algos.elsa import ELSARecommender
from main.recommender.algos.elsa_sae import ELSAERecommender
from main.recommender.algos.ease import EASERecommender
from main.recommender.algos.i_knn import ItemKNNRecommender
from main.recommender.algos.u_knn import UserKNNRecommender
from main.recommender.algos.popularity import PopularityRecommender
from main.recommender.algos.bpr import BPRRecommender
from main.recommender.algos.similarity import ContentSimilarityRecommender

RUNS = 5
TOP_N = 15

MODEL_CLASSES = {
    "content": ContentSimilarityRecommender,
    "popularity": PopularityRecommender,
    "item_knn": ItemKNNRecommender,
    "user_knn": UserKNNRecommender,
    "ease": EASERecommender,
    "elsa": ELSARecommender,
    "elsae": ELSAERecommender,
    "bpr": BPRRecommender,
}


GRID_PARAMS = {
    "elsae": {
        "embedding_dim_elsa": [64],
        "embedding_dim_sae": [128, 256],
        "sae_k": [16],
        "sae_l1_coef": [0.1, 0.25],
        "lr": [0.001,],
        "elsa_epochs": [200],
        "sae_epochs": [100],
        "patience": [10],
        "batch_size": [64],
        "device": ["cpu"]
    },
    "elsa": {
        "embedding_dim": [64, 128, 256],
        "lr": [0.001, 0.002,],
        "epochs": [100, 200],
        "batch_size": [64],
        "patience": [10],
        "device": ["cpu"]
    },
    "ease": {
        "lambda_": [50, 200, 500, 600, 700, 800, 900, 1000]
    },
    "bpr": {
        "embedding_dim": [128, 256],
        "lr": [0.05],
        "reg": [0.005],
        "epochs": [50, 100],
        "use_bias": [False, True]
    },
    "user_knn": {
        "top_k_neighbors": [30, 50, 100]
    },
    "item_knn": {
        "metric": "cosine",  # (only cosine supported for now)
    },
    "content": {
        "similarity": ["cosine", "jaccard"]
    }

}
