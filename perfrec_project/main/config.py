"""
config.py

Centralized configuration for algorithm toggles, file paths, and parameters
used across the perfume recommender system.
"""

# -----------------------
# Enabled Recommendation Algorithms
# -----------------------
ENABLED_ALGORITHMS = {
    "elsae": True,
    "item_knn": True,
    "ease": True,
    "popularity": True,
    "content": True,
    "bpr": False,
    "elsa": False,
    "user_knn": True,
}

DEFAULT_REC_ALG = "ease"

# --- Validity check ---
assert ENABLED_ALGORITHMS[
    DEFAULT_REC_ALG], f"Default recommender '{DEFAULT_REC_ALG}' is not enabled."

# --- Hardcoded enforcement ---
for key in ["item_knn", "popularity", "content"]:
    assert ENABLED_ALGORITHMS.get(
        key) is True, f"Mandatory algorithm '{key}' must be enabled."

# -----------------------
# Feature Toggles
# -----------------------
# Enables semantic neuron labeling for explanations
NEURON_TAGGING_ENABLED = True

# -----------------------
# File Paths
# -----------------------
# TF-IDF or similar vectorizer
VECTORIZER_PATH = "serialized/perfume_vectorizer.pkl"
# Precomputed perfume vectors
VECTORS_PATH = "serialized/perfume_vectors.npy"
# Tags for SAE/ELSAE neurons
NEURON_TAG_PATH = "results/neuron_tag_dict.json"
# Path template for model files
MODEL_PATH_TEMPLATE = "serialized/{name}_model.pkl"

# -----------------------
# Algorithm Hyperparameters
# -----------------------
ALGO_PARAMS = {
    "elsae": {
        "embedding_dim_elsa": 64,
        "embedding_dim_sae": 128,
        "sae_k": 16,
        "sae_l1_coef": 0.1,
        "lr": 0.01,
        "elsa_epochs": 200,
        "sae_epochs": 100,
        "batch_size": 64,
        "patience": 10,
        "device": "cpu"  # or "cuda"
    },
    "ease": {
        "lambda_": 250,
    },
    "bpr": {
        "embedding_dim": 64,
        "lr": 0.01,
        "reg": 0.1,
        "epochs": 2,
        "updates_per_epoch": 50000,
    },
    "item_knn": {
        "metric": "cosine",  # (only cosine supported for now)
    },
    "content": {
        "similarity": "cosine"  # or "jaccard"
    },
    "elsa": {
        "embedding_dim": 64,
        "lr": 0.01,
        "epochs": 200,
        "batch_size": 64,
        "patience": 10,
        "device": "cpu",  # or "cuda"
    },
    "user_knn": {
        "top_k_neighbors": 50,
    }
}

# -----------------------
# Mappings for Category Boosting in Filters
# -----------------------
CATEGORY_ATTRS = {
    "season": "season_chart",
    "occasion": "occasion_chart",
    "type": "type_chart",
}

# -----------------------
# User Profile Gender Preferences
# -----------------------
GENDER_CHOICES = [
    ('Male', 'Male'),
    ('Female', 'Female'),
    ('Unisex', 'Unisex'),
    ('None', 'No Preference'),
]

# -----------------------
# User Profile Gender Preferences
# -----------------------

# Available gender options for user profiles and gender-based filtering
GENDER_CHOICES = [
    ('Male', 'Male'),
    ('Female', 'Female'),
    ('Unisex', 'Unisex'),
    # Used when the user has no specific gender preference
    ('None', 'No Preference'),
]

# -----------------------
# Number of top recommendations to return in user-facing views (default cutoff)
# -----------------------

TOP_N = 15
MAX_N = 60

# -----------------------
# Tags used to classify perfumes and allow contextual recommendations
# -----------------------

SEASON_FILTERS = ["Spring", "Summer", "Fall", "Winter"]

OCCASION_FILTERS = [
    "Business", "Daily", "Sport", "Leisure", "Evening", "Night Out"
]
TYPE_FILTERS = [
    "Powdery", "Floral", "Sweet", "Fruity", "Spicy", "Oriental", "Gourmand", "Creamy", "Citrus",
    "Green", "Fresh", "Aquatic", "Synthetic", "Smoky", "Leathery", "Woody", "Resinous"
]
