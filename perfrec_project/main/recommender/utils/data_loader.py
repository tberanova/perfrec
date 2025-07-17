"""
This module provides utilities for loading and constructing a user–item interaction matrix
from CSV data and Django ORM objects. The result is a combined sparse matrix of implicit
feedback from both external and real users, suitable for use in recommender systems.

The loader:
- Parses a CSV file with `user_id`, `perfume_id` interactions
- Maps these to ORM objects via `external_id`
- Constructs a sparse interaction matrix
- Appends rows for real Django users based on their liked perfumes
- Returns various mappings for further use in model training and evaluation
"""

import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack
from django.contrib.auth.models import User
from main.models import Perfume

logger = logging.getLogger(__name__)


def load_perfume_interaction_data(
    interaction_csv_path="../data/interactions.csv"
):
    """
    Loads user–perfume interaction data from a CSV and Django ORM, and constructs
    a sparse interaction matrix including both synthetic and real Django users.

    Args:
        interaction_csv_path (str): Path to the CSV file with columns `user_id` and `perfume_id`.

    Returns:
        dict: A dictionary containing:
            - `interaction_matrix` (csr_matrix): Combined user–item matrix.
            - `django_user_id_to_matrix_id` (dict): Maps Django user IDs to matrix row indices.
            - `matrix_id_to_django_user_id` (dict): Inverse mapping of above.
            - `matrix_id_to_orm_perfume` (dict): Maps matrix column indices to ORM `Perfume` objects.
            - `real_users` (list): List of Django `User` instances.
            - `django_user_id_to_user` (dict): Maps Django user IDs to `User` instances.
            - `ext_perfume_id_to_matrix_id` (dict): Maps perfume external IDs to matrix column indices.
            - `matrix_id_to_ext_perfume_id` (dict): Inverse mapping of above.
    """
    logger.info("Loading interaction CSV...")
    interactions_df = pd.read_csv(
        interaction_csv_path, dtype={"perfume_id": str})
    logger.info(f"→ Total interactions: {len(interactions_df)}")

    logger.info("Querying perfumes with external_id...")
    orm_perfumes = Perfume.objects.exclude(
        external_id__isnull=True).exclude(external_id__exact="")
    ext_id_to_orm_perfume = {p.external_id: p for p in orm_perfumes}

    unique_ext_perfume_ids = sorted(ext_id_to_orm_perfume.keys())
    ext_perfume_id_to_matrix_id = {pid: i for i,
                                   pid in enumerate(unique_ext_perfume_ids)}
    matrix_id_to_ext_perfume_id = {i: pid for pid,
                                   i in ext_perfume_id_to_matrix_id.items()}

    logger.info("Mapping external user IDs...")
    unique_ext_user_ids = interactions_df["user_id"].unique()
    ext_user_id_to_matrix_id = {uid: i for i,
                                uid in enumerate(unique_ext_user_ids)}

    user_matrix_ids = interactions_df["user_id"].map(
        ext_user_id_to_matrix_id).values
    perfume_matrix_ids = interactions_df["perfume_id"].map(
        ext_perfume_id_to_matrix_id).values
    data = np.ones(len(interactions_df), dtype=np.float32)

    logger.info("Building interaction matrix...")
    interaction_matrix = csr_matrix(
        (data, (user_matrix_ids, perfume_matrix_ids)),
        shape=(len(ext_user_id_to_matrix_id), len(ext_perfume_id_to_matrix_id))
    )

    logger.info("Fetching real Django users...")
    real_users = list(User.objects.all())
    django_user_id_to_user = {user.id: user for user in real_users}

    logger.info("Mapping ORM perfume IDs to matrix cols...")
    orm_perfume_id_to_matrix_col = {
        p.id: ext_perfume_id_to_matrix_id[p.external_id]
        for p in orm_perfumes
        if p.external_id in ext_perfume_id_to_matrix_id
    }

    row_indices = []
    col_indices = []

    logger.info("Mapping user.profile liked perfumes...")
    for i, user in enumerate(real_users):
        profile = getattr(user, "profile", None)
        if not profile:
            continue
        for perfume in profile.liked_perfumes.all():
            col = orm_perfume_id_to_matrix_col.get(perfume.id)
            if col is not None:
                row_indices.append(i)
                col_indices.append(col)

    real_data = np.ones(len(row_indices), dtype=np.float32)
    real_user_matrix = csr_matrix(
        (real_data, (row_indices, col_indices)),
        shape=(len(real_users), interaction_matrix.shape[1])
    )

    logger.info("Stacking interaction matrix + real user matrix...")
    full_matrix = vstack([interaction_matrix, real_user_matrix])

    start_row = interaction_matrix.shape[0]
    django_user_id_to_matrix_id = {
        user.id: start_row + i for i, user in enumerate(real_users)}
    matrix_id_to_django_user_id = {v: k for k,
                                   v in django_user_id_to_matrix_id.items()}

    # Validate ORM consistency
    missing_ext_ids = [
        ext_id for ext_id in ext_perfume_id_to_matrix_id
        if ext_id not in ext_id_to_orm_perfume
    ]
    if missing_ext_ids:
        logger.warning(
            f"{len(missing_ext_ids)} external_ids in CSV not found in ORM!")
        logger.warning(f"Example missing IDs: {missing_ext_ids[:10]}")
        raise ValueError("Mismatch between interaction CSV and ORM perfumes.")
    else:
        logger.info("All perfume external_ids in CSV are matched in the ORM.")

    matrix_id_to_orm_perfume = {
        mid: ext_id_to_orm_perfume.get(ext_id)
        for ext_id, mid in ext_perfume_id_to_matrix_id.items()
        if ext_id in ext_id_to_orm_perfume
    }

    # Prune matrix columns to valid ORM perfumes
    valid_perfume_indices = sorted(matrix_id_to_orm_perfume.keys())
    full_matrix = full_matrix[:, valid_perfume_indices]
    new_matrix_id_to_orm_perfume = {
        new_idx: matrix_id_to_orm_perfume[old_idx]
        for new_idx, old_idx in enumerate(valid_perfume_indices)
    }

    logger.info("Summary:")
    logger.info(f"Final matrix shape: {full_matrix.shape}")
    logger.info(f"Perfumes in DB (ORM): {orm_perfumes.count()}")
    logger.info(f"Perfumes in interaction file: {len(unique_ext_perfume_ids)}")
    logger.info(f"Perfumes mapped to ORM: {len(matrix_id_to_orm_perfume)}")
    logger.info(
        f"Perfumes used in recommender: {len(new_matrix_id_to_orm_perfume)}")

    return {
        "interaction_matrix": full_matrix,
        "django_user_id_to_matrix_id": django_user_id_to_matrix_id,
        "matrix_id_to_django_user_id": matrix_id_to_django_user_id,
        "matrix_id_to_orm_perfume": new_matrix_id_to_orm_perfume,
        "real_users": real_users,
        "django_user_id_to_user": django_user_id_to_user,
        "ext_perfume_id_to_matrix_id": ext_perfume_id_to_matrix_id,
        "matrix_id_to_ext_perfume_id": matrix_id_to_ext_perfume_id,
    }
