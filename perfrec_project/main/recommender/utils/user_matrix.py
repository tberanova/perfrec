"""
Matrix utilities for managing user interaction data in a recommender system.

This module provides two main functions:
- `add_user_row_to_matrix`: Appends a new empty interaction row for a newly seen user.
- `update_user_row_in_matrix`: Updates an existing user's interaction vector based on their liked perfumes.

The interaction matrix is a CSR sparse matrix where rows represent users and columns represent items.
"""

import numpy as np
from scipy.sparse import csr_matrix, vstack


def add_user_row_to_matrix(interaction_matrix, real_users, user_id, id_to_user, id_to_index):
    """
    Adds a new empty user row to the interaction matrix.

    This function is typically called when a user interacts with the system for the first time.
    It extends the matrix with a zero-filled row corresponding to the new user.

    Args:
        interaction_matrix (csr_matrix): The existing user-item interaction matrix.
        real_users (list): A list of known user IDs (used to track order/index).
        user_id (Any): Unique ID of the user to be added.
        id_to_user (dict): A mapping from user ID to user instance.
        id_to_index (dict): A mapping from user ID to matrix row index.

    Returns:
        csr_matrix: Updated interaction matrix with a new row for the user.
    """
    if user_id in id_to_index:
        print(f"[WARN] User {user_id} already exists in matrix.")
        return interaction_matrix

    print("user row added")
    n_items = interaction_matrix.shape[1]
    empty_row = csr_matrix((1, n_items), dtype=np.float32)
    new_matrix = vstack([interaction_matrix, empty_row])

    new_index = new_matrix.shape[0] - 1
    id_to_index[user_id] = new_index
    real_users.append(user_id)
    id_to_user[user_id] = id_to_user.get(
        user_id)  # Optionally store user object

    print(f"[INFO] Added new user {user_id} at row {new_index}")
    return new_matrix


def update_user_row_in_matrix(interaction_matrix, user_id, user, id_to_index, ext_id_to_matrix_id):
    """
    Updates the row of a given user in the interaction matrix based on their liked perfumes.

    This function reconstructs the row vector from scratch by iterating over the user's liked perfumes
    and setting the corresponding indices to 1.0.

    Args:
        interaction_matrix (csr_matrix): The user-item interaction matrix.
        user_id (Any): ID of the user whose row should be updated.
        user (User): Django user instance (should have `profile` with `liked_perfumes`).
        id_to_index (dict): Mapping from user ID to matrix row index.
        ext_id_to_matrix_id (dict): Mapping from perfume.external_id to column index in the matrix.

    Returns:
        csr_matrix: Updated interaction matrix.
    """
    if user_id not in id_to_index:
        print(f"[WARN] Tried to update unknown user: {user}")
        return interaction_matrix

    row_idx = id_to_index[user_id]
    profile = getattr(user, "profile", None)
    if profile is None:
        return interaction_matrix

    # Create new row with zeros and set 1.0 where perfume is liked
    new_row = np.zeros((1, interaction_matrix.shape[1]), dtype=np.float32)
    for perfume in profile.liked_perfumes.all():
        ext_id = getattr(perfume, "external_id", None)
        col_idx = ext_id_to_matrix_id.get(ext_id)
        if col_idx is not None:
            new_row[0, col_idx] = 1.0

    # Efficiently modify sparse matrix by converting to LIL format temporarily
    interaction_matrix = interaction_matrix.tolil()
    interaction_matrix[row_idx] = new_row
    interaction_matrix = interaction_matrix.tocsr()

    return interaction_matrix
