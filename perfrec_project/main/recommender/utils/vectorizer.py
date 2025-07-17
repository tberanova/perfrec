"""
PerfumeVectorizer module for encoding perfume metadata into numerical feature vectors.

This module defines a `PerfumeVectorizer` class that transforms various textual,
categorical, and multi-label fields from perfume objects into fixed-size NumPy vectors.
The vectorizer supports TF-IDF for descriptions, one-hot encoding for brand and gender,
and multi-label binarization for accords and notes.

Serialization to and from disk is also supported via pickle.

Dependencies:
    - NumPy
    - Scikit-learn
"""

import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
# from sklearn.feature_extraction.text import TfidfVectorizer

# Description vectorization is currently disabled, as I felt descriptions
# were short and of little value.


class PerfumeVectorizer:
    """
    Vectorizer for transforming perfume ORM objects into fixed-size feature vectors.

    Attributes:
        tfidf (TfidfVectorizer): TF-IDF vectorizer for perfume descriptions.
        ohe (OneHotEncoder): One-hot encoder for categorical features (brand, gender).
        mlb_accords (MultiLabelBinarizer): Encoder for perfume accords (multi-label).
        mlb_notes (MultiLabelBinarizer): Encoder for perfume notes (multi-label).
    """

    def __init__(self):
        self.tfidf = None
        self.ohe = None
        self.mlb_accords = None
        self.mlb_notes = None

    def fit(self, perfumes):
        """
        Fits all internal vectorizers (TF-IDF, OHE, and MLBs) on a list of perfumes.

        Args:
            perfumes (Iterable[Perfume]): A list of Django ORM perfume instances.
        """
        # descriptions = [p.description or "" for p in perfumes]
        categorical = [[p.brand.name if p.brand else "missing",
                        p.gender or "unknown"] for p in perfumes]
        accords = [[a.accord.name for a in p.perfumeaccord_set.all()]
                   for p in perfumes]
        notes = [[n.note.name for n in p.perfumenote_set.all()]
                 for p in perfumes]

        # self.tfidf = TfidfVectorizer(max_features=500)
        # self.tfidf.fit(descriptions)

        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.ohe.fit(categorical)

        self.mlb_accords = MultiLabelBinarizer()
        self.mlb_accords.fit(accords)

        self.mlb_notes = MultiLabelBinarizer()
        self.mlb_notes.fit(notes)

    def transform(self, perfume):
        """
        Transforms a single perfume instance into a numerical feature vector.

        Args:
            perfume (Perfume): A single Django ORM perfume instance.

        Returns:
            np.ndarray: A 1D array representing the concatenated feature vector.
        """
        # desc = self.tfidf.transform([perfume.description or ""]).toarray()[0]
        cat = self.ohe.transform(
            [[perfume.brand.name if perfume.brand else "missing",
              perfume.gender or "unknown"]]
        ).toarray()[0]

        accords = [a.accord.name for a in perfume.perfumeaccord_set.all()]
        notes = [n.note.name for n in perfume.perfumenote_set.all()]
        acc_vec = self.mlb_accords.transform([accords])[0]
        note_vec = self.mlb_notes.transform([notes])[0]

        return np.concatenate([cat, acc_vec, note_vec])

    def fit_transform(self, perfumes):
        """
        Fits vectorizers and transforms the input perfumes into feature vectors.

        Args:
            perfumes (Iterable[Perfume]): A list of Django ORM perfume instances.

        Returns:
            np.ndarray: A 2D array of shape (n_perfumes, n_features).
        """
        self.fit(perfumes)
        return np.array([self.transform(p) for p in perfumes])

    def save(self, path):
        """
        Saves the trained vectorizer to disk using pickle.

        Args:
            path (str): Path to the output `.pkl` file.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        """
        Loads a saved vectorizer from disk.

        Args:
            path (str): Path to the `.pkl` file to load.
        """
        with open(path, "rb") as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
