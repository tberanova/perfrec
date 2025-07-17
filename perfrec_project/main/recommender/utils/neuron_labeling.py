"""
neuron_tagging.py

This module defines the utility function `build_neuron_tag_dict`, which analyzes latent neuron
activations in a sparse autoencoder (SAE) and maps them to interpretable semantic tags based on
perfume metadata (e.g., gender, brand, accords, notes, seasonal/occasion charts).

The function saves a JSON file containing an IDF-weighted summary of the most relevant tags
for each neuron, which can be used to interpret and label latent dimensions.

In addition to saving the summary, the function returns the full raw neuron-to-tag activation
dictionary, useful for further custom analysis or debugging.
"""

import json
import math
from collections import defaultdict

import numpy as np
import torch


def build_neuron_tag_dict(model, matrix_id_to_orm_perfume: dict,
                          save_path: str = "results/neuron_tag_dict.json") -> dict:
    """
    Computes and saves a tag-based explanation mapping for neurons in a trained SAE model.

    For each perfume, the function:
    - Extracts metadata tags (e.g., brand, gender, notes, top seasonal/occasion labels)
    - Runs the perfume as input through the ELSA → SAE encoder pipeline
    - Accumulates each tag's activation across all neurons
    - Computes IDF weights per tag (ignores tags with document frequency < 5)
    - Aggregates the top-5 weighted tags for each neuron into a summary dictionary

    The summary is saved to disk (as JSON) and the full raw activation dictionary is returned.

    Parameters:
    - model (object): A model with `.elsa`, `.sae`, `.device`, and `.num_items` attributes.
    - matrix_id_to_orm_perfume (dict): Maps matrix row indices to ORM Perfume objects.
    - save_path (str): Path to write the final IDF-weighted summary JSON.

    Returns:
    - dict: Raw neuron-to-tag activation structure (neuron_id → tag → list of activations).
    """

    print("[INFO] Mapping neurons to tags with IDF normalization")

    # Raw accumulation: neuron_id - tag - list of activation values
    neuron_tag_dict = defaultdict(lambda: defaultdict(list))
    tag_document_frequency = defaultdict(int)

    input_dim = model.num_items
    total_docs = 0

    # Iterate over all perfumes (indexed by their matrix ID)
    for matrix_id, perfume in matrix_id_to_orm_perfume.items():
        total_docs += 1

        # One-hot perfume vector (as user input simulation)
        multihot = np.zeros(input_dim, dtype=np.float32)
        multihot[matrix_id] = 1.0
        input_tensor = torch.Tensor(multihot).unsqueeze(0).to(model.device)

        # Run through ELSA -> SAE to get latent activation
        with torch.no_grad():
            elsa_emb = model.elsa.encode(input_tensor)
            sae_emb = model.sae(elsa_emb)[0].squeeze(0)

        # --- Extract all relevant semantic tags from this perfume ---
        tags = set()

        if perfume.brand:
            tags.add(perfume.brand.name.strip().lower())

        # if perfume.gender:
        #     tags.add(perfume.gender.strip().lower())

        for acc in perfume.perfumeaccord_set.all():
            tags.add(acc.accord.name.strip().lower())

        for note in perfume.perfumenote_set.all():
            raw = note.note.name.strip().lower()
            tags.add(raw)          # add full note also
            tags.update(raw.split())  # tokenize multiword notes

        # if perfume.season_chart:
        #     top_season = max(perfume.season_chart,
        #                      key=perfume.season_chart.get)
        #     tags.add(top_season.strip().lower())

        # if perfume.occasion_chart:
        #     top_occasion = max(perfume.occasion_chart,
        #                        key=perfume.occasion_chart.get)
        #     tags.add(top_occasion.strip().lower())

        if perfume.type_chart:
            tags.update(label.strip().lower() for label in perfume.type_chart)

        if perfume.style_chart:
            tags.update(label.strip().lower() for label in perfume.style_chart)

        # Update tag document frequencies
        for tag in tags:
            tag_document_frequency[tag] += 1

        # Assign current activations to each tag
        for neuron_id, activation in enumerate(sae_emb.cpu().numpy()):
            for tag in tags:
                # neuron_tag_dict[neuron_id][tag].append(float(activation))
                if activation > 0:
                    neuron_tag_dict[neuron_id][tag].append(float(1))

    print("[INFO] Computing IDF and aggregating tag scores")

    # Compute IDF: zero out tags with low support (df < 5)
    tag_idf = {
        tag: math.log(total_docs / df) if df >= 20 else 0.0
        for tag, df in tag_document_frequency.items()
    }

    beta = 1.2  # add full note also  # dampening factor for IDF influence

    # Final IDF-weighted tag scores: neuron
    summed_dict = {
        str(neuron_id): {
            tag: round(sum(acts) * tag_idf.get(tag, 0.0), 6)
            for tag, acts in sorted(
                tag_map.items(),
                key=lambda item:
                (sum(item[1]) * (tag_idf.get(item[0], 0.0) ** beta)),
                reverse=True
            )[:5]  # top most activated tags per neuron
        }
        for neuron_id, tag_map in neuron_tag_dict.items()
    }

    # Save result to JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summed_dict, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved top IDF-weighted tags to {save_path}")
    return summed_dict
