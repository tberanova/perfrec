# Perfrec – Perfume Recommender System

**Perfrec** (PERFume RECommender) is a proof‑of‑concept web application that helps fragrance lovers discover scents they are likely to enjoy.  It couples a classic Django stack (Models → Views → Templates) with a **modular recommender engine** that blends collaborative filtering, content similarity and a lightweight *contextual boosting* layer so users can nudge results for a given season, occasion or mood.

**Note:** The provided dataset is minimal and synthetic to enable the testing of basic functionality. Contextual boosting works only with the provided chart data, which is only the SEASON data.

### Requirements

- Python 3.10.11

---

## 1 · Quick‑Start


```bash
# create & activate virtual‑env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# set up database & initial data
cd ./perfrec_project
python manage.py migrate
python manage.py import_perfumes ../data/perfumes.csv   # sample CSV – adjust path
python manage.py warm_recommender                    # pre‑compute caches (optional)

# run dev server
python manage.py runserver
```

Environment variables (e.g. `DJANGO_SECRET_KEY`) can live in a local `.env`.

### Custom Management Commands (CLI)

| Command | Purpose |
|---------|---------|
| `import_perfumes` | Load CSV of perfumes into DB. |
| `evaluate_models` | Offline evaluation across metrics / grids. |
| `warm_recommender` | Pre-fit model to reduce first-load latency. |
| `clear_serialized` | Delete stale pickled models / vectors. |

---

## 2 · Repository Layout
All classes, modules, and methods are thoroughly documented in-line. This README serves as a high-level overview of the architecture and entry points.

```text
perfrec_project/main              
├── config.py                     # Central switches & constants
├── urls.py                       # Root URLConf
├── models.py                     # Core ORM models (Perfume, Brand, Profile …)
├── forms.py                      # Auth & profile forms
├── views/                        # Request handlers (≅ controllers)
│   ├── main_page.py              # Homepage & recommendations
│   ├── perfume_detail.py         # Item page with similar perfumes
│   ├── perfume_browser.py        # Faceted catalogue browser
│   ├── interactions.py           # Like / unlike, feedback capture
│   ├── search.py                 # Full‑text search & autocomplete
│   └── authentication.py         # Login / registration flow
├── templates/                    # Django templates (Bootstrap 5)
├── recommender/                  # Modular recommendation subsystem
│   ├── manager.py                # Orchestrates algorithms & boosting
│   ├── manager_singleton.py      # Lazy singleton wrapper
│   ├── algos/                    # Individual algorithms
│   │   ├── base.py               # Shared interface
│   │   ├── popularity.py         # Most‑liked baseline
│   │   ├── i_knn.py              # Item‑based k‑NN
│   │   ├── u_knn.py              # User‑based k‑NN
│   │   ├── similarity.py         # Fast cosine / Jaccard content-based recommender
│   │   ├── ease.py               # EASE linear model
│   │   ├── bpr.py                # Bayesian Personalised Ranking
│   │   ├── elsa.py               # Lightweight autoencoder (ELSA)
│   │   ├── sae.py                # Sparse Autoencoder (SAE)
│   │   └── elsa_sae.py           # ELSA with embedded TopK‑SAE
│   ├── utils/                    # Helper modules
│   │   ├── data_loader.py        # Interaction matrix builders
│   │   ├── vectorizer.py         # Perfume embeddings
│   │   ├── vectorizer_io.py      # Serialize / load embeddings
│   │   ├── user_matrix.py        # CSR utils
│   │   ├── boosting.py           # Contextual re‑weighting
│   │   └── neuron_tagging.py     # Explainability – latent neuron labels
│   └── serialized/               # *.pkl weights & embeddings
├── management/commands/          # Custom CLI tasks
│   ├── import_perfumes.py        # Load CSV ➜ DB 
│   ├── evaluate_models.py        # Offline evaluation pipeline
│   ├── warm_recommender.py       # Pre‑fit models
│   ├── clear_serialized.py       # Purge stale artefacts
│   └── utils/
│       ├── eval_config.py        # Default eval and grid search setups
│       └── plot.py               # Matplotlib result charts
├── migrations/                   # Django migration history
├── manage.py                     # Django entrypoint
├── requirements.txt              # Dependency pinning
└── README.md                     # → you are here 📝
```

---

## 3 · Key Modules & Interfaces

| Area                          | File(s)                               | Why you care                                                                                       |
| ----------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Recommender orchestration** | `recommender/manager.py`              | Singleton that keeps models in RAM, routes calls, applies contextual boosting.                     |
| **Algorithm interface**       | `recommender/algos/base.py`           | Defines `fit`, `recommend()` and `save/load` so any new model is plug‑and‑play. |
| **Contextual Boosting**       | `recommender/utils/boosting.py`       | User can tick *evening*, *summer*, *fruity* ➜ re‑weights raw scores on‑the‑fly.                    |
| **Explainability**            | `recommender/utils/neuron_tagging.py` | Labels top‑activated latent neurons with human tags (e.g. *vanilla*, *woody*).                     |
| **CLI utilities**             | `management/commands/*`               | Reproducible data import, evaluation & cache‑warming.                                              |

---
## 4 · Core Django Views & Endpoints

| URL name / path\* | HTTP | View / Function | Auth? | Purpose |
|-------------------|------|-----------------|-------|---------|
| `main_page` (`/`) | GET/POST | `main_page()` | optional | Homepage with popular / personalised recs; POST = AJAX filter update. |
| `perfume_browser` (`/browse/`) | GET | `PerfumeListView` | open | Paginated catalogue with brand/accord filters & full-text search.|
| `perfume_detail` (`/p/<slug>/`) | GET | `PerfumeDetailView` | open | Perfume detail page + “similar” and “also liked” suggestions. |
| `search_suggestions` (`/api/search/`) | GET | `search_suggestions()` | open | JSON autocomplete for perfume names / brands. |
| `add_liked_perfume` (`/api/like/`) | POST | `add_liked_perfume()` | login | Add perfume to user’s likes (updates model + recommender). |
| `remove_liked_perfume` (`/api/unlike/<id>/`) | POST | `remove_liked_perfume()` | login | Remove perfume from likes. |
| `login` (`/login/`) | GET/POST | `CustomLoginView` | open | Login page (redirects if already authed).|
| `register` (`/register/`) | GET/POST | `register()` | open | User sign-up form. |
| `profile` (`/profile/`) | GET/POST | `account_profile()` | login | View / edit profile & preferences.|

### Signal Handlers

| Signal | Receiver | What it does |
|--------|----------|--------------|
| `post_save(User)` | `create_user_profile` | Creates `Profile` + adds empty row to recommender. |
| `post_save(User)` | `save_user_profile` | Persists profile changes on every user save. |

---
## 5 · Extending the Project

1. **Add a new algorithm**

   * Drop implementation in `recommender/algos/` inheriting `BaseRecommender`.
   * Register it in `recommender/algos/__init__.py`.
   * Enable via `config.py`.

---
