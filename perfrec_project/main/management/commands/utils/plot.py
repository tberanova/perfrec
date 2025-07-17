import pandas as pd
from django.core.management.base import BaseCommand
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


def plot_metrics(df, top_k):
    metrics = [f"precision@{top_k}", f"recall@{top_k}",
               f"hits@{top_k}", f"ndcg@{top_k}"]
    std_metrics = [f"std_precision@{top_k}", f"std_recall@{top_k}",
                   f"std_hits@{top_k}", f"std_ndcg@{top_k}"]
    titles = ["Precision", "Recall", "Hits", "nDCG"]

    algorithms = df["algorithm"].unique()
    cmap = cm.get_cmap("Set2", len(algorithms))
    color_map = {algo: mcolors.to_hex(cmap(i))
                 for i, algo in enumerate(algorithms)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (metric, std_metric) in enumerate(zip(metrics, std_metrics)):
        sorted_df = df.sort_values(metric, ascending=False)
        ax = axes[idx]
        x = np.arange(len(sorted_df))
        colors = [color_map[algo] for algo in sorted_df["algorithm"]]
        heights = sorted_df[metric].values
        errors = sorted_df[std_metric].values if std_metric in sorted_df.columns else None

        ax.bar(x, heights, yerr=errors, capsize=6, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_df["algorithm"], rotation=15)
        ax.set_title(titles[idx])
        ax.set_ylabel("Score")
        ax.grid(True, linestyle='--', alpha=0.5)

        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        ax_single.bar(x, heights, yerr=errors, capsize=6, color=colors)
        ax_single.set_xticks(x)
        ax_single.set_xticklabels(sorted_df["algorithm"], rotation=15)
        ax_single.set_title(f"{titles[idx]} @ {top_k}")
        ax_single.set_ylabel("Score")
        ax_single.grid(True, linestyle='--', alpha=0.5)
        fig_single.tight_layout()
        fig_single.savefig(f"results/plots/{metric}.png")
        plt.close(fig_single)

    plt.suptitle(f"Evaluation Metrics @ {top_k}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/plots/4x_metrics_grid.png")
    plt.show()

    # --- ILD Plot ---
    if "ild" in df.columns:
        fig_ild, ax_ild = plt.subplots(figsize=(8, 6))
        sorted_df = df.sort_values("ild", ascending=False)
        x = np.arange(len(sorted_df))
        colors = [color_map[algo] for algo in sorted_df["algorithm"]]
        ax_ild.bar(x, sorted_df["ild"].values, color=colors)
        ax_ild.set_xticks(x)
        ax_ild.set_xticklabels(sorted_df["algorithm"], rotation=15)
        ax_ild.set_title(f"Intra-List Diversity (ILD) @ {top_k}")
        ax_ild.set_ylabel("Average Dissimilarity")
        ax_ild.grid(True, linestyle='--', alpha=0.5)
        fig_ild.tight_layout()
        fig_ild.savefig(f"results/plots/ild.png")
        plt.show()

    # --- PopLift Plot ---
    if "poplift" in df.columns:
        fig_pop, ax_pop = plt.subplots(figsize=(8, 6))
        sorted_df = df.sort_values("poplift", ascending=False)
        x = np.arange(len(sorted_df))
        colors = [color_map[algo] for algo in sorted_df["algorithm"]]
        ax_pop.bar(x, sorted_df["poplift"].values, color=colors)
        ax_pop.set_xticks(x)
        ax_pop.set_xticklabels(sorted_df["algorithm"], rotation=15)
        ax_pop.set_title(f"Popularity Lift (PopLift) @ {top_k}")
        ax_pop.set_ylabel("Relative Popularity Increase")
        ax_pop.grid(True, linestyle='--', alpha=0.5)
        fig_pop.tight_layout()
        fig_pop.savefig(f"results/plots/poplift.png")
        plt.show()

        # --- Fit Time Plot ---
    if "fit_time" in df.columns:
        fig_fit, ax_fit = plt.subplots(figsize=(8, 6))
        sorted_df = df.sort_values("fit_time", ascending=False)
        x = np.arange(len(sorted_df))
        colors = [color_map[algo] for algo in sorted_df["algorithm"]]
        fit_values = sorted_df["fit_time"].values

        ax_fit.bar(x, fit_values, color=colors)
        ax_fit.set_yscale("log")  # <--- HERE
        ax_fit.set_xticks(x)
        ax_fit.set_xticklabels(sorted_df["algorithm"], rotation=15)
        ax_fit.set_title("Model Fit Time (log-scale seconds)")
        ax_fit.set_ylabel("Seconds (log scale)")
        ax_fit.grid(True, which='both', linestyle='--', alpha=0.5)
        fig_fit.tight_layout()
        fig_fit.savefig("results/plots/fit_time.png")
        plt.show()

    # --- Recommend Time Plot ---
    if "rec_time" in df.columns:
        fig_rec, ax_rec = plt.subplots(figsize=(8, 6))
        sorted_df = df.sort_values("rec_time", ascending=False)
        x = np.arange(len(sorted_df))
        colors = [color_map[algo] for algo in sorted_df["algorithm"]]
        ax_rec.bar(x, sorted_df["rec_time"].values, color=colors)
        ax_rec.set_xticks(x)
        ax_rec.set_xticklabels(sorted_df["algorithm"], rotation=15)
        ax_rec.set_title("Recommendation Time (seconds)")
        ax_rec.set_ylabel("Seconds")
        ax_rec.grid(True, linestyle='--', alpha=0.5)
        fig_rec.tight_layout()
        fig_rec.savefig("results/plots/rec_time.png")
        plt.show()


class Command(BaseCommand):
    help = "Plot evaluation metrics from a JSON file"

    def add_arguments(self, parser):
        parser.add_argument(
            "--file", type=str, required=True,
            help="Path to the JSON file containing metrics (e.g. 'results/comparison_results.json')"
        )
        parser.add_argument(
            "--top_k", type=int, default=20,
            help="Top-K value used in the evaluation"
        )

    def handle(self, *args, **options):
        path = options["file"]
        top_k = options["top_k"]

        try:
            df = pd.read_json(path)
        except Exception as e:
            self.stderr.write(f"[ERROR] Failed to load JSON: {e}")
            return

        print(f"[INFO] Plotting metrics from {path} with top_k={top_k}")
        plot_metrics(df, top_k)
