import json
import pandas as pd
from pathlib import Path

# === Paths ===
root = Path(__file__).resolve().parents[4]
input_path = root / "results" / "nested_eval_summary.json"
output_acc = root / "results" / "results_accuracy.tex"
output_beyond = root / "results" / "results_beyond_accuracy.tex"

# === Load JSON ===
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Metric definitions ===
accuracy_metrics = [
    ("precision@15", "Precision@15"),
    ("recall@15", "Recall@15"),
    ("hits@15", "Hits@15"),
    ("ndcg@15", "nDCG@15"),
]

beyond_metrics = [
    ("ild", "ILD"),
    ("poplift", "PopLift"),
    ("fit_time", "Fit Time (s)"),
    ("rec_time", "Rec Time (s)")
]

std_keys = {
    "precision@15": "std_precision@15",
    "recall@15": "std_recall@15",
    "hits@15": "std_hits@15",
    "ndcg@15": "std_ndcg@15"
}

# === Format metric values ===


def combine(mean, std):
    return f"{mean:.3f} \\pm {std:.3f}"


def build_accuracy_table():
    rows = []
    for entry in data:
        row = {"Algorithm": entry["algorithm"]}
        for key, label in accuracy_metrics:
            row[label.replace("@15", "@$15$")
                ] = combine(entry[key], entry[std_keys[key]])
        rows.append(row)
    return pd.DataFrame(rows)


def build_beyond_table():
    rows = []
    for entry in data:
        row = {"Algorithm": entry["algorithm"]}
        for key, label in beyond_metrics:
            row[label] = f"{entry[key]:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


df_acc = build_accuracy_table()
df_beyond = build_beyond_table()

# === Bold best values (only for accuracy) ===


def bold_best(df, columns, reverse=False):
    for col in columns:
        values = df[col].str.extract(r"^([0-9.]+)").astype(float)
        best = values[0].min() if reverse else values[0].max()
        for i in df.index:
            val = float(values.loc[i, 0])
            if abs(val - best) < 1e-6:
                df.at[i, col] = f"\\textbf{{{df.at[i, col]}}}"


bold_best(df_acc, df_acc.columns[1:])  # Higher is better

# === Convert DataFrame to LaTeX table ===


def df_to_latex(df, caption, label):
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{" + "l" + "r" * (len(df.columns) - 1) + "}",
        "\\toprule",
        " & ".join([f"\\textbf{{{c}}}" for c in df.columns]) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(cell) for cell in row) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\label{{{label}}}",
        "\\end{table}"
    ]
    return "\n".join(lines)


# === Save LaTeX tables ===
with open(output_acc, "w", encoding="utf-8") as f:
    f.write(df_to_latex(
        df_acc, "Accuracy-based metrics (mean $\\pm$ std) for each algorithm", "tab:results_accuracy"))

with open(output_beyond, "w", encoding="utf-8") as f:
    f.write(df_to_latex(df_beyond, "Beyond-accuracy and efficiency metrics (mean values)",
            "tab:results_beyond_accuracy"))

print("✅ LaTeX tables saved:")
print(f"  - {output_acc.name}")
print(f"  - {output_beyond.name}")
