import json, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

json_files = [
    "phrase_level/Metrics/bert_scores.json",
    "para_level/Metrics/bert_scores.json",
    "synthetic_dataset/Metrics/bert_scores.json"
]

metrics_cols = ["Precision", "Recall", "F1"]

# read all files
all_models_set = set()
dfs = []

for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    param_re = re.compile(r"_T(?P<T>[\d.]+)")
    rows = []
    for dataset, configs in raw.items():
        for cfg, metrics in configs.items():
            m = param_re.search(cfg)
            temp = float(m.group("T")) if m else None
            model = cfg.split("_T")[0].split("/")[-1]
            rows.append({
                "dataset": dataset,
                "model": model,
                "temp": temp,
                **metrics
            })
            all_models_set.add(model)
    
    df = pd.DataFrame(rows)
    for m in metrics_cols:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    dfs.append(df)


all_models = sorted(all_models_set)
cmap = plt.get_cmap("tab20")
model_colors = {model: cmap(i) for i, model in enumerate(all_models)}

# radar chart
angles = np.linspace(0, 2 * np.pi, len(metrics_cols), endpoint=False).tolist()
angles += angles[:1]

for idx, df in enumerate(dfs):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, polar=True)
    
    for model in all_models:
        if model in df["model"].values:
            vals = df[df["model"] == model][metrics_cols].mean().tolist()
            vals += vals[:1]
            ax.plot(angles, vals, label=model, marker="o", color=model_colors[model], linewidth=2)
            ax.fill(angles, vals, alpha=0.1, color=model_colors[model])
    
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_cols)
    ax.set_ylim(0, 1.0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"bert_radar_chart_{idx+1}.pdf", bbox_inches="tight")
    plt.show()
