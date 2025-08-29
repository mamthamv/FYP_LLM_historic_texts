import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def heatmap_top_models(bert_json_path, top_n=4):
    # Load JSON
    with open(bert_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    param_re = re.compile(r"_T(?P<T>[\d.]+)")
    for dataset, configs in raw.items():
        for cfg, metrics in configs.items():
            m = param_re.search(cfg)
            temp = float(m.group("T")) if m else None
            model = cfg.split("_T")[0].split("/")[-1]
            rows.append({
                "dataset": dataset.split("/")[-1],  # shorten filename
                "model": model,
                "temp": temp,
                **metrics
            })

    df = pd.DataFrame(rows)

    # Rank by F1 score (highest first)
    df_top = df.sort_values(by="F1", ascending=False).head(top_n)

    # Melt into long form: each metric is a separate row
    df_melt = df_top.melt(
        id_vars=["model", "temp"],
        value_vars=["Precision", "Recall", "F1"],
        var_name="metric",
        value_name="score"
    )

    # Pivot so each row = model+config, each column = metric
    df_pivot = df_melt.pivot_table(
        index=df_melt["model"] + "_T" + df_melt["temp"].astype(str), 
        columns="metric", 
        values="score"
    )

    # Plot heatmap with model names
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df_pivot,
        annot=True, fmt=".3f", cmap="viridis",
        cbar_kws={"label": "Score"},vmin=0, vmax=1,
        xticklabels=True, yticklabels=True,
         annot_kws={"size": 16} 
    )
    plt.xlabel("Metrics", fontsize=14, weight="bold")
    plt.ylabel("Models", fontsize=14, weight="bold")
    plt.title(f"Gemini Model - Paragraph level",fontsize=14, weight="bold")
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    

    plt.savefig("bert_scores_heatmap_para.png", dpi=200)

    return df, df_pivot

bert_json_path = "../../para_level/Metrics/bert_scores.json"
heatmap_top_models(bert_json_path)