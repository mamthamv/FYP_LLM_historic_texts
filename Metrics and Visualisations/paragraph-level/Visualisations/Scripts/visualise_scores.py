import json, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load data

# bert scores dataframe
bert_json_path = "../Metrics/bert_scores.json"
with open(bert_json_path, "r", encoding="utf-8") as f:
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
df_bert = pd.DataFrame(rows)
metrics_cols = ["Precision", "Recall", "F1"]
for m in metrics_cols:
    df_bert[m] = pd.to_numeric(df_bert[m], errors="coerce")

# all the other metrics dataframe
metrics_json_path = "../Metrics/all_metrics_scores.json"
with open(metrics_json_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

rows = []
for dataset, configs in raw.items():
    for cfg, metrics in configs.items():
        m = param_re.search(cfg)
        temp = float(m.group("T")) if m else None
        model = cfg.split("_T")[0].split("/")[-1]
        model = model.replace(".gguf", "")
        LLM_type = "Local" if any(x in dataset.lower() for x in ["unsloth","gguf","llama","qwen","phi","gemma"]) else "Proprietary"
        rows.append({
            "dataset": dataset,
            "model": model,
            "temp": temp,
            "llm_type": LLM_type,
            **metrics
        })

df_metrics = pd.DataFrame(rows)
metrics_list = ["BLEU", "METEOR", "SentenceBert"]
json_keys = ["bleu", "meteor", "cosine_similarity"] 
for key in json_keys:
    df_metrics[key] = pd.to_numeric(df_metrics[key], errors="coerce")

#plot settings
plt.rcParams.update({
    "font.size": 14,        
    "axes.titlesize": 18,   
    "axes.labelsize": 16,  
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,  
    "legend.fontsize": 14,  
})

# radar chart
metrics_cols = ["Precision", "Recall", "F1"]
all_models = df_bert["model"].unique()
angles = np.linspace(0, 2*np.pi, len(metrics_cols), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, polar=True)

cmap = plt.get_cmap("tab20")
colors = [cmap(i) for i in range(len(all_models))]

for i, model in enumerate(all_models):
    vals = df_bert[df_bert["model"] == model][metrics_cols].mean().tolist()
    vals += vals[:1]
    ax.plot(angles, vals, label=model, marker="o", color=colors[i], linewidth=2)
    ax.fill(angles, vals, alpha=0.1, color=colors[i])

ax.set_thetagrids(np.degrees(angles[:-1]), metrics_cols, fontsize=16)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
ax.legend(bbox_to_anchor=(1.15, 1.05), loc="upper left", fontsize=14)
plt.tight_layout()
plt.savefig("bert_radar_all_models.pdf", bbox_inches="tight")
plt.show()

#average score across all models

#data preparation
metrics_list = ["BLEU", "METEOR", "SentenceBert"]
json_keys = ["bleu", "meteor", "cosine_similarity"]
colors = ["#4C72B0", "#DD8452", "#55A868"]  # Blue, orange, green

# Aggregate data
agg = df_metrics.groupby("model")[json_keys].agg(['mean', 'sem'])
agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
agg = agg.reset_index()
agg = agg.sort_values("cosine_similarity_mean", ascending=True).reset_index(drop=True)

n_models = len(agg)
bar_width = 0.3 
group_spacing = 1.4  

fig, ax = plt.subplots(figsize=(10, max(10, n_models * 0.9)))
y_base = np.arange(n_models) * group_spacing

# Plot each metric
for i, (metric, label, color) in enumerate(zip(json_keys, metrics_list, colors)):
    means = agg[f"{metric}_mean"].values
    sems = agg[f"{metric}_sem"].values
    y_pos = y_base + (i - 1) * bar_width

    bars = ax.barh(
        y_pos, means,
        height=bar_width,
        xerr=sems,
        color=color,
        edgecolor='white',
        linewidth=0.5,
        label=label,
        capsize=3,
        alpha=0.85
    )

    # Add value annotations
    for bar, mean, sem in zip(bars, means, sems):
        text_x = mean + sem + 0.02
        if text_x < 0.05:
            text_x = 0.05
        ax.text(
            text_x,
            bar.get_y() + bar.get_height()/2,
            f"{mean:.3f} ± {sem:.3f}",
            va='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
        )

ax.set_yticks(y_base)
ax.set_yticklabels(agg["model"], fontsize=12, fontweight='bold')

ax.set_xlabel("Score Value (mean ± SE)", fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.1)
ax.set_title("Average Translation Metrics by Model", fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()

ax.xaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)

ax.legend(loc='upper right', frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.0))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xticks(np.arange(0, 1.1, 0.1))

plt.tight_layout()
plt.savefig("translation_metrics_comparison.pdf", bbox_inches="tight", dpi=300)
plt.show()


# Average by LLM type + SEM
avg_by_llm = df_metrics.groupby("llm_type")[json_keys].mean().reset_index()
sem_by_llm = df_metrics.groupby("llm_type")[json_keys].sem().reset_index()


for key in json_keys:
    avg_by_llm[key] = pd.to_numeric(avg_by_llm[key], errors="coerce")
    sem_by_llm[key] = pd.to_numeric(sem_by_llm[key], errors="coerce")

x = np.arange(len(avg_by_llm["llm_type"]))
width = 0.25
fig, ax = plt.subplots(figsize=(10,6))

# Plot with error bars
for i, (label, key, color) in enumerate(zip(metrics_list, json_keys, plt.cm.Set2.colors)):
    bars = ax.bar(
        x + i*width - width,
        avg_by_llm[key],
        width,
        yerr=sem_by_llm[key],       
        capsize=5,               
        label=label,
        color=color,
        alpha=0.9,
        edgecolor="black"
    )
    # Add bar labels
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.025,
                f'{h:.2f}', ha='center', va='bottom', fontsize=14, fontweight="bold")


ax.set_xticks(x)
ax.set_xticklabels(avg_by_llm["llm_type"], fontsize=14, fontweight="bold")
ax.set_ylabel("Average Score", fontsize=16, fontweight="bold", labelpad=5)
ax.set_ylim(0, 1.05)
ax.set_title("Average Translation Metrics by LLM Type", fontsize=18, fontweight="bold", pad=15)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=14, loc="upper left", frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig("avg_llm_type.pdf", bbox_inches="tight")
plt.show()

