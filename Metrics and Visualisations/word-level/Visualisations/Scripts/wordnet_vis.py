import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

with open("../Metrics/wordnet_similarity_summary.json", "r") as f:
    data = json.load(f)

rows = []
for model, settings in data.items():
    for setting, score in settings.items():
        # Extract temperature
        match = re.match(r"(temp_\d+(?:\.\d+)?)(.*)", setting)
        if match:
            base_temp = float(match.group(1).replace("temp_", ""))
        else:
            base_temp = None
        rows.append({
            "Model": model,
            "Setting": setting,
            "BaseTemp": base_temp,
            "Score": score 
        })

df = pd.DataFrame(rows)

# average score plot
exclude_keywords = ["gpt", "gemini", "grok"]
df["Type"] = df["Model"].apply(
    lambda x: "Proprietary" if any(k in x.lower() for k in exclude_keywords) else "Local"
)
#mean score per model
model_order = df.groupby("Model")["Score"].mean().sort_values(ascending=True).index

# Plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Horizontal barplot with error bars
ax = sns.barplot(
    data=df,
    y="Model", x="Score",
    hue="Type",
    dodge=False,
    palette={"Local":"steelblue", "Proprietary":"orange"},
    order=model_order,
    errorbar=("se", 1.96)
)


# Titles and labels
plt.xlabel("Average Similarity Score")
plt.ylabel("")
plt.title("Average WordNet Similarity Score by Model", fontsize=16, weight="bold")
plt.xlim(0, 1)
plt.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("avg_wordnet_similarity.pdf", bbox_inches="tight")
plt.close()


# temp vs local models
exclude_keywords = ["grok", "gemini", "gpt", "gemma"]
local_df = df[~df["Model"].str.lower().str.contains("|".join(exclude_keywords))]

plt.figure(figsize=(10, 5))
sns.lineplot(
    data=local_df,
    x="BaseTemp", y="Score",
    hue="Model", marker="o",
    errorbar=("se", 1.96)  
)
plt.title("Temperature vs WordNet Similarity (Locally Deployed Models Only)", fontsize=16, weight="bold")
plt.ylabel("Similarity Score")
plt.xlabel("Temperature")
plt.ylim(0, 1)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("temp_vs_score_localmodels.pdf", bbox_inches="tight")
plt.close()

