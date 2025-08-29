import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import joypy

#load data
with open("../Metrics/wordnet_similarity_summary.json", "r") as f:
    data = json.load(f)

rows = []
for model, settings in data.items():
    for setting, score in settings.items():
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

#local vs prop
exclude_keywords = ["gpt", "gemini", "grok"]
df["Type"] = df["Model"].apply(
    lambda x: "Proprietary" if any(k in x.lower() for k in exclude_keywords) else "Local"
)

# order by mean
model_order = df.groupby("Model")["Score"].mean().sort_values(ascending=True).index

plt.figure(figsize=(10, 6))
joypy.joyplot(
    data=df, by="Model", column="Score",
    colormap=plt.cm.Set2, x_range=[0,1],
    figsize=(10,6), kind="normalized_counts"
)
plt.title("Distribution of WordNet Similarity Scores by Model (Ridgeline)", fontsize=16, weight="bold")
plt.xlabel("Similarity Score")

plt.savefig("avg_wordnet_similarity_ridgeline.pdf", bbox_inches="tight")
plt.close()

