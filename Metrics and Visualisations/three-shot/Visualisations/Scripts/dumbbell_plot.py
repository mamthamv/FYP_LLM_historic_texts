import json
import numpy as np
import matplotlib.pyplot as plt

# load data
with open("../Metrics/all_metrics_scores_para.json", "r") as f:
    data1 = json.load(f)
with open("../../para_level/Metrics/all_metrics_scores.json", "r") as f:
    data2 = json.load(f)


models = sorted(list(set(data1.keys()) & set(data2.keys())))
labels = [model.split('_')[0] for model in models]
metrics = ['bleu', 'meteor', 'cosine_similarity']
colors = ['red', 'blue', 'green']

plt.figure(figsize=(12, 8))

# plot each metric for each model
y_pos = 0
for model in models:
    for j, metric in enumerate(metrics):
       
        score_3shot = np.mean([config[metric] for config in data1[model].values()])
        score_0shot = np.mean([config[metric] for config in data2[model].values()])
        
        y = y_pos + j * 0.25
        
        # line connecting dots
        plt.plot([score_3shot, score_0shot], [y, y], 'gray', linewidth=2, alpha=0.6)
        
        # draw dots
        plt.scatter(score_3shot, y, color=colors[j], s=120, label='3-Shot' if model == models[0] and j == 0 else "")
        plt.scatter(score_0shot, y, color=colors[j], s=120, marker='s', edgecolor='black', label='Zero-Shot' if model == models[0] and j == 0 else "")
        
        # add scores as text - position based on which is higher
        if score_3shot < score_0shot:
            plt.text(score_3shot - 0.02, y, f'{score_3shot:.3f}', ha='right', va='center', fontweight='bold', fontsize=13)
            plt.text(score_0shot + 0.02, y, f'{score_0shot:.3f}', ha='left', va='center', fontweight='bold', fontsize=13)
        else:
            plt.text(score_3shot + 0.02, y, f'{score_3shot:.3f}', ha='left', va='center', fontweight='bold', fontsize=13)
            plt.text(score_0shot - 0.02, y, f'{score_0shot:.3f}', ha='right', va='center', fontweight='bold', fontsize=13)
    
    y_pos += 1

plt.yticks([i + 0.25 for i in range(len(models))], labels, fontsize=12, fontweight="bold")
plt.xlabel('Score', fontsize=14, fontweight='bold')
plt.title('3-Shot vs Zero-Shot Performance', fontsize=14, fontweight='bold')
plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.1, 0.1),fontsize=12,fontweight="bold")
plt.grid(axis='x', alpha=0.3)

# legend with both metrics and shot types
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [Patch(facecolor='red', label='BLEU'),
                  Patch(facecolor='blue', label='METEOR'), 
                  Patch(facecolor='green', label='Cosine Sim'),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='3-Shot'),
                  Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label='Zero-Shot')]
plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig("3shot vs 0shot_para.pdf", bbox_inches="tight")