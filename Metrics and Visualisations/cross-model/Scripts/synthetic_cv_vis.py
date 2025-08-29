import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# SCORES ACROSS MODELS
data = {
    "SentenceBert": {
        "gemini_mi_texts": {
            "gpt4o translation": 0.8137766718864441,
            "grok translation": 0.8106598854064941
        },
        "gpt4o_mi_texts": {
            "gemini translation": 0.8609020709991455,
            "grok translation": 0.8523596525192261
        },
        "grok_mi_texts": {
            "gpt4o translation": 0.873917281627655,
            "gemini translation": 0.8649353981018066
        }
    },
    "BLEU": {
        "gemini_mi_texts": {
            "gpt4o translation": 0.2879967973054449,
            "grok translation": 0.2950177064085543
        },
        "gpt4o_mi_texts": {
            "gemini translation": 0.369156619021112,
            "grok translation": 0.3971889113970245
        },
        "grok_mi_texts": {
            "gpt4o translation": 0.42766658430304727,
            "gemini translation": 0.47054079196413967
        }
    },
    "METEOR": {
        "gemini_mi_texts": {
            "gpt4o translation": 0.596242059796119,
            "grok translation": 0.614967852423572
        },
        "gpt4o_mi_texts": {
            "gemini translation":  0.6428144392202464,
            "grok translation": 0.6632303195829269
        },
        "grok_mi_texts": {
            "gpt4o translation":  0.7113629068954468,
            "gemini translation":  0.7439702643404916
        }
    }
}

# BERTScore F1 data
bertscore_f1 = {
    "Gemini → GPT-4o": 0.601922333240509,
    "Gemini → Grok": 0.6106029748916626,
    "GPT-4o → Gemini": 0.623598039150238,
    "GPT-4o → Grok": 0.6214948296546936,
    "Grok → GPT-4o": 0.6589946746826172,
    "Grok → Gemini": 0.6739703416824341
}

# Bigger font settings
plt.rcParams.update({
    "font.size": 18,        
    "axes.titlesize": 22,   
    "axes.labelsize": 20,  
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,  
    "legend.fontsize": 18,  
})

# Create figure
fig, ax = plt.subplots(figsize=(16, 9))

# PLOT 1: Grouped Bar Chart for Main Metrics
translation_pairs = []
sb_scores = []
bleu_scores = []
meteor_scores = []

for source_model in data['SentenceBert']:
    for target_model in data['SentenceBert'][source_model]:
        source = source_model.replace('_mi_texts', '').replace('gpt4o', 'GPT-4o').replace('gemini', 'Gemini').replace('grok', 'Grok').title()
        target = target_model.replace(' translation', '').replace('gpt4o', 'GPT-4o').replace('gemini', 'Gemini').replace('grok', 'Grok').title()
        pair = f"{source} → {target}"
        translation_pairs.append(pair)
        
        sb_scores.append(data['SentenceBert'][source_model][target_model])
        bleu_scores.append(data['BLEU'][source_model][target_model])
        meteor_scores.append(data['METEOR'][source_model][target_model])

x = np.arange(len(translation_pairs))
width = 0.25

bars1 = ax.bar(x - width, sb_scores, width, label='SentenceBERT', color='#66c2a5', alpha=0.8)
bars2 = ax.bar(x, bleu_scores, width, label='BLEU', color='#fc8d62', alpha=0.8)
bars3 = ax.bar(x + width, meteor_scores, width, label='METEOR', color='#8da0cb', alpha=0.8)

ax.set_xlabel('Translation Direction', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Translation Quality by Metric', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(translation_pairs, rotation=45, ha='right')
ax.set_ylim(0, 1)

# Legend outside
ax.legend(frameon=True, fancybox=True, shadow=True, loc="center left", bbox_to_anchor=(1, 0.5))

# Value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.025,
                f'{height:.2f}', ha='center', va='bottom', fontsize=18)

ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("overall_cross_model_results_metrics.pdf", bbox_inches="tight")


# PLOT 2: F1 BERTScore Comparison
fig, ax = plt.subplots(figsize=(14, 8))
pairs = list(bertscore_f1.keys())
scores = list(bertscore_f1.values())

# Create colors based on source model
colors = []
for pair in pairs:
    if pair.startswith('Grok'):
        colors.append('#e78ac3')
    elif pair.startswith('GPT-4o'):
        colors.append('#a6d854')
    else:  # Gemini
        colors.append('#ffd92f')

bars = ax.bar(range(len(pairs)), scores, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

ax.set_xlabel('Translation Direction',  fontweight='bold')
ax.set_ylabel('BERTScore F1', fontweight='bold')
ax.set_title('BERTScore F1 Performance',fontweight='bold', pad=20)
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels(pairs, rotation=45, ha='right')
ax.set_ylim(0, 1)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, scores)):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{score:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

ax.grid(True, alpha=0.3, axis='y')

# Add legend for source models
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e78ac3', alpha=0.8, label='Grok Source'),
    Patch(facecolor='#a6d854', alpha=0.8, label='GPT-4o Source'),
    Patch(facecolor='#ffd92f', alpha=0.8, label='Gemini Source')
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig("overall_cross_model_results_bert.pdf", bbox_inches="tight")


#  summary
print("\n" + "="*60)
print("   SYNTHETIC CROSS VALIDATION TRANSLATION QUALITY")
print("="*60)
print(f"{'Metric':<15} {'Best Score':<12} {'Translation Pair':<25}")
print("-"*60)

# Find best for each metric
metrics_data = [
    ('SentenceBERT', sb_scores, translation_pairs),
    ('BLEU', bleu_scores, translation_pairs),
    ('METEOR', meteor_scores, translation_pairs),
    ('BERTScore F1', list(bertscore_f1.values()), list(bertscore_f1.keys()))
]

for metric_name, scores, pairs in metrics_data:
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_pair = pairs[best_idx]
    print(f"{metric_name:<15} {best_score:<12.3f} {best_pair:<25}")
