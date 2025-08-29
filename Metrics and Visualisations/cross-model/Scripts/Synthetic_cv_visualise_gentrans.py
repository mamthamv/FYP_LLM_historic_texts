import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set clean, modern style
plt.style.use('seaborn-v0_8-whitegrid')

# data
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
    },
    "BERTScore": {
        "gpt4o_mi_texts": {
            "gemini translation": {
                "Precision": 0.5964449644088745,
                "Recall": 0.6541641354560852,
                "F1": 0.623598039150238
            },
            "grok translation": {
                "Precision": 0.5935048460960388,
                "Recall": 0.6529397368431091,
                "F1": 0.6214948296546936
            }
        },
        "gemini_mi_texts": {
            "gpt4o translation": {
                "Precision": 0.5821593403816223,
                "Recall": 0.6235329508781433,
                "F1": 0.601922333240509
            },
            "grok translation": {
                "Precision": 0.5908210873603821,
                "Recall": 0.6323604583740234,
                "F1": 0.6106029748916626
            }
        },
        "grok_mi_texts": {
            "gpt4o translation": {
                "Precision": 0.6370473504066467,
                "Recall": 0.6830205321311951,
                "F1": 0.6589946746826172
            },
            "gemini translation": {
                "Precision": 0.6545771360397339,
                "Recall": 0.6951387524604797,
                "F1": 0.6739703416824341
            }
        }
    }
}

# Calculate average scores for each model as generator and translator
models = ['Gemini', 'GPT-4o', 'Grok']
metrics = ['SentenceBert', 'BLEU', 'METEOR', 'BERTScore']

# Average quality when others translate their text
as_generator = {model: [] for model in models}
# Average quality when they translate others' text  
as_translator = {model: [] for model in models}

# Process metrics
for metric in metrics[:3]:
    for model in models:
        model_key = model.lower().replace('-', '') + '_mi_texts'
        
        # As Generator
        if model_key in data[metric]:
            generator_scores = list(data[metric][model_key].values())
            as_generator[model].append(np.mean(generator_scores))
        
        # As Translator
        translator_scores = []
        target_key = model.lower().replace('-', '') + ' translation'
        for source_key in data[metric]:
            if target_key in data[metric][source_key]:
                translator_scores.append(data[metric][source_key][target_key])
        if translator_scores:
            as_translator[model].append(np.mean(translator_scores))

# Add BERTScore
for model in models:
    model_key = model.lower().replace('-', '') + '_mi_texts'
    
    # As Generator
    if model_key in data["BERTScore"]:
        generator_scores = [v["F1"] for v in data["BERTScore"][model_key].values()]
        as_generator[model].append(np.mean(generator_scores))
    
    # As Translator
    translator_scores = []
    target_key = model.lower().replace('-', '') + ' translation'
    for source_key in data["BERTScore"]:
        if target_key in data["BERTScore"][source_key]:
            translator_scores.append(data["BERTScore"][source_key][target_key]["F1"])
    if translator_scores:
        as_translator[model].append(np.mean(translator_scores))

# Create the plot
fig, ax = plt.subplots(figsize=(16, 10))

# Calculate averages
generator_avg = [np.mean(as_generator[model]) for model in models]
translator_avg = [np.mean(as_translator[model]) for model in models]

x = np.arange(len(models))
width = 0.35

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars1 = ax.bar(x - width/2, generator_avg, width, label='EN→MI', 
               color=colors, alpha=0.8, edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, translator_avg, width, label='MI→EN', 
               color=colors, alpha=0.5, edgecolor='white', linewidth=2, hatch='///')


ax.set_xlabel('Proprietary Models', fontsize=20, fontweight='bold')
ax.set_ylabel('Average Quality Score', fontsize=20, fontweight='bold')
ax.set_title('Proprietary Models: EN→MI vs MI→EN Performance\n(Averaged across SentenceBERT, BLEU, METEOR, BERTScore)', 
             fontsize=24, fontweight='bold', pad=25)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=18, fontweight='bold')
ax.legend(fontsize=18, frameon=True, fancybox=True, shadow=True)

# Add value labels
for bars, values in [(bars1, generator_avg), (bars2, translator_avg)]:
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{value:.3f}', ha='center', va='bottom', 
                fontsize=20, fontweight='bold')

# Grid and limits
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0,1)
ax.tick_params(axis='both', which='major', labelsize=18)

for i, (gen, trans) in enumerate(zip(generator_avg, translator_avg)):
    if gen > trans:
        ax.annotate('', xy=(i - width/2, gen - 0.01), xytext=(i + width/2, trans + 0.01),
                    arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.7, lw=2))
        mid_point = (gen + trans) / 2
        ax.text(i, mid_point, f'+{gen - trans:.3f}', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7),
                fontsize=14, fontweight='bold')
    else:
        ax.annotate('', xy=(i + width/2, trans - 0.01), xytext=(i - width/2, gen + 0.01),
                    arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.7, lw=2))
        mid_point = (gen + trans) / 2
        ax.text(i, mid_point, f'+{trans - gen:.3f}', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7),
                fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("Cross-model_results.pdf", bbox_inches="tight")
plt.show()
