import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#read outputs form gpt and phi models
df1 = pd.read_csv('unsloth_Phi-4-mini-instruct-GGUF_Phi-4-mini-instruct-Q2_K_L.gguf_word_benchmark.csv',encoding="utf-8")
df2 = pd.read_csv('medieval_irish_words_edil_chatgpt_new.csv',encoding="utf-8")



original_meanings = " ".join(str(x) for x in df1.iloc[:,1].dropna())
phi_meanings = " ".join(str(x) for x in df1.iloc[:,2].dropna())
gpt_meanings = " ".join(str(x) for x in df2.iloc[:,2].dropna())

# create word clouds
wc1 = WordCloud(width=800, height=400, background_color="white").generate(original_meanings)
wc2 = WordCloud(width=800, height=400, background_color="white").generate(phi_meanings)
wc3 = WordCloud(width=800, height=400, background_color="white").generate(gpt_meanings)

fig, axes = plt.subplots(1, 3, figsize=(30, 10))
plt.subplots_adjust(wspace=0.3) 

axes[0].imshow(wc1, interpolation="bilinear")
axes[0].set_title("eDIL Original Meanings", fontsize=18)
axes[0].axis("off")

axes[1].imshow(wc3, interpolation="bilinear")
axes[1].set_title("GPT-4o Meanings", fontsize=18)
axes[1].axis("off")

axes[2].imshow(wc2, interpolation="bilinear")
axes[2].set_title("Phi-4-mini-instruct-Q2_K_L Meanings", fontsize=18)
axes[2].axis("off")

plt.tight_layout()
plt.savefig("wordcloud_comparison.png", dpi=300)
plt.show()