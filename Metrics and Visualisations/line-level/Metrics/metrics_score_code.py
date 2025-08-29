import os
import pandas as pd
import json
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
from sentence_transformers import SentenceTransformer, util

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

directory = os.path.join("..", "model_outputs")

all_scores = {}
smooth_fn = SmoothingFunction().method1  # BLEU smoothing

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath, encoding='utf-8')

        print(f"\nProcessing file: {filepath}")

        # References for BLEU
        refs_tokens = [[word_tokenize(str(ref))] for ref in df.iloc[:, 1].astype(str).tolist()]
        # References for METEOR
        refs_text = df.iloc[:, 1].astype(str).fillna("").tolist()
        # References for SENTENCEBERT
        ref_embeddings = model.encode(refs_text, convert_to_tensor=True)

        file_scores = {}

        for col in df.columns[2:]:
            # Prepare hypotheses
            hyps_text = df[col].astype(str).fillna("").tolist()

            # BLUE
            hyps_tokens = [
                word_tokenize(h) if h.strip() and h.lower() != "nan" else []
                for h in hyps_text
            ]
            try:
                bleu_val = corpus_bleu(refs_tokens, hyps_tokens, smoothing_function=smooth_fn)
            except Exception as e:
                print(f"Error computing BLEU for {col}: {e}")
                bleu_val = 0.0

            # METEOR
            meteor_scores = []
            for r, h in zip(refs_text, hyps_text):
                if h.strip() == "" or h.lower() == "nan":
                    meteor_scores.append(0.0)
                else:
                    meteor_scores.append(meteor_score([word_tokenize(r)], word_tokenize(h)))
            meteor_val = sum(meteor_scores) / len(meteor_scores)

            # Sentence bert 
            hyp_embeddings = model.encode(hyps_text, convert_to_tensor=True)
            cosine_similarities = util.cos_sim(ref_embeddings, hyp_embeddings)
            avg_cosine = float(cosine_similarities.diag().cpu().numpy().mean())

            # Store all three
            file_scores[col] = {
                "bleu": bleu_val,
                "meteor": meteor_val,
                "cosine_similarity": avg_cosine
            }

            print(f"{col} <--> BLEU: {bleu_val:.4f}, METEOR: {meteor_val:.4f}, Cosine: {avg_cosine:.4f}")

        all_scores[filename] = file_scores

# Save JSON
output_path = "all_metrics_scores.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_scores, f, indent=4)

print("similarity scores calculation and saving completed.")
