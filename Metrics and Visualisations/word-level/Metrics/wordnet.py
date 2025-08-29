import os
import pandas as pd
import json
import re
from ast import literal_eval
from nltk.corpus import wordnet as wn
import nltk

# Download WordNet
nltk.download('wordnet')
nltk.download('omw-1.4')


# WordNet Similarity Function
def wordnet_similarity(list1, list2):
    if not list1 and not list2:
        return 1.0
    if not list1 or not list2:
        return 0.0

    def avg_max_similarity(source, target):
        sims = []
        for w1 in source:
            syn1 = wn.synsets(w1)
            if not syn1:
                continue
            max_sim = 0
            for w2 in target:
                syn2 = wn.synsets(w2)
                if not syn2:
                    continue
                for s1 in syn1:
                    for s2 in syn2:
                        sim = s1.wup_similarity(s2) or 0
                        if sim > max_sim:
                            max_sim = sim
            sims.append(max_sim)
        return sum(sims) / len(sims) if sims else 0.0

    # similarity on both the directions
    return (avg_max_similarity(list1, list2) + avg_max_similarity(list2, list1)) / 2

# Process raw meaning column
def process_meaning(raw):
    if pd.isna(raw):
        return []
    try:
        items = literal_eval(raw) if isinstance(raw, str) and raw.strip().startswith("[") else [raw]
    except Exception:
        items = [raw]

    result = []
    for item in items:
        item = re.sub(r'\(.*?\)', '', str(item))
        item = item.replace("'", "").replace('"', "")
        splits = re.split(r'[;,]', item)
        for split in splits:
            token = split.strip().lower()
            if token:
                result.append(token)
    return result

# Process predictions
def process_prediction(pred):
    if pred is None or (isinstance(pred, float) and pd.isna(pred)):
        return []

    try:
        if isinstance(pred, str) and pred.strip().startswith("[") and pred.strip().endswith("]"):
            items = literal_eval(pred)
        else:
            items = [pred]
    except Exception:
        items = [pred]

    result = []
    for item in items:
        item = re.sub(r'\(.*?\)', '', str(item))
        item = re.sub(r'<.*?>', '', item)
        item = item.replace("```", "").replace('"', "").replace("'", "")
        splits = re.split(r'[;,]', item)
        for split in splits:
            token = split.strip().lower()
            if token:
                result.append(token)
    return result

directory = 'C:/Users/Mamtha/Downloads/results_models/word-level'
all_similarity_scores = {}

for filename in os.listdir(directory):
    if not filename.endswith('.csv'):
        continue

    filepath = os.path.join(directory, filename)
    df = pd.read_csv(filepath, encoding='utf-8')

    print(f"\nProcessing file: {filepath}")

    # Process references
    df['ProcessedMeaning'] = df.iloc[:, 1].apply(process_meaning)
    references = df['ProcessedMeaning'].tolist()
    file_scores = {}

    # For each prediction column
    for col in df.columns[2:-1]:
        predictions = df[col].tolist()
        wordnet_scores = []

        for ref_tokens, pred_raw in zip(references[1:], predictions[1:]):
            pred_tokens = process_prediction(pred_raw)
            score = wordnet_similarity(ref_tokens, pred_tokens)
            wordnet_scores.append(score)

        avg_score = sum(wordnet_scores) / len(wordnet_scores) if wordnet_scores else 0.0
        file_scores[col] = avg_score*100
        print(f"Average WordNet Similarity for {col}: {avg_score:.4f}")


    all_similarity_scores[filename] = file_scores



# Save results
output_path = 'C:/Users/Mamtha/Downloads/wordnet_similarity_summary_20aug.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_similarity_scores, f, indent=4)
print("\nWordNet similarity scores saved successfully.")
