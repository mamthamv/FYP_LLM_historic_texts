import pandas as pd
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# Load Excel file
df = pd.read_excel("words_exported.xlsx")

results=[]
checked_words = set()


def fetch_meaning(word):
  response =  requests.get(f'https://www.dil.ie/search?q={word}&search_in=headword')
  content = BeautifulSoup(response.content, 'html.parser')
  paras = content.find_all("p", class_="result")
  return paras if paras else None

def process_word(word):
  if word in checked_words:
    return
  checked_words.add(word)

  paras = fetch_meaning(word)
  if not paras:
    print(f"{word} not found")
    return

  print(f"Processing word {word}")
  translations = []
  for p in paras:
    if p.find_all("span", class_="trans"):
        trans_spans = p.find_all("span", class_="trans")
        for span in trans_spans:
          translations.append((span.get_text()).strip())
  if translations:
    results.append({"Word": word, "Meaning": translations})

for word in df["Word"]:
    process_word(word)

df = pd.DataFrame(results)
df.to_excel("medieval_irish_words_edil.xlsx", index=False)
print("Saved to medieval_irish_words_final.xlsx")